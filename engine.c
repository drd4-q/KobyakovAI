#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================
// Конфигурация модели (Иерархический MoE из 144 нод)
// ============================================================
typedef struct {
    int vocab_size;
    int block_size;
    int n_embd;
    int n_layer;
    int n_head;
    int num_parents;      // 4
    int num_sub_parents;  // 4
    int num_leaf_experts; // 9 (всего 4 * 4 * 9 = 144 ноды)
    int top_k;            // 2
    int mult;             // 2 (hidden = mult * n_embd)
} Config;

typedef struct {
    float* w1; // [mult * n_embd, n_embd]
    float* b1; // [mult * n_embd]
    float* w2; // [n_embd, mult * n_embd]
    float* b2; // [n_embd]
} ExpertWeights;

typedef struct {
    float* gate_w; // [num_leaf_experts, n_embd]
    float* gate_b; // [num_leaf_experts]
    ExpertWeights* experts; // [num_leaf_experts] (9 экспертов)
} SubParentWeights;

typedef struct {
    float* gate_w; // [num_sub_parents, n_embd]
    float* gate_b; // [num_sub_parents]
    SubParentWeights* sub_parents; // [num_sub_parents] (4 сабродителя)
} ParentWeights;

typedef struct {
    float* sa_ln_w;
    float* sa_ln_b;
    float* sa_wq;
    float* sa_wk;
    float* sa_wv;
    float* sa_wo;
    float* moe_ln_w;
    float* moe_ln_b;
    float* accuracy_gate_w; // [num_parents, n_embd]
    float* accuracy_gate_b; // [num_parents]
    ParentWeights* parents; // [num_parents] (4 родителя)
} LayerWeights;

typedef struct {
    Config config;
    float* token_embedding_table;
    float* position_embedding_table;
    LayerWeights* layers;
    float* ln_f_w;
    float* ln_f_b;
    float* lm_head;
    float* raw_data;
} MoEModel;

typedef struct {
    float* x;
    float* xb;
    float* q;
    float* k;
    float* v;
    float* att;
    float* key_cache;
    float* val_cache;
    float* parent_acc_logits; // [4]
    float* sub_logits;        // [4]
    float* leaf_logits;       // [9]
    float* sub_acc;           // [n_embd]
    float* parent_acc;        // [n_embd]
    float* exp_h;             // [mult * n_embd]
    float* exp_out;           // [n_embd]
    float* moe_acc;           // [n_embd]
    float* logits;            // [vocab_size]
} RunState;

typedef struct {
    char** vocab;
    int* vocab_lens;
    int vocab_size;
} Tokenizer;

// ============================================================
// Математические операторы (Чистый ISO C99: x86, ARM, RISC-V)
// ============================================================

void layernorm(float* out, float* x, float* w, float* b, int size) {
    float mean = 0.0f;
    for (int i = 0; i < size; i++) mean += x[i];
    mean /= size;

    float variance = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = x[i] - mean;
        variance += diff * diff;
    }
    variance /= size;
    float inv_std = 1.0f / sqrtf(variance + 1e-5f);

    for (int i = 0; i < size; i++) {
        out[i] = ((x[i] - mean) * inv_std) * w[i] + b[i];
    }
}

void matmul(float* out, float* x, float* w, int d_in, int d_out) {
    for (int i = 0; i < d_out; i++) {
        float val = 0.0f;
        const float* w_row = w + i * d_in;
        for (int j = 0; j < d_in; j++) {
            val += w_row[j] * x[j];
        }
        out[i] = val;
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / (sum + 1e-8f);
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

void gelu(float* x, int size) {
    const float k = sqrtf(2.0f / (float)M_PI);
    for (int i = 0; i < size; i++) {
        float xi = x[i];
        float cube = 0.044715f * xi * xi * xi;
        x[i] = 0.5f * xi * (1.0f + tanhf(k * (xi + cube)));
    }
}

// ============================================================
// Загрузка модели и памяти
// ============================================================

int load_model(const char* checkpoint_path, MoEModel* model) {
    FILE* f = fopen(checkpoint_path, "rb");
    if (!f) return 0;

    if (fread(&model->config, sizeof(Config), 1, f) != 1) {
        fclose(f);
        return 0;
    }

    Config c = model->config;
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    long weights_bytes = file_size - sizeof(Config);
    fseek(f, sizeof(Config), SEEK_SET);

    model->raw_data = (float*)malloc(weights_bytes);
    if (!model->raw_data) {
        fclose(f);
        return 0;
    }

    if (fread(model->raw_data, 1, weights_bytes, f) != (size_t)weights_bytes) {
        free(model->raw_data);
        fclose(f);
        return 0;
    }
    fclose(f);

    float* ptr = model->raw_data;

    model->token_embedding_table = ptr; ptr += (size_t)c.vocab_size * c.n_embd;
    model->position_embedding_table = ptr; ptr += (size_t)c.block_size * c.n_embd;

    model->layers = (LayerWeights*)malloc(c.n_layer * sizeof(LayerWeights));
    for (int l = 0; l < c.n_layer; l++) {
        LayerWeights* lw = &model->layers[l];
        lw->sa_ln_w = ptr; ptr += c.n_embd;
        lw->sa_ln_b = ptr; ptr += c.n_embd;
        lw->sa_wq = ptr; ptr += c.n_embd * c.n_embd;
        lw->sa_wk = ptr; ptr += c.n_embd * c.n_embd;
        lw->sa_wv = ptr; ptr += c.n_embd * c.n_embd;
        lw->sa_wo = ptr; ptr += c.n_embd * c.n_embd;
        lw->moe_ln_w = ptr; ptr += c.n_embd;
        lw->moe_ln_b = ptr; ptr += c.n_embd;

        // Master Accuracy Gate
        lw->accuracy_gate_w = ptr; ptr += c.num_parents * c.n_embd;
        lw->accuracy_gate_b = ptr; ptr += c.num_parents;

        lw->parents = (ParentWeights*)malloc(c.num_parents * sizeof(ParentWeights));
        for (int p_idx = 0; p_idx < c.num_parents; p_idx++) {
            ParentWeights* pw = &lw->parents[p_idx];
            pw->gate_w = ptr; ptr += c.num_sub_parents * c.n_embd;
            pw->gate_b = ptr; ptr += c.num_sub_parents;

            pw->sub_parents = (SubParentWeights*)malloc(c.num_sub_parents * sizeof(SubParentWeights));
            for (int s_idx = 0; s_idx < c.num_sub_parents; s_idx++) {
                SubParentWeights* spw = &pw->sub_parents[s_idx];
                spw->gate_w = ptr; ptr += c.num_leaf_experts * c.n_embd;
                spw->gate_b = ptr; ptr += c.num_leaf_experts;

                spw->experts = (ExpertWeights*)malloc(c.num_leaf_experts * sizeof(ExpertWeights));
                for (int e = 0; e < c.num_leaf_experts; e++) {
                    ExpertWeights* ew = &spw->experts[e];
                    int hidden = c.mult * c.n_embd;
                    ew->w1 = ptr; ptr += (size_t)hidden * c.n_embd;
                    ew->b1 = ptr; ptr += hidden;
                    ew->w2 = ptr; ptr += (size_t)c.n_embd * hidden;
                    ew->b2 = ptr; ptr += c.n_embd;
                }
            }
        }
    }

    model->ln_f_w = ptr; ptr += c.n_embd;
    model->ln_f_b = ptr; ptr += c.n_embd;
    model->lm_head = ptr; ptr += (size_t)c.vocab_size * c.n_embd;

    return 1;
}

void init_run_state(RunState* s, Config* c) {
    int hidden = c->mult * c->n_embd;
    s->x = (float*)calloc(c->n_embd, sizeof(float));
    s->xb = (float*)calloc(c->n_embd, sizeof(float));
    s->q = (float*)calloc(c->n_embd, sizeof(float));
    s->k = (float*)calloc(c->n_embd, sizeof(float));
    s->v = (float*)calloc(c->n_embd, sizeof(float));
    s->att = (float*)calloc(c->n_head * c->block_size, sizeof(float));
    s->key_cache = (float*)calloc((size_t)c->n_layer * c->block_size * c->n_embd, sizeof(float));
    s->val_cache = (float*)calloc((size_t)c->n_layer * c->block_size * c->n_embd, sizeof(float));
    s->parent_acc_logits = (float*)calloc(c->num_parents, sizeof(float));
    s->sub_logits = (float*)calloc(c->num_sub_parents, sizeof(float));
    s->leaf_logits = (float*)calloc(c->num_leaf_experts, sizeof(float));
    s->sub_acc = (float*)calloc(c->n_embd, sizeof(float));
    s->parent_acc = (float*)calloc(c->n_embd, sizeof(float));
    s->exp_h = (float*)calloc(hidden, sizeof(float));
    s->exp_out = (float*)calloc(c->n_embd, sizeof(float));
    s->moe_acc = (float*)calloc(c->n_embd, sizeof(float));
    s->logits = (float*)calloc(c->vocab_size, sizeof(float));
}

// ============================================================
// Прямой проход (Forward Pass: Self-Attention + Hierarchical MoE)
// ============================================================

float* forward(MoEModel* model, RunState* s, int token, int pos) {
    Config* p = &model->config;
    int d = p->n_embd;
    int head_size = d / p->n_head;

    float* token_emb = model->token_embedding_table + token * d;
    float* pos_emb = model->position_embedding_table + pos * d;
    for (int i = 0; i < d; i++) {
        s->x[i] = token_emb[i] + pos_emb[i];
    }

    for (int l = 0; l < p->n_layer; l++) {
        LayerWeights* lw = &model->layers[l];

        // 1. Self-Attention
        layernorm(s->xb, s->x, lw->sa_ln_w, lw->sa_ln_b, d);

        matmul(s->q, s->xb, lw->sa_wq, d, d);
        matmul(s->k, s->xb, lw->sa_wk, d, d);
        matmul(s->v, s->xb, lw->sa_wv, d, d);

        int cache_offset = (l * p->block_size + pos) * d;
        memcpy(s->key_cache + cache_offset, s->k, d * sizeof(float));
        memcpy(s->val_cache + cache_offset, s->v, d * sizeof(float));

        float* sa_out = s->xb;
        memset(sa_out, 0, d * sizeof(float));

        for (int h = 0; h < p->n_head; h++) {
            float* q_head = s->q + h * head_size;
            float* att_head = s->att + h * p->block_size;

            for (int t = 0; t <= pos; t++) {
                int k_offset = (l * p->block_size + t) * d + h * head_size;
                float* k_head = s->key_cache + k_offset;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q_head[i] * k_head[i];
                }
                score /= sqrtf((float)head_size);
                att_head[t] = score;
            }

            softmax(att_head, pos + 1);

            for (int t = 0; t <= pos; t++) {
                float a = att_head[t];
                int v_offset = (l * p->block_size + t) * d + h * head_size;
                float* v_head = s->val_cache + v_offset;
                for (int i = 0; i < head_size; i++) {
                    sa_out[h * head_size + i] += a * v_head[i];
                }
            }
        }

        matmul(s->q, sa_out, lw->sa_wo, d, d);
        for (int i = 0; i < d; i++) {
            s->x[i] += s->q[i];
        }

        // 2. Hierarchical MoE (144 эксперта через 4 параллельных родителя)
        layernorm(s->xb, s->x, lw->moe_ln_w, lw->moe_ln_b, d);

        // a) Master Accuracy Gate: оцениваем точность 4 родителей
        matmul(s->parent_acc_logits, s->xb, lw->accuracy_gate_w, d, p->num_parents);
        for (int e = 0; e < p->num_parents; e++) {
            s->parent_acc_logits[e] += lw->accuracy_gate_b[e];
        }
        softmax(s->parent_acc_logits, p->num_parents); // [w0, w1, w2, w3]

        memset(s->moe_acc, 0, d * sizeof(float));

        // b) 4 параллельные родительские ноды
        for (int p_idx = 0; p_idx < p->num_parents; p_idx++) {
            float parent_w = s->parent_acc_logits[p_idx];
            ParentWeights* parent = &lw->parents[p_idx];

            // Маршрутизация сабродителей
            matmul(s->sub_logits, s->xb, parent->gate_w, d, p->num_sub_parents);
            for (int s_idx = 0; s_idx < p->num_sub_parents; s_idx++) {
                s->sub_logits[s_idx] += parent->gate_b[s_idx];
            }
            softmax(s->sub_logits, p->num_sub_parents);

            // Выбор Top-K сабродителей (top-2)
            int top_subs[4];
            float top_sub_weights[4];
            int sub_k = (p->top_k < p->num_sub_parents) ? p->top_k : p->num_sub_parents;
            for (int k = 0; k < sub_k; k++) {
                int best_idx = -1;
                float best_val = -1e9f;
                for (int s_idx = 0; s_idx < p->num_sub_parents; s_idx++) {
                    int already = 0;
                    for (int prev = 0; prev < k; prev++) {
                        if (top_subs[prev] == s_idx) already = 1;
                    }
                    if (!already && s->sub_logits[s_idx] > best_val) {
                        best_val = s->sub_logits[s_idx];
                        best_idx = s_idx;
                    }
                }
                top_subs[k] = best_idx;
                top_sub_weights[k] = best_val;
            }
            float sub_sum = 0.0f;
            for (int k = 0; k < sub_k; k++) sub_sum += top_sub_weights[k];
            for (int k = 0; k < sub_k; k++) top_sub_weights[k] /= (sub_sum + 1e-8f);

            memset(s->parent_acc, 0, d * sizeof(float));

            for (int k = 0; k < sub_k; k++) {
                int s_idx = top_subs[k];
                float sub_w = top_sub_weights[k];
                SubParentWeights* sub = &parent->sub_parents[s_idx];

                // Маршрутизация листовых экспертов (9 экспертов)
                matmul(s->leaf_logits, s->xb, sub->gate_w, d, p->num_leaf_experts);
                for (int l_idx = 0; l_idx < p->num_leaf_experts; l_idx++) {
                    s->leaf_logits[l_idx] += sub->gate_b[l_idx];
                }
                softmax(s->leaf_logits, p->num_leaf_experts);

                // Выбор Top-K листовых экспертов (top-2)
                int top_leaves[8];
                float top_leaf_weights[8];
                int leaf_k = (p->top_k < p->num_leaf_experts) ? p->top_k : p->num_leaf_experts;
                for (int lk = 0; lk < leaf_k; lk++) {
                    int best_idx = -1;
                    float best_val = -1e9f;
                    for (int l_idx = 0; l_idx < p->num_leaf_experts; l_idx++) {
                        int already = 0;
                        for (int prev = 0; prev < lk; prev++) {
                            if (top_leaves[prev] == l_idx) already = 1;
                        }
                        if (!already && s->leaf_logits[l_idx] > best_val) {
                            best_val = s->leaf_logits[l_idx];
                            best_idx = l_idx;
                        }
                    }
                    top_leaves[lk] = best_idx;
                    top_leaf_weights[lk] = best_val;
                }
                float leaf_sum = 0.0f;
                for (int lk = 0; lk < leaf_k; lk++) leaf_sum += top_leaf_weights[lk];
                for (int lk = 0; lk < leaf_k; lk++) top_leaf_weights[lk] /= (leaf_sum + 1e-8f);

                memset(s->sub_acc, 0, d * sizeof(float));

                // Вычисление выбранных листовых экспертов
                int hidden = p->mult * d;
                for (int lk = 0; lk < leaf_k; lk++) {
                    int exp_id = top_leaves[lk];
                    float exp_w = top_leaf_weights[lk];
                    ExpertWeights* ew = &sub->experts[exp_id];

                    matmul(s->exp_h, s->xb, ew->w1, d, hidden);
                    for (int i = 0; i < hidden; i++) s->exp_h[i] += ew->b1[i];
                    gelu(s->exp_h, hidden);

                    matmul(s->exp_out, s->exp_h, ew->w2, hidden, d);
                    for (int i = 0; i < d; i++) {
                        s->sub_acc[i] += exp_w * (s->exp_out[i] + ew->b2[i]);
                    }
                }

                for (int i = 0; i < d; i++) {
                    s->parent_acc[i] += sub_w * s->sub_acc[i];
                }
            }

            // Взвешенное суммирование через коэффициент точности родителя
            for (int i = 0; i < d; i++) {
                s->moe_acc[i] += parent_w * s->parent_acc[i];
            }
        }

        for (int i = 0; i < d; i++) {
            s->x[i] += s->moe_acc[i];
        }
    }

    layernorm(s->xb, s->x, model->ln_f_w, model->ln_f_b, d);
    matmul(s->logits, s->xb, model->lm_head, d, p->vocab_size);

    return s->logits;
}

// ============================================================
// Токенизатор и сэмплирование
// ============================================================

int load_tokenizer(const char* tok_path, Tokenizer* tok) {
    FILE* f = fopen(tok_path, "rb");
    if (!f) return 0;

    if (fread(&tok->vocab_size, sizeof(int), 1, f) != 1) {
        fclose(f);
        return 0;
    }

    tok->vocab = (char**)malloc(tok->vocab_size * sizeof(char*));
    tok->vocab_lens = (int*)malloc(tok->vocab_size * sizeof(int));

    for (int i = 0; i < tok->vocab_size; i++) {
        int len;
        if (fread(&len, sizeof(int), 1, f) != 1) break;
        tok->vocab_lens[i] = len;
        tok->vocab[i] = (char*)malloc(len + 1);
        if (len > 0) {
            fread(tok->vocab[i], 1, len, f);
        }
        tok->vocab[i][len] = '\0';
    }
    fclose(f);
    return 1;
}

int encode_prompt(Tokenizer* tok, const char* text, int* out_tokens, int max_tokens) {
    int n = 0;
    int len = (int)strlen(text);
    int i = 0;

    while (i < len && n < max_tokens) {
        int best_id = -1;
        int best_len = 0;

        for (int v = 0; v < tok->vocab_size; v++) {
            int vlen = tok->vocab_lens[v];
            if (vlen > 0 && vlen <= (len - i)) {
                if (memcmp(text + i, tok->vocab[v], (size_t)vlen) == 0) {
                    if (vlen > best_len) {
                        best_len = vlen;
                        best_id = v;
                    }
                }
            }
        }

        if (best_id >= 0 && best_len > 0) {
            out_tokens[n++] = best_id;
            i += best_len;
        } else {
            unsigned char b = (unsigned char)text[i];
            for (int v = 0; v < tok->vocab_size; v++) {
                if (tok->vocab_lens[v] == 1 && (unsigned char)tok->vocab[v][0] == b) {
                    best_id = v;
                    break;
                }
            }
            if (best_id >= 0) {
                out_tokens[n++] = best_id;
            }
            i++;
        }
    }
    return n;
}

int sample_token(float* logits, int vocab_size, float temperature, int top_k, const int* recent_tokens, int num_recent, float repeat_penalty) {
    if (repeat_penalty > 1.0f && num_recent > 0) {
        for (int r = 0; r < num_recent; r++) {
            int tok_id = recent_tokens[r];
            if (tok_id >= 0 && tok_id < vocab_size) {
                if (logits[tok_id] > 0.0f) {
                    logits[tok_id] /= repeat_penalty;
                } else {
                    logits[tok_id] *= repeat_penalty;
                }
            }
        }
    }

    if (temperature <= 0.01f) {
        int best_idx = 0;
        float best_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_idx = i;
            }
        }
        return best_idx;
    }

    if (top_k > 0 && top_k < vocab_size) {
        int top_indices[64];
        if (top_k > 64) top_k = 64;

        for (int k = 0; k < top_k; k++) {
            int best_idx = -1;
            float best_val = -1e9f;
            for (int i = 0; i < vocab_size; i++) {
                int already = 0;
                for (int p = 0; p < k; p++) {
                    if (top_indices[p] == i) { already = 1; break; }
                }
                if (!already && logits[i] > best_val) {
                    best_val = logits[i];
                    best_idx = i;
                }
            }
            top_indices[k] = best_idx;
        }

        for (int i = 0; i < vocab_size; i++) {
            int is_top = 0;
            for (int k = 0; k < top_k; k++) {
                if (top_indices[k] == i) { is_top = 1; break; }
            }
            if (!is_top) logits[i] = -1e9f;
        }
    }

    for (int i = 0; i < vocab_size; i++) logits[i] /= temperature;
    softmax(logits, vocab_size);

    float r = ((float)rand() / (float)RAND_MAX);
    float cdf = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        cdf += logits[i];
        if (r <= cdf) return i;
    }
    return vocab_size - 1;
}

// ============================================================
// Генератор ответов (CLI & Interactive)
// ============================================================

void generate_response(MoEModel* model, Tokenizer* tok, RunState* state, const char* input_text) {
    char input_buf[1024];
    strncpy(input_buf, input_text, sizeof(input_buf) - 1);
    input_buf[sizeof(input_buf) - 1] = '\0';

    size_t slen = strlen(input_buf);
    while (slen > 0 && (input_buf[slen - 1] == '\n' || input_buf[slen - 1] == '\r')) {
        input_buf[--slen] = '\0';
    }

    // Удаление UTF-8 BOM маркера (если вывод передан через PowerShell pipe)
    char* clean_input = input_buf;
    if ((unsigned char)clean_input[0] == 0xEF && 
        (unsigned char)clean_input[1] == 0xBB && 
        (unsigned char)clean_input[2] == 0xBF) {
        clean_input += 3;
    }
    while (*clean_input == ' ' || *clean_input == '\t') clean_input++;

    if (strlen(clean_input) == 0) return;

    int prompt_tokens[256];
    int num_prompt_tokens = 0;
    // 1. Префикс: "User:"
    prompt_tokens[num_prompt_tokens++] = 12982; // 'User'
    prompt_tokens[num_prompt_tokens++] = 25;    // ':'

    // 2. Ввод пользователя с ведущим пробелом для точного выравнивания GPT-2 BPE
    char spaced_input[1024];
    snprintf(spaced_input, sizeof(spaced_input), " %s", clean_input);
    int n_user = encode_prompt(tok, spaced_input, prompt_tokens + num_prompt_tokens, 200);
    num_prompt_tokens += n_user;

    // 3. Суффикс: "\n\nKobyakovAI:\n"
    const int suffix[8] = {198, 198, 42, 26730, 44715, 20185, 25, 198};
    for (int s = 0; s < 8; s++) {
        prompt_tokens[num_prompt_tokens++] = suffix[s];
    }

    printf("\n🤖 \033[96mKobyakovAI:\033[0m\n");
    fflush(stdout);

    clock_t start = clock();

    int current_token = prompt_tokens[0];
    int pos = 0;

    for (int p = 0; p < num_prompt_tokens - 1 && p < model->config.block_size - 1; p++) {
        forward(model, state, prompt_tokens[p], pos++);
    }
    current_token = prompt_tokens[num_prompt_tokens - 1];

    int max_new_tokens = 150;
    int generated_count = 0;
    int recent_tokens[64];
    int num_recent = 0;

    for (; generated_count < max_new_tokens && pos < model->config.block_size - 1; pos++, generated_count++) {
        float* logits = forward(model, state, current_token, pos);
        current_token = sample_token(logits, model->config.vocab_size, 0.0f, 0, recent_tokens, num_recent, 1.1f);

        // Остановка при маркере конца текста (50256) или начале нового раунда User: (12982)
        if (current_token == 50256 || current_token == 12982) break;

        if (current_token < tok->vocab_size) {
            const char* piece = tok->vocab[current_token];
            if (strstr(piece, "User:") != NULL) break;
            printf("%s", piece);
            fflush(stdout);
        }

        recent_tokens[num_recent % 64] = current_token;
        if (num_recent < 64) num_recent++;
    }

    printf("\n");
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double tps = (elapsed > 0.0001) ? (generated_count / elapsed) : 0.0;
    printf("\033[90m[Generated %d tokens in %.2fs (%.1f tok/s)]\033[0m\n", generated_count, elapsed, tps);
    printf("------------------------------------------------------------\n");
}

// ============================================================
// Интерактивный генератор (Интерпретатор)
// ============================================================

void run_interactive(MoEModel* model, Tokenizer* tok) {
    RunState state;
    init_run_state(&state, &model->config);

    char input_buf[1024];
    int total_leaf = model->config.num_parents * model->config.num_sub_parents * model->config.num_leaf_experts;

    printf("============================================================\n");
    printf("  🤖 KobyakovAI - Hierarchical MoE Shell (Pure C / Zig)\n");
    printf("  Nodes: %d Leaf Experts (4 Parents x 4 Subs x 9 Leaves)\n", total_leaf);
    printf("  Portability: x86_64, ARM64 (Apple/Pi), RISC-V Compatible\n");
    printf("  Type your prompt and press Enter.\n");
    printf("  Type 'exit' or 'quit' to exit.\n");
    printf("============================================================\n\n");

    while (1) {
        printf("\n\033[1mUser:\033[0m ");
        fflush(stdout);

        if (!fgets(input_buf, sizeof(input_buf), stdin)) break;

        char* trimmed = input_buf;
        while (*trimmed == ' ' || *trimmed == '\t' || *trimmed == '\r' || *trimmed == '\n') trimmed++;
        if (strcmp(trimmed, "exit") == 0 || strcmp(trimmed, "quit") == 0) {
            printf("Goodbye!\n");
            break;
        }

        generate_response(model, tok, &state, input_buf);
    }
}

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif

    srand((unsigned int)time(NULL));

    const char* model_path = (argc > 1) ? argv[1] : "model.bin";
    const char* tok_path = (argc > 2) ? argv[2] : "tokenizer.bin";

    MoEModel model;
    if (!load_model(model_path, &model)) {
        printf("Error: failed to load %s\n", model_path);
        return 1;
    }

    Tokenizer tok;
    if (!load_tokenizer(tok_path, &tok)) {
        printf("Error: failed to load %s\n", tok_path);
        return 1;
    }

    if (argc > 3) {
        // Одиночный запуск промпта из аргументов командной строки
        RunState state;
        init_run_state(&state, &model.config);
        generate_response(&model, &tok, &state, argv[3]);
    } else {
        // Интерактивный режим
        run_interactive(&model, &tok);
    }

    return 0;
}
