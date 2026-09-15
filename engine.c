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
// Конфигурация модели
// ============================================================
typedef struct {
    int vocab_size;
    int block_size;
    int n_embd;
    int n_layer;
    int n_head;
    int num_experts;
    int top_k;
} Config;

typedef struct {
    float* w1; // [4 * n_embd, n_embd]
    float* b1; // [4 * n_embd]
    float* w2; // [n_embd, 4 * n_embd]
    float* b2; // [n_embd]
} ExpertWeights;

typedef struct {
    float* sa_ln_w;
    float* sa_ln_b;
    float* sa_wq;
    float* sa_wk;
    float* sa_wv;
    float* sa_wo;
    float* moe_ln_w;
    float* moe_ln_b;
    float* moe_gate_w;
    float* moe_gate_b;
    ExpertWeights* experts;
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
    float* gate_logits;
    float* exp_h;
    float* exp_out;
    float* moe_acc;
    float* logits;
} RunState;

typedef struct {
    char** vocab;
    int* vocab_lens;
    int vocab_size;
} Tokenizer;

// ============================================================
// Математические операторы
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
        lw->moe_gate_w = ptr; ptr += c.num_experts * c.n_embd;
        lw->moe_gate_b = ptr; ptr += c.num_experts;

        lw->experts = (ExpertWeights*)malloc(c.num_experts * sizeof(ExpertWeights));
        for (int e = 0; e < c.num_experts; e++) {
            ExpertWeights* ew = &lw->experts[e];
            ew->w1 = ptr; ptr += 4 * c.n_embd * c.n_embd;
            ew->b1 = ptr; ptr += 4 * c.n_embd;
            ew->w2 = ptr; ptr += c.n_embd * 4 * c.n_embd;
            ew->b2 = ptr; ptr += c.n_embd;
        }
    }

    model->ln_f_w = ptr; ptr += c.n_embd;
    model->ln_f_b = ptr; ptr += c.n_embd;
    model->lm_head = ptr; ptr += (size_t)c.vocab_size * c.n_embd;

    return 1;
}

void init_run_state(RunState* s, Config* c) {
    s->x = (float*)calloc(c->n_embd, sizeof(float));
    s->xb = (float*)calloc(c->n_embd, sizeof(float));
    s->q = (float*)calloc(c->n_embd, sizeof(float));
    s->k = (float*)calloc(c->n_embd, sizeof(float));
    s->v = (float*)calloc(c->n_embd, sizeof(float));
    s->att = (float*)calloc(c->n_head * c->block_size, sizeof(float));
    s->key_cache = (float*)calloc((size_t)c->n_layer * c->block_size * c->n_embd, sizeof(float));
    s->val_cache = (float*)calloc((size_t)c->n_layer * c->block_size * c->n_embd, sizeof(float));
    s->gate_logits = (float*)calloc(c->num_experts, sizeof(float));
    s->exp_h = (float*)calloc(4 * c->n_embd, sizeof(float));
    s->exp_out = (float*)calloc(c->n_embd, sizeof(float));
    s->moe_acc = (float*)calloc(c->n_embd, sizeof(float));
    s->logits = (float*)calloc(c->vocab_size, sizeof(float));
}

// ============================================================
// Forward Pass
// ============================================================

float* forward(MoEModel* model, RunState* s, int token, int pos) {
    Config* p = &model->config;
    int d = p->n_embd;
    int head_size = d / p->n_head;

    float* tok_vec = model->token_embedding_table + token * d;
    float* pos_vec = model->position_embedding_table + (pos % p->block_size) * d;
    for (int i = 0; i < d; i++) {
        s->x[i] = tok_vec[i] + pos_vec[i];
    }

    for (int l = 0; l < p->n_layer; l++) {
        LayerWeights* lw = &model->layers[l];

        // Self-Attention
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

        // Sparse MoE
        layernorm(s->xb, s->x, lw->moe_ln_w, lw->moe_ln_b, d);

        matmul(s->gate_logits, s->xb, lw->moe_gate_w, d, p->num_experts);
        for (int e = 0; e < p->num_experts; e++) {
            s->gate_logits[e] += lw->moe_gate_b[e];
        }
        softmax(s->gate_logits, p->num_experts);

        int top_indices[8];
        float top_weights[8];
        for (int k = 0; k < p->top_k; k++) {
            int best_idx = -1;
            float best_val = -1e9f;
            for (int e = 0; e < p->num_experts; e++) {
                int already_chosen = 0;
                for (int prev = 0; prev < k; prev++) {
                    if (top_indices[prev] == e) already_chosen = 1;
                }
                if (!already_chosen && s->gate_logits[e] > best_val) {
                    best_val = s->gate_logits[e];
                    best_idx = e;
                }
            }
            top_indices[k] = best_idx;
            top_weights[k] = best_val;
        }

        float weight_sum = 0.0f;
        for (int k = 0; k < p->top_k; k++) weight_sum += top_weights[k];
        for (int k = 0; k < p->top_k; k++) top_weights[k] /= (weight_sum + 1e-8f);

        memset(s->moe_acc, 0, d * sizeof(float));
        for (int k = 0; k < p->top_k; k++) {
            int exp_id = top_indices[k];
            float exp_w = top_weights[k];
            ExpertWeights* ew = &lw->experts[exp_id];

            matmul(s->exp_h, s->xb, ew->w1, d, 4 * d);
            for (int i = 0; i < 4 * d; i++) s->exp_h[i] += ew->b1[i];
            gelu(s->exp_h, 4 * d);

            matmul(s->exp_out, s->exp_h, ew->w2, 4 * d, d);
            for (int i = 0; i < d; i++) {
                s->moe_acc[i] += exp_w * (s->exp_out[i] + ew->b2[i]);
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
// Tokenizer & Sampler
// ============================================================

int load_tokenizer(const char* path, Tokenizer* tok) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    if (fread(&tok->vocab_size, sizeof(int), 1, f) != 1) { fclose(f); return 0; }

    tok->vocab = (char**)malloc(tok->vocab_size * sizeof(char*));
    tok->vocab_lens = (int*)malloc(tok->vocab_size * sizeof(int));

    for (int i = 0; i < tok->vocab_size; i++) {
        int len = 0;
        if (fread(&len, sizeof(int), 1, f) != 1) len = 0;
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

// Поиск токенов для пользовательского промпта
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
            // fallback: одиночный символ
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
    // 1. Repetition penalty (штраф за повторы)
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

    // Top-K фильтрация
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
// Интерактивный генератор (Интерпретатор)
// ============================================================

void run_interactive(MoEModel* model, Tokenizer* tok) {
    RunState state;
    init_run_state(&state, &model->config);

    char input_buf[1024];
    int prompt_tokens[256];

    printf("============================================================\n");
    printf("  🤖 KobyakovAI - MoE Deep Intelligence Shell\n");
    printf("  Pure C Brain (Compiled with Zig cc) | 52.6M Params (8 Experts)\n");
    printf("  Type your prompt and press Enter.\n");
    printf("  Type 'exit' or 'quit' to exit.\n");
    printf("============================================================\n\n");

    while (1) {
        printf("\n\033[1mUser:\033[0m ");
        fflush(stdout);

        if (!fgets(input_buf, sizeof(input_buf), stdin)) break;

        size_t slen = strlen(input_buf);
        while (slen > 0 && (input_buf[slen - 1] == '\n' || input_buf[slen - 1] == '\r')) {
            input_buf[--slen] = '\0';
        }

        // Strip UTF-8 BOM if piped from PowerShell
        char* clean_input = input_buf;
        if ((unsigned char)clean_input[0] == 0xEF && 
            (unsigned char)clean_input[1] == 0xBB && 
            (unsigned char)clean_input[2] == 0xBF) {
            clean_input += 3;
        }
        while (*clean_input == ' ' || *clean_input == '\t') clean_input++;

        if (strlen(clean_input) == 0) continue;
        if (strcmp(clean_input, "exit") == 0 || strcmp(clean_input, "quit") == 0) {
            printf("Goodbye!\n");
            break;
        }

        int num_prompt_tokens = 0;
        // 1. Prefix: "User:"
        prompt_tokens[num_prompt_tokens++] = 12982; // 'User'
        prompt_tokens[num_prompt_tokens++] = 25;    // ':'

        // 2. User text with leading space for exact BPE tokenization
        char spaced_input[1024];
        snprintf(spaced_input, sizeof(spaced_input), " %s", clean_input);
        int n_user = encode_prompt(tok, spaced_input, prompt_tokens + num_prompt_tokens, 200);
        num_prompt_tokens += n_user;

        // 3. Suffix: "\n\nKobyakovAI:\n"
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
            forward(model, &state, prompt_tokens[p], pos++);
        }
        current_token = prompt_tokens[num_prompt_tokens - 1];

        int max_new_tokens = 120;
        int generated_count = 0;
        int recent_tokens[64];
        int num_recent = 0;

        for (; generated_count < max_new_tokens && pos < model->config.block_size - 1; pos++, generated_count++) {
            float* logits = forward(model, &state, current_token, pos);
            current_token = sample_token(logits, model->config.vocab_size, 0.0f, 0, recent_tokens, num_recent, 1.1f);

            // Остановка при маркере конца текста или переходе к User:
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

    run_interactive(&model, &tok);

    return 0;
}
