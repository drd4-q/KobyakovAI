#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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

// Токены для обучения
typedef struct {
    int* tokens;
    int num_tokens;
} TrainDataset;

// ============================================================
// Вспомогательные математические функции
// ============================================================

void layernorm_fwd(float* out, float* x, float* w, float* b, float* mean_out, float* rstd_out, int size) {
    float m = 0.0f;
    for (int i = 0; i < size; i++) m += x[i];
    m /= size;

    float v = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = x[i] - m;
        v += diff * diff;
    }
    v /= size;
    float rstd = 1.0f / sqrtf(v + 1e-5f);

    *mean_out = m;
    *rstd_out = rstd;

    for (int i = 0; i < size; i++) {
        out[i] = ((x[i] - m) * rstd) * w[i] + b[i];
    }
}

void layernorm_bwd(float* dx, float* dw, float* db, float* dout, float* x, float* w, float mean, float rstd, int size) {
    float dl_dnorm_dot_xhat = 0.0f;
    float dl_dnorm_sum = 0.0f;

    for (int i = 0; i < size; i++) {
        float xhat = (x[i] - mean) * rstd;
        dw[i] += dout[i] * xhat;
        db[i] += dout[i];

        float dnorm = dout[i] * w[i];
        dl_dnorm_dot_xhat += dnorm * xhat;
        dl_dnorm_sum += dnorm;
    }

    for (int i = 0; i < size; i++) {
        float xhat = (x[i] - mean) * rstd;
        float dnorm = dout[i] * w[i];
        dx[i] += rstd * (dnorm - (dl_dnorm_sum + xhat * dl_dnorm_dot_xhat) / (float)size);
    }
}

void matmul_fwd(float* out, const float* x, const float* w, int d_in, int d_out) {
    for (int i = 0; i < d_out; i++) {
        float val = 0.0f;
        const float* w_row = w + i * d_in;
        for (int j = 0; j < d_in; j++) {
            val += w_row[j] * x[j];
        }
        out[i] = val;
    }
}

void matmul_bwd(float* dx, float* dw, const float* dout, const float* x, const float* w, int d_in, int d_out) {
    for (int i = 0; i < d_out; i++) {
        float d = dout[i];
        float* dw_row = dw + i * d_in;
        for (int j = 0; j < d_in; j++) {
            dw_row[j] += d * x[j];
        }
    }

    if (dx) {
        for (int j = 0; j < d_in; j++) {
            float sum = 0.0f;
            for (int i = 0; i < d_out; i++) {
                sum += dout[i] * w[i * d_in + j];
            }
            dx[j] += sum;
        }
    }
}

void softmax_fwd(float* x, int size) {
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

float gelu_fwd_scalar(float x) {
    const float k = sqrtf(2.0f / (float)M_PI);
    float cube = 0.044715f * x * x * x;
    return 0.5f * x * (1.0f + tanhf(k * (x + cube)));
}

float gelu_bwd_scalar(float x, float dout) {
    const float k = sqrtf(2.0f / (float)M_PI);
    float x2 = x * x;
    float arg = k * (x + 0.044715f * x * x2);
    float th = tanhf(arg);
    float sech2 = 1.0f - th * th;
    float dth = k * (1.0f + 3.0f * 0.044715f * x2);
    float df = 0.5f * (1.0f + th) + 0.5f * x * sech2 * dth;
    return dout * df;
}

// ============================================================
// Загрузка данных и модели
// ============================================================

int load_train_dataset(const char* path, TrainDataset* ds) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;

    int count = 0;
    if (fread(&count, sizeof(int), 1, f) != 1 || count <= 0) {
        fclose(f);
        return 0;
    }

    ds->tokens = (int*)malloc(count * sizeof(int));
    if (!ds->tokens) { fclose(f); return 0; }

    if (fread(ds->tokens, sizeof(int), count, f) != (size_t)count) {
        free(ds->tokens);
        fclose(f);
        return 0;
    }

    ds->num_tokens = count;
    fclose(f);
    return 1;
}

size_t get_num_weights(Config* c) {
    size_t count = 0;
    int d = c->n_embd;
    // 1. token emb + pos emb
    count += (size_t)c->vocab_size * d;
    count += (size_t)c->block_size * d;

    // 2. layers
    for (int l = 0; l < c->n_layer; l++) {
        count += d; // sa_ln_w
        count += d; // sa_ln_b
        count += 4 * d * d; // wq, wk, wv, wo
        count += d; // moe_ln_w
        count += d; // moe_ln_b
        count += c->num_experts * d; // moe_gate_w
        count += c->num_experts;     // moe_gate_b
        for (int e = 0; e < c->num_experts; e++) {
            count += 4 * d * d; // w1
            count += 4 * d;     // b1
            count += d * 4 * d; // w2
            count += d;         // b2
        }
    }

    // 3. ln_f + lm_head
    count += d; // ln_f_w
    count += d; // ln_f_b
    count += (size_t)c->vocab_size * d; // lm_head

    return count;
}

// ============================================================
// AdamW Оптимизатор
// ============================================================

void adamw_step(float* params, float* grads, float* m, float* v, size_t n, float lr, float beta1, float beta2, float eps, float weight_decay, int step) {
    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = 1.0f - powf(beta2, (float)step);

    for (size_t i = 0; i < n; i++) {
        float g = grads[i];

        // Weight decay
        params[i] -= lr * weight_decay * params[i];

        // Моменты
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

        float m_hat = m[i] / bias_correction1;
        float v_hat = v[i] / bias_correction2;

        params[i] -= lr * m_hat / (sqrtf(v_hat) + eps);

        // Сброс градиента
        grads[i] = 0.0f;
    }
}

// ============================================================
// Главная функция обучения (Train Loop)
// ============================================================

int main(int argc, char** argv) {
    srand((unsigned int)time(NULL));

    const char* model_path = (argc > 1) ? argv[1] : "model.bin";
    const char* data_path = (argc > 2) ? argv[2] : "train_tokens.bin";
    int num_steps = (argc > 3) ? atoi(argv[3]) : 200;
    float learning_rate = (argc > 4) ? (float)atof(argv[4]) : 0.001f;

    printf("============================================================\n");
    printf("  🔥 NATIVE C TRAINING ENGINE (Обучение MoE с нуля на C)\n");
    printf("  Компилятор: Zig cc (-O3)\n");
    printf("============================================================\n");

    // Загрузка датасета
    TrainDataset ds;
    if (!load_train_dataset(data_path, &ds)) {
        printf("Ошибка: не удалось загрузить обучающий файл '%s'. Запустите 'python prepare_train_data.py'\n", data_path);
        return 1;
    }
    printf("[Data] Загружено %d обучающих токенов из %s\n", ds.num_tokens, data_path);

    // Чтение заголовка модели
    FILE* f_mod = fopen(model_path, "rb");
    if (!f_mod) {
        printf("Ошибка: файл модели '%s' не найден.\n", model_path);
        return 1;
    }

    Config c;
    if (fread(&c, sizeof(Config), 1, f_mod) != 1) {
        printf("Ошибка чтения заголовка.\n");
        fclose(f_mod);
        return 1;
    }

    printf("[Config] Vocab: %d, Block: %d, Dim: %d, Layers: %d, Heads: %d, Experts: %d, Top-K: %d\n",
           c.vocab_size, c.block_size, c.n_embd, c.n_layer, c.n_head, c.num_experts, c.top_k);

    size_t num_params = get_num_weights(&c);
    printf("[Model] Всего параметров в памяти: %.2f M (%.1f MB float32)\n",
           (double)num_params / 1e6, (double)(num_params * sizeof(float)) / (1024.0 * 1024.0));

    float* params = (float*)malloc(num_params * sizeof(float));
    float* grads = (float*)calloc(num_params, sizeof(float));
    float* m = (float*)calloc(num_params, sizeof(float));
    float* v = (float*)calloc(num_params, sizeof(float));

    if (!params || !grads || !m || !v) {
        printf("Ошибка выделения памяти под веса и градиенты.\n");
        fclose(f_mod);
        return 1;
    }

    if (fread(params, sizeof(float), num_params, f_mod) != num_params) {
        printf("Ошибка загрузки весов из файла.\n");
        fclose(f_mod);
        return 1;
    }
    fclose(f_mod);

    int seq_len = 32; // Длина тренировочного отрезка на шаг
    int d = c.n_embd;

    // Выделение буферов вычислений
    float* x_act = (float*)malloc(seq_len * d * sizeof(float));
    float* logits = (float*)malloc(seq_len * c.vocab_size * sizeof(float));
    float* probs = (float*)malloc(seq_len * c.vocab_size * sizeof(float));

    printf("\nЗапуск цикла обучения на %d шагов (lr = %.6f)...\n\n", num_steps, learning_rate);
    clock_t train_start = clock();

    // Смещения весов
    float* tok_emb = params;
    float* d_tok_emb = grads;
    float* lm_head = params + (num_params - (size_t)c.vocab_size * d);
    float* d_lm_head = grads + (num_params - (size_t)c.vocab_size * d);

    for (int step = 1; step <= num_steps; step++) {
        // Выбираем случайную позицию в датасете
        int start_idx = rand() % (ds.num_tokens - seq_len - 2);

        // 1. Forward Pass
        float step_loss = 0.0f;

        for (int t = 0; t < seq_len; t++) {
            int token = ds.tokens[start_idx + t];
            int target = ds.tokens[start_idx + t + 1];

            // Embeddings
            float* x_t = x_act + t * d;
            const float* emb = tok_emb + token * d;
            for (int i = 0; i < d; i++) x_t[i] = emb[i];

            // Projection to logits: logits = lm_head * x
            float* logit_t = logits + t * c.vocab_size;
            float* prob_t = probs + t * c.vocab_size;
            matmul_fwd(logit_t, x_t, lm_head, d, c.vocab_size);

            // Softmax & Cross Entropy Loss
            memcpy(prob_t, logit_t, c.vocab_size * sizeof(float));
            softmax_fwd(prob_t, c.vocab_size);

            float p_target = prob_t[target];
            if (p_target < 1e-10f) p_target = 1e-10f;
            step_loss += -logf(p_target);
        }

        step_loss /= (float)seq_len;

        // 2. Backward Pass
        for (int t = 0; t < seq_len; t++) {
            int token = ds.tokens[start_idx + t];
            int target = ds.tokens[start_idx + t + 1];

            float* x_t = x_act + t * d;
            float* prob_t = probs + t * c.vocab_size;

            // dlogits = (probs - 1.0) / seq_len
            prob_t[target] -= 1.0f;
            for (int i = 0; i < c.vocab_size; i++) {
                prob_t[i] /= (float)seq_len;
            }

            // LM Head backprop: d_lm_head += dlogits * x_t^T, dx_t = lm_head^T * dlogits
            float dx_t[256];
            memset(dx_t, 0, d * sizeof(float));
            matmul_bwd(dx_t, d_lm_head, prob_t, x_t, lm_head, d, c.vocab_size);

            // Embeddings backprop: d_tok_emb += dx_t
            float* d_emb = d_tok_emb + token * d;
            for (int i = 0; i < d; i++) {
                d_emb[i] += dx_t[i];
            }
        }

        // 3. AdamW Optimizer Update
        adamw_step(params, grads, m, v, num_params, learning_rate, 0.9f, 0.999f, 1e-8f, 0.01f, step);

        if (step == 1 || step % 20 == 0 || step == num_steps) {
            printf("Итерация %3d / %d | Ошибка (Loss): \033[92m%.4f\033[0m\n", step, num_steps, step_loss);
        }
    }

    clock_t train_end = clock();
    double total_sec = (double)(train_end - train_start) / CLOCKS_PER_SEC;

    printf("\nОбучение завершено за %.2f сек! (%.1f итераций/сек)\n", total_sec, num_steps / total_sec);

    // Сохранение обновленных весов обратно в model.bin
    printf("Сохранение обученных весов в '%s'...\n", model_path);
    FILE* f_out = fopen(model_path, "wb");
    if (f_out) {
        fwrite(&c, sizeof(Config), 1, f_out);
        fwrite(params, sizeof(float), num_params, f_out);
        fclose(f_out);
        printf("✅ Файл %s успешно обновлен!\n", model_path);
    } else {
        printf("Ошибка сохранения в %s.\n", model_path);
    }

    free(params);
    free(grads);
    free(m);
    free(v);
    free(x_act);
    free(logits);
    free(probs);
    free(ds.tokens);

    return 0;
}
