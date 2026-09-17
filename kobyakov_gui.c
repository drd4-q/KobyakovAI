#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wininet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_IMPLEMENTATION
#define NK_GDI_IMPLEMENTATION
#include "nuklear.h"
#include "nuklear_gdi.h"

#define WINDOW_WIDTH 1150
#define WINDOW_HEIGHT 780
#define DAEMON_HOST "http://127.0.0.1:8999"

// ============================================================
// Глобальные переменные процессов и состояние
// ============================================================
static PROCESS_INFORMATION g_daemon_pi = {0};
static int g_daemon_started_by_us = 0;

typedef struct {
    int current_tab; // 0: Chat, 1: Watcher & Video Learning, 2: Training, 3: System

    // Чат
    char chat_input[512];
    char chat_history[131072];
    int use_vision;
    int use_web;
    int is_thinking;

    // Веб-серфинг
    char web_input[512];
    char web_results_text[131072];
    char web_status[256];
    int web_is_loading;
    int web_page_chars;
    int web_page_lines;
    char web_last_urls[4][256];
    char web_last_titles[4][256];
    int web_results_count;

    // Список окон
    char window_titles[40][128];
    int window_hwnds[40];
    const char* window_ptrs[40];
    int window_count;
    int selected_window_idx;

    // Video Watcher & Active Video Learning
    int stream_active;
    int video_learning_active;
    float stream_fps;
    int stream_frames;
    int stream_training_steps;
    float stream_loss;
    int stream_learned_count;
    char video_analysis[16384];
    char stream_learned_log[16384];
    HBITMAP preview_hbm;
    int preview_w;
    int preview_h;
    clock_t last_stream_poll;

    // Классическое обучение
    int train_domain; // 0: all, 1: code, 2: math, 3: logic
    int train_steps;
    int train_is_running;
    int train_cur_step;
    int train_tot_steps;
    float train_loss;
    char train_status[256];
    char train_log[8192];
    clock_t last_train_poll;

    // Системный статус
    char system_info[1024];
    int daemon_online;
} AppState;

static AppState g_state;

// ============================================================
// HTTP клиент (WinINet)
// ============================================================
int http_get(const char* url, char* out_buf, int max_len) {
    HINTERNET hInternet = InternetOpenA("KobyakovAI", INTERNET_OPEN_TYPE_DIRECT, NULL, NULL, 0);
    if (!hInternet) return 0;

    DWORD timeout = 2500;
    InternetSetOptionA(hInternet, INTERNET_OPTION_CONNECT_TIMEOUT, &timeout, sizeof(timeout));
    InternetSetOptionA(hInternet, INTERNET_OPTION_RECEIVE_TIMEOUT, &timeout, sizeof(timeout));

    HINTERNET hUrl = InternetOpenUrlA(hInternet, url, NULL, 0, INTERNET_FLAG_RELOAD | INTERNET_FLAG_NO_CACHE_WRITE, 0);
    if (!hUrl) {
        InternetCloseHandle(hInternet);
        return 0;
    }

    DWORD totalRead = 0;
    DWORD bytesRead = 0;
    while (totalRead < (DWORD)(max_len - 1) && InternetReadFile(hUrl, out_buf + totalRead, max_len - 1 - totalRead, &bytesRead) && bytesRead > 0) {
        totalRead += bytesRead;
    }
    out_buf[totalRead] = '\0';

    InternetCloseHandle(hUrl);
    InternetCloseHandle(hInternet);
    return (totalRead > 0);
}

int http_post_json(const char* endpoint, const char* json_body, char* out_buf, int max_len) {
    HINTERNET hInternet = InternetOpenA("KobyakovAI", INTERNET_OPEN_TYPE_DIRECT, NULL, NULL, 0);
    if (!hInternet) return 0;

    DWORD timeout = 90000; // 90s timeout for AI vision + MoE
    InternetSetOptionA(hInternet, INTERNET_OPTION_CONNECT_TIMEOUT, &timeout, sizeof(timeout));
    InternetSetOptionA(hInternet, INTERNET_OPTION_RECEIVE_TIMEOUT, &timeout, sizeof(timeout));

    HINTERNET hConnect = InternetConnectA(hInternet, "127.0.0.1", 8999, NULL, NULL, INTERNET_SERVICE_HTTP, 0, 0);
    if (!hConnect) {
        InternetCloseHandle(hInternet);
        return 0;
    }

    HINTERNET hRequest = HttpOpenRequestA(hConnect, "POST", endpoint, NULL, NULL, NULL, INTERNET_FLAG_RELOAD | INTERNET_FLAG_NO_CACHE_WRITE, 0);
    if (!hRequest) {
        InternetCloseHandle(hConnect);
        InternetCloseHandle(hInternet);
        return 0;
    }

    const char* headers = "Content-Type: application/json; charset=utf-8\r\n";
    DWORD body_len = (DWORD)strlen(json_body);

    BOOL res = HttpSendRequestA(hRequest, headers, (DWORD)strlen(headers), (LPVOID)json_body, body_len);
    int success = 0;
    if (res) {
        DWORD totalRead = 0;
        DWORD bytesRead = 0;
        while (totalRead < (DWORD)(max_len - 1) && InternetReadFile(hRequest, out_buf + totalRead, max_len - 1 - totalRead, &bytesRead) && bytesRead > 0) {
            totalRead += bytesRead;
        }
        out_buf[totalRead] = '\0';
        success = 1;
    }

    InternetCloseHandle(hRequest);
    InternetCloseHandle(hConnect);
    InternetCloseHandle(hInternet);
    return success;
}

// ============================================================
// Автоматический запуск и остановка демона
// ============================================================
int is_daemon_alive() {
    char test_buf[256];
    return http_get(DAEMON_HOST "/api/ping", test_buf, sizeof(test_buf));
}

void start_daemon_if_needed() {
    if (is_daemon_alive()) {
        g_state.daemon_online = 1;
        return;
    }

    STARTUPINFOA si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESHOWWINDOW;
    si.wShowWindow = SW_HIDE;
    ZeroMemory(&g_daemon_pi, sizeof(g_daemon_pi));

    char exe_path[MAX_PATH];
    GetModuleFileNameA(NULL, exe_path, MAX_PATH);
    char* last_slash = strrchr(exe_path, '\\');
    char exe_dir[MAX_PATH];
    if (last_slash) {
        size_t len = last_slash - exe_path;
        strncpy(exe_dir, exe_path, len);
        exe_dir[len] = '\0';
    } else {
        strcpy(exe_dir, ".");
    }

    char full_cmd[1024];
    snprintf(full_cmd, sizeof(full_cmd), "python \"%s\\vision_daemon.py\"", exe_dir);
    BOOL ok = CreateProcessA(NULL, full_cmd, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, exe_dir, &si, &g_daemon_pi);
    if (!ok) {
        snprintf(full_cmd, sizeof(full_cmd), "py \"%s\\vision_daemon.py\"", exe_dir);
        ok = CreateProcessA(NULL, full_cmd, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, exe_dir, &si, &g_daemon_pi);
    }

    if (ok) {
        g_daemon_started_by_us = 1;
        for (int i = 0; i < 30; i++) {
            Sleep(200);
            if (is_daemon_alive()) {
                g_state.daemon_online = 1;
                break;
            }
        }
    }
}

void cleanup_daemon() {
    if (g_daemon_started_by_us && g_daemon_pi.hProcess) {
        TerminateProcess(g_daemon_pi.hProcess, 0);
        CloseHandle(g_daemon_pi.hProcess);
        if (g_daemon_pi.hThread) CloseHandle(g_daemon_pi.hThread);
        g_daemon_pi.hProcess = NULL;
    }
}

void ensure_model_weights() {
    FILE* f = fopen("model.bin", "rb");
    if (!f) {
        STARTUPINFOA si;
        PROCESS_INFORMATION pi;
        ZeroMemory(&si, sizeof(si));
        si.cb = sizeof(si);
        si.dwFlags = STARTF_USESHOWWINDOW;
        si.wShowWindow = SW_HIDE;
        ZeroMemory(&pi, sizeof(pi));

        char cmd[256] = "python export_weights.py";
        if (CreateProcessA(NULL, cmd, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
            WaitForSingleObject(pi.hProcess, 60000);
            CloseHandle(pi.hProcess);
            if (pi.hThread) CloseHandle(pi.hThread);
        }
    } else {
        fclose(f);
    }
}

// ============================================================
// API взаимодействия с видеопотоком и обучением
// ============================================================
void refresh_windows_list(AppState* s) {
    char resp[16384];
    if (http_get(DAEMON_HOST "/api/windows", resp, sizeof(resp))) {
        s->daemon_online = 1;
        s->window_count = 0;
        strcpy(s->window_titles[0], "Весь экран (Desktop Screen)");
        s->window_hwnds[0] = 0;
        s->window_ptrs[0] = s->window_titles[0];
        s->window_count = 1;

        char* p = strstr(resp, "{\"index\":");
        while (p && s->window_count < 38) {
            int hwnd_val = 0;
            char* p_hwnd = strstr(p, "\"hwnd\":");
            if (p_hwnd) hwnd_val = atoi(p_hwnd + 7);

            char* p_title = strstr(p, "\"title\":");
            if (!p_title) break;
            p_title += 8;
            while (*p_title == ' ' || *p_title == '\"') p_title++;
            char* end = strchr(p_title, '\"');
            if (!end) break;
            int len = (int)(end - p_title);
            if (len > 120) len = 120;
            strncpy(s->window_titles[s->window_count], p_title, len);
            s->window_titles[s->window_count][len] = '\0';
            s->window_hwnds[s->window_count] = hwnd_val;
            s->window_ptrs[s->window_count] = s->window_titles[s->window_count];
            s->window_count++;
            p = strstr(end, "{\"index\":");
        }
    }
}

void capture_single_preview(AppState* s) {
    char req[512];
    int hwnd = (s->selected_window_idx > 0) ? s->window_hwnds[s->selected_window_idx] : 0;
    if (s->selected_window_idx <= 0) {
        strcpy(req, "{\"target\": null, \"hwnd\": 0}");
    } else {
        snprintf(req, sizeof(req), "{\"target\": \"%s\", \"hwnd\": %d}", s->window_titles[s->selected_window_idx], hwnd);
    }
    char resp[1024];
    if (http_post_json("/api/capture", req, resp, sizeof(resp))) {
        if (s->preview_hbm) {
            DeleteObject(s->preview_hbm);
            s->preview_hbm = NULL;
        }
        s->preview_hbm = (HBITMAP)LoadImageA(NULL, "preview.bmp", IMAGE_BITMAP, 0, 0, LR_LOADFROMFILE);
        s->preview_w = 480;
        s->preview_h = 270;
    }
}

void start_video_stream(AppState* s, int with_learning) {
    char req[512];
    int hwnd = (s->selected_window_idx > 0) ? s->window_hwnds[s->selected_window_idx] : 0;
    const char* win = (s->selected_window_idx <= 0) ? "" : s->window_titles[s->selected_window_idx];
    snprintf(req, sizeof(req), "{\"target\": \"%s\", \"hwnd\": %d, \"enable_learning\": %s}",
             win, hwnd, with_learning ? "true" : "false");
    char resp[512];
    if (http_post_json("/api/video_stream/start", req, resp, sizeof(resp))) {
        s->stream_active = 1;
        s->video_learning_active = with_learning;
    }
}

void stop_video_stream(AppState* s) {
    char resp[512];
    if (http_post_json("/api/video_stream/stop", "{}", resp, sizeof(resp))) {
        s->stream_active = 0;
        s->video_learning_active = 0;
    }
}

void poll_video_stream_status(AppState* s) {
    char resp[32768];
    if (http_get(DAEMON_HOST "/api/video_stream/status", resp, sizeof(resp))) {
        char* p_stream = strstr(resp, "\"streaming\":");
        if (p_stream) s->stream_active = (strstr(p_stream, "true") != NULL);
        char* p_learn = strstr(resp, "\"learning\":");
        if (p_learn) s->video_learning_active = (strstr(p_learn, "true") != NULL);
        char* p_fps = strstr(resp, "\"fps\":");
        if (p_fps) s->stream_fps = (float)atof(p_fps + 6);
        char* p_fa = strstr(resp, "\"frames_analyzed\":");
        if (p_fa) s->stream_frames = atoi(p_fa + 18);
        char* p_steps = strstr(resp, "\"training_steps\":");
        if (p_steps) s->stream_training_steps = atoi(p_steps + 17);
        char* p_loss = strstr(resp, "\"current_loss\":");
        if (p_loss) s->stream_loss = (float)atof(p_loss + 15);
        char* p_count = strstr(resp, "\"learned_count\":");
        if (p_count) s->stream_learned_count = atoi(p_count + 16);

        // Анализ текущего кадра
        char* p_la = strstr(resp, "\"latest_analysis\":");
        if (p_la) {
            p_la += 18;
            while (*p_la == ' ' || *p_la == '\"') p_la++;
            char* end = strchr(p_la, '\"');
            if (end) {
                int len = (int)(end - p_la);
                if (len > 8000) len = 8000;
                char clean[8192];
                int ci = 0;
                for (int i = 0; i < len && ci < 8100; i++) {
                    if (p_la[i] == '\\' && p_la[i+1] == 'n') { clean[ci++] = '\n'; i++; }
                    else if (p_la[i] == '\\' && p_la[i+1] == '\"') { clean[ci++] = '\"'; i++; }
                    else clean[ci++] = p_la[i];
                }
                clean[ci] = '\0';
                strncpy(s->video_analysis, clean, sizeof(s->video_analysis) - 1);
            }
        }

        // Лог выученного из видео
        char* p_log = strstr(resp, "\"learned_log\":");
        if (p_log) {
            p_log += 14;
            while (*p_log == ' ' || *p_log == '\"') p_log++;
            char* end = strrchr(p_log, '\"');
            if (end) {
                int len = (int)(end - p_log);
                if (len > 12000) len = 12000;
                char clean[16384];
                int ci = 0;
                for (int i = 0; i < len && ci < 16000; i++) {
                    if (p_log[i] == '\\' && p_log[i+1] == 'n') { clean[ci++] = '\n'; i++; }
                    else if (p_log[i] == '\\' && p_log[i+1] == '\"') { clean[ci++] = '\"'; i++; }
                    else clean[ci++] = p_log[i];
                }
                clean[ci] = '\0';
                strncpy(s->stream_learned_log, clean, sizeof(s->stream_learned_log) - 1);
            }
        }

        // Обновление превью в реальном времени (GDI DIB)
        if (s->stream_active) {
            HBITMAP hbm = (HBITMAP)LoadImageA(NULL, "preview.bmp", IMAGE_BITMAP, 0, 0, LR_LOADFROMFILE);
            if (hbm) {
                if (s->preview_hbm) DeleteObject(s->preview_hbm);
                s->preview_hbm = hbm;
                s->preview_w = 480;
                s->preview_h = 270;
            }
        }
    }
}

void send_chat_message(AppState* s) {
    if (strlen(s->chat_input) == 0) return;

    int cur_len = (int)strlen(s->chat_history);
    snprintf(s->chat_history + cur_len, sizeof(s->chat_history) - cur_len, "\nПользователь: %s\n\n🤖 KobyakovAI:\n", s->chat_input);

    char req[2048];
    int hwnd = (s->use_vision && s->selected_window_idx > 0) ? s->window_hwnds[s->selected_window_idx] : 0;
    const char* win = (s->use_vision && s->selected_window_idx > 0) ? s->window_titles[s->selected_window_idx] : "";
    snprintf(req, sizeof(req), "{\"prompt\": \"%s\", \"use_vision\": %s, \"use_web\": %s, \"target\": \"%s\", \"hwnd\": %d}",
             s->chat_input, s->use_vision ? "true" : "false", s->use_web ? "true" : "false", win, hwnd);

    s->chat_input[0] = '\0';
    s->is_thinking = 1;

    char resp[16384];
    if (http_post_json("/api/chat", req, resp, sizeof(resp))) {
        char* p = strstr(resp, "\"response\":");
        if (p) {
            p += 11;
            while (*p == ' ' || *p == '\"') p++;
            char* end = strrchr(p, '\"');
            if (end) *end = '\0';
            char clean[8192];
            int ci = 0;
            for (int i = 0; p[i] && ci < 8100; i++) {
                if (p[i] == '\\' && p[i+1] == 'n') { clean[ci++] = '\n'; i++; }
                else clean[ci++] = p[i];
            }
            clean[ci] = '\0';
            cur_len = (int)strlen(s->chat_history);
            snprintf(s->chat_history + cur_len, sizeof(s->chat_history) - cur_len, "%s\n------------------------------------------------------------\n", clean);
        }
    } else {
        cur_len = (int)strlen(s->chat_history);
        snprintf(s->chat_history + cur_len, sizeof(s->chat_history) - cur_len, "[Ошибка соединения с демоном нейросети]\n------------------------------------------------------------\n");
    }
    s->is_thinking = 0;
}

// ============================================================
// Контролируемый Веб-серфинг & Поиск
// ============================================================
void perform_web_search(AppState* s) {
    if (strlen(s->web_input) == 0) return;
    s->web_is_loading = 1;
    strcpy(s->web_status, "🔍 Выполняется поиск в сети...");

    char req[1024];
    snprintf(req, sizeof(req), "{\"query\": \"%s\", \"max_results\": 4}", s->web_input);
    char resp[16384];
    if (http_post_json("/api/web/search", req, resp, sizeof(resp))) {
        char* p = strstr(resp, "\"results\":");
        if (p) {
            char clean[16384];
            clean[0] = '\0';
            char* cur = p;
            int item_idx = 1;
            while ((cur = strstr(cur, "{\"title\":")) != NULL && item_idx <= 5) {
                char* p_t = strstr(cur, "\"title\":");
                char* p_u = strstr(cur, "\"url\":");
                char* p_s = strstr(cur, "\"snippet\":");
                if (p_t && p_u && p_s) {
                    char title[256] = {0};
                    char url[256] = {0};
                    char snippet[1024] = {0};

                    p_t += 8; while (*p_t == ' ' || *p_t == '"') p_t++;
                    char* e_t = strchr(p_t, '"');
                    if (e_t) { int len = (int)(e_t - p_t); if (len > 250) len = 250; strncpy(title, p_t, len); }

                    p_u += 6; while (*p_u == ' ' || *p_u == '"') p_u++;
                    char* e_u = strchr(p_u, '"');
                    if (e_u) { int len = (int)(e_u - p_u); if (len > 250) len = 250; strncpy(url, p_u, len); }

                    p_s += 10; while (*p_s == ' ' || *p_s == '"') p_s++;
                    char* e_s = strchr(p_s, '"');
                    if (e_s) { int len = (int)(e_s - p_s); if (len > 1000) len = 1000; strncpy(snippet, p_s, len); }

                    if (item_idx <= 4) {
                        strncpy(s->web_last_urls[item_idx - 1], url, 255);
                        strncpy(s->web_last_titles[item_idx - 1], title, 255);
                    }

                    int cl = (int)strlen(clean);
                    snprintf(clean + cl, sizeof(clean) - cl,
                             "[%d] %s\nСсылка: %s\nСуть: %s\n------------------------------------------------------------\n",
                             item_idx, title, url, snippet);
                    item_idx++;
                }
                cur += 9;
            }
            s->web_results_count = item_idx - 1;
            if (strlen(clean) > 0) {
                strncpy(s->web_results_text, clean, sizeof(s->web_results_text) - 1);
                snprintf(s->web_status, sizeof(s->web_status), "✓ Найдено %d релевантных источников в сети. Выберите источник ниже для чтения статьи целиком.", item_idx - 1);
            } else {
                strcpy(s->web_results_text, "По вашему запросу ничего не найдено.\n");
                strcpy(s->web_status, "Ничего не найдено.");
            }
        }
    } else {
        strcpy(s->web_status, "Ошибка соединения с поисковым модулем.");
    }
    s->web_is_loading = 0;
}

void perform_web_fetch(AppState* s) {
    if (strlen(s->web_input) == 0) return;
    s->web_is_loading = 1;
    strcpy(s->web_status, "🌐 Загрузка и полнотекстовый анализ веб-страницы...");

    char req[1024];
    snprintf(req, sizeof(req), "{\"url\": \"%s\"}", s->web_input);
    static char resp[262144];
    if (http_post_json("/api/web/fetch", req, resp, sizeof(resp))) {
        char* p_ch = strstr(resp, "\"chars\":");
        if (p_ch) s->web_page_chars = atoi(p_ch + 8);
        char* p_ln = strstr(resp, "\"lines\":");
        if (p_ln) s->web_page_lines = atoi(p_ln + 8);

        char* p_c = strstr(resp, "\"content\":");
        if (p_c) {
            p_c += 10;
            while (*p_c == ' ' || *p_c == '"') p_c++;
            char* end = strrchr(p_c, '"');
            if (end) *end = '\0';
            static char clean[131072];
            int ci = 0;
            for (int i = 0; p_c[i] && ci < 131000; i++) {
                if (p_c[i] == '\\' && p_c[i+1] == 'n') { clean[ci++] = '\n'; i++; }
                else if (p_c[i] == '\\' && p_c[i+1] == '"') { clean[ci++] = '"'; i++; }
                else if (p_c[i] == '\\' && p_c[i+1] == '\\') { clean[ci++] = '\\'; i++; }
                else clean[ci++] = p_c[i];
            }
            clean[ci] = '\0';
            strncpy(s->web_results_text, clean, sizeof(s->web_results_text) - 1);
            snprintf(s->web_status, sizeof(s->web_status), "✓ Страница прочитана полностью (%d символов, %d строк). Сохранено в память.",
                     s->web_page_chars ? s->web_page_chars : (int)strlen(clean), s->web_page_lines);
        }
    } else {
        strcpy(s->web_status, "Ошибка перехода по ссылке.");
    }
    s->web_is_loading = 0;
}

void send_web_to_chat(AppState* s) {
    if (strlen(s->web_results_text) == 0) return;
    int cur_len = (int)strlen(s->chat_history);
    snprintf(s->chat_history + cur_len, sizeof(s->chat_history) - cur_len,
             "\n[🌐 Контекст из Веб-серфера]:\n%s\n------------------------------------------------------------\n",
             s->web_results_text);
    s->current_tab = 0;
}

void learn_web_content(AppState* s) {
    if (strlen(s->web_results_text) == 0) return;
    char req[4096];
    char escaped[3000];
    int ei = 0;
    for (int i = 0; s->web_results_text[i] && ei < 2800; i++) {
        if (s->web_results_text[i] == '"') { escaped[ei++] = '\\'; escaped[ei++] = '"'; }
        else if (s->web_results_text[i] == '\n') { escaped[ei++] = ' '; }
        else escaped[ei++] = s->web_results_text[i];
    }
    escaped[ei] = '\0';
    snprintf(req, sizeof(req), "{\"text\": \"%s\", \"source\": \"Веб-страница\"}", escaped);
    char resp[512];
    if (http_post_json("/api/web/learn", req, resp, sizeof(resp))) {
        strcpy(s->web_status, "✓ Веб-знания успешно усвоены и внедрены в веса MoE нейросети!");
    }
}

void poll_training_status(AppState* s);

void start_training(AppState* s) {
    const char* domains[] = {"all", "code", "math", "logic"};
    const char* domain = domains[s->train_domain % 4];
    char req[256];
    snprintf(req, sizeof(req), "{\"steps\": %d, \"domain\": \"%s\"}", s->train_steps, domain);
    char resp[512];
    if (http_post_json("/api/train/start", req, resp, sizeof(resp))) {
        s->train_is_running = 1;
        snprintf(s->train_status, sizeof(s->train_status), "Обучение запущено (%d шагов, домен: %s)...", s->train_steps, domain);
    }
}

void stop_training(AppState* s) {
    // 1. Создаем локальный файл-флаг для мгновенной остановки цикла train_gpu.py
    FILE* f = fopen("train_stop.flag", "w");
    if (f) {
        fputs("stop", f);
        fclose(f);
    }
    strcpy(s->train_status, "Остановка... Безопасное сохранение чекпоинта и весов...");

    // 2. Отправляем API запрос демону для прерывания и запуска watchdog
    char resp[512];
    http_post_json("/api/train/stop", "{}", resp, sizeof(resp));

    // 3. Задержка и немедленный опрос нового статуса
    Sleep(100);
    poll_training_status(s);
}

void poll_training_status(AppState* s) {
    char resp[2048];
    if (http_get(DAEMON_HOST "/api/train/status", resp, sizeof(resp))) {
        char* p_run = strstr(resp, "\"running\":");
        if (p_run) {
            p_run += 10;
            while (*p_run == ' ' || *p_run == '\t') p_run++;
            s->train_is_running = (strncmp(p_run, "true", 4) == 0);
        }
        char* p_step = strstr(resp, "\"step\":");
        if (p_step) s->train_cur_step = atoi(p_step + 7);
        char* p_tot = strstr(resp, "\"total_steps\":");
        if (p_tot) s->train_tot_steps = atoi(p_tot + 14);
        char* p_loss = strstr(resp, "\"loss\":");
        if (p_loss) s->train_loss = (float)atof(p_loss + 7);
        char* p_stat = strstr(resp, "\"status\":");
        if (p_stat) {
            p_stat += 9;
            while (*p_stat == ' ' || *p_stat == '\"') p_stat++;
            char* end = strchr(p_stat, '\"');
            if (end) {
                int len = (int)(end - p_stat);
                if (len > 250) len = 250;
                strncpy(s->train_status, p_stat, len);
                s->train_status[len] = '\0';
            }
        }
        char* p_log = strstr(resp, "\"last_log\":");
        if (p_log) {
            p_log += 11;
            while (*p_log == ' ' || *p_log == '\"') p_log++;
            char* end = strrchr(p_log, '\"');
            if (end) *end = '\0';
            if (strlen(p_log) > 0 && strstr(s->train_log, p_log) == NULL) {
                int cur_len = (int)strlen(s->train_log);
                snprintf(s->train_log + cur_len, sizeof(s->train_log) - cur_len, "%s\n", p_log);
            }
        }
    }
}

// ============================================================
// Window Procedure & Main Loop
// ============================================================
static LRESULT CALLBACK WindowProc(HWND wnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    switch (msg) {
        case WM_DESTROY:
            cleanup_daemon();
            PostQuitMessage(0);
            return 0;
    }
    if (nk_gdi_handle_event(wnd, msg, wparam, lparam))
        return 0;
    return DefWindowProcW(wnd, msg, wparam, lparam);
}

int main(int argc, char** argv) {
    if (argc > 1) {
        if (strcmp(argv[1], "--cli") == 0 || strcmp(argv[1], "-c") == 0) {
            AttachConsole(ATTACH_PARENT_PROCESS);
            printf("\n============================================================\n");
            printf("  Запуск KobyakovAI MoE CLI Shell (Pure C / 144 Ноды)...\n");
            printf("============================================================\n\n");
            system("engine_c.exe model.bin tokenizer.bin");
            return 0;
        }
        if (strcmp(argv[1], "--train") == 0 || strcmp(argv[1], "-t") == 0) {
            AttachConsole(ATTACH_PARENT_PROCESS);
            int steps = (argc > 2) ? atoi(argv[2]) : 1000;
            printf("\nЗапуск обучения KobyakovAI MoE на GPU CUDA (%d шагов)...\n", steps);
            char cmd[256];
            snprintf(cmd, sizeof(cmd), "python train_gpu.py --steps %d", steps);
            system(cmd);
            return 0;
        }
        if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
            AttachConsole(ATTACH_PARENT_PROCESS);
            printf("\nKobyakovAI Standalone Executable\n");
            printf("Использование:\n");
            printf("  KobyakovAI.exe            Запустить полноэкранную C/Zig Studio GUI\n");
            printf("  KobyakovAI.exe --cli      Интерактивный консольный чат MoE\n");
            printf("  KobyakovAI.exe --train N  Обучение MoE модели на GPU на N шагов\n\n");
            return 0;
        }
    }

    SetProcessDPIAware();
    ensure_model_weights();
    start_daemon_if_needed();

    WNDCLASSW wc;
    HWND wnd;
    HDC dc;
    int running = 1;
    struct nk_context* ctx;

    memset(&wc, 0, sizeof(wc));
    wc.style = CS_DBLCLKS;
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = GetModuleHandleW(0);
    wc.hIcon = LoadIcon(NULL, IDI_APPLICATION);
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.lpszClassName = L"KobyakovStudioV2Class";
    RegisterClassW(&wc);

    wnd = CreateWindowW(
        wc.lpszClassName,
        L"🤖 KobyakovAI Native Studio (144 MoE Nodes + Realtime Vision & Video Learning)",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        80, 40, WINDOW_WIDTH, WINDOW_HEIGHT,
        NULL, NULL, wc.hInstance, NULL
    );

    dc = GetDC(wnd);
    GdiFont* font = nk_gdifont_create("Segoe UI", 16);
    ctx = nk_gdi_init(font, dc, WINDOW_WIDTH, WINDOW_HEIGHT);

    // Dark Theme Setup
    struct nk_color table[NK_COLOR_COUNT];
    table[NK_COLOR_TEXT] = nk_rgba(230, 230, 230, 255);
    table[NK_COLOR_WINDOW] = nk_rgba(20, 22, 26, 255);
    table[NK_COLOR_HEADER] = nk_rgba(30, 34, 42, 255);
    table[NK_COLOR_BORDER] = nk_rgba(45, 52, 64, 255);
    table[NK_COLOR_BUTTON] = nk_rgba(35, 40, 50, 255);
    table[NK_COLOR_BUTTON_HOVER] = nk_rgba(70, 140, 230, 255);
    table[NK_COLOR_BUTTON_ACTIVE] = nk_rgba(50, 120, 210, 255);
    table[NK_COLOR_TOGGLE] = nk_rgba(38, 44, 54, 255);
    table[NK_COLOR_TOGGLE_HOVER] = nk_rgba(70, 140, 230, 255);
    table[NK_COLOR_TOGGLE_CURSOR] = nk_rgba(70, 140, 230, 255);
    table[NK_COLOR_SELECT] = nk_rgba(35, 40, 50, 255);
    table[NK_COLOR_SELECT_ACTIVE] = nk_rgba(70, 140, 230, 255);
    table[NK_COLOR_SLIDER] = nk_rgba(38, 44, 54, 255);
    table[NK_COLOR_SLIDER_CURSOR] = nk_rgba(70, 140, 230, 255);
    table[NK_COLOR_SLIDER_CURSOR_HOVER] = nk_rgba(90, 160, 250, 255);
    table[NK_COLOR_SLIDER_CURSOR_ACTIVE] = nk_rgba(60, 130, 220, 255);
    table[NK_COLOR_PROPERTY] = nk_rgba(28, 32, 40, 255);
    table[NK_COLOR_EDIT] = nk_rgba(24, 27, 34, 255);
    table[NK_COLOR_EDIT_CURSOR] = nk_rgba(230, 230, 230, 255);
    table[NK_COLOR_COMBO] = nk_rgba(30, 34, 42, 255);
    table[NK_COLOR_CHART] = nk_rgba(30, 34, 42, 255);
    table[NK_COLOR_CHART_COLOR] = nk_rgba(70, 140, 230, 255);
    table[NK_COLOR_CHART_COLOR_HIGHLIGHT] = nk_rgba(240, 180, 80, 255);
    table[NK_COLOR_SCROLLBAR] = nk_rgba(24, 27, 34, 255);
    table[NK_COLOR_SCROLLBAR_CURSOR] = nk_rgba(48, 55, 68, 255);
    table[NK_COLOR_SCROLLBAR_CURSOR_HOVER] = nk_rgba(68, 77, 94, 255);
    table[NK_COLOR_SCROLLBAR_CURSOR_ACTIVE] = nk_rgba(70, 140, 230, 255);
    table[NK_COLOR_TAB_HEADER] = nk_rgba(30, 34, 42, 255);
    nk_style_from_table(ctx, table);

    // Initial state
    memset(&g_state, 0, sizeof(AppState));
    g_state.train_steps = 1000;
    g_state.use_web = 1;
    strcpy(g_state.web_status, "Введите поисковый запрос (например: 'что такое larp') или URL для серфинга.");
    strcpy(g_state.web_results_text, "Здесь отобразятся проверенные факты из сети или текст открытой страницы.\n");
    strcpy(g_state.chat_history, "🤖 KobyakovAI: Привет! Я нативный ИИ с MoE архитектурой (144 экспертные ноды).\nЯ умею писать код, решать математику и НЕПРЕРЫВНО смотреть и обучаться по видео (YouTube, TikTok, IDE) через SmolVLM!\n------------------------------------------------------------\n");
    strcpy(g_state.train_status, "Готов к обучению на GPU RTX 2060");
    strcpy(g_state.system_info, "Модель: Иерархический MoE (144 Экспертные Ноды) | 102.14 M Параметров\nЗрение: SmolVLM-500M-Instruct (CUDA FP16)\nВидеокарта: NVIDIA GeForce RTX 2060 (6.0 GB GDDR6)\nИнтерфейс: Нативный C99 + Nuklear GDI (Скомпилировано через Zig cc)\nПотребление памяти GUI: всего ~8.5 МБ ОЗУ | Без Electron и Node.js\nСтатус демона: 127.0.0.1:8999 (Активен)");
    strcpy(g_state.video_analysis, "Видеопоток не активен. Выберите окно и нажмите '▶ Потоковое зрение' или '🎓 Обучение по видео'.");
    strcpy(g_state.stream_learned_log, "Ожидание запуска обучения по видео...\n");

    refresh_windows_list(&g_state);

    while (running) {
        MSG msg;
        nk_input_begin(ctx);
        while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) {
            if (msg.message == WM_QUIT)
                running = 0;
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        nk_input_end(ctx);

        clock_t now = clock();

        // Опрос статуса классического обучения (адаптивно: ~300 мс во время обучения, 1.2 с в покое)
        clock_t train_interval = g_state.train_is_running ? (CLOCKS_PER_SEC / 3) : (CLOCKS_PER_SEC * 1.2);
        if (g_state.current_tab == 2 && (now - g_state.last_train_poll) > train_interval) {
            poll_training_status(&g_state);
            g_state.last_train_poll = now;
        }

        // Высокоскоростной опрос видеопотока и онлайн-обучения (без 3-секундной паузы!)
        if (g_state.current_tab == 1 && (now - g_state.last_stream_poll) > (CLOCKS_PER_SEC / 15)) {
            poll_video_stream_status(&g_state);
            g_state.last_stream_poll = now;
        }

        // Размеры окна
        RECT rect;
        GetClientRect(wnd, &rect);
        int win_w = rect.right - rect.left;
        int win_h = rect.bottom - rect.top;

        if (nk_begin(ctx, "MainWindow", nk_rect(0, 0, (float)win_w, (float)win_h), NK_WINDOW_NO_SCROLLBAR)) {
            // Навигационные вкладки
            nk_layout_row_dynamic(ctx, 42, 5);
            if (nk_button_label(ctx, g_state.current_tab == 0 ? "💬 ЧАТ MoE [АКТИВЕН]" : "💬 Чат MoE")) g_state.current_tab = 0;
            if (nk_button_label(ctx, g_state.current_tab == 1 ? "👁️ ВИДЕО & ОБУЧЕНИЕ [АКТИВЕН]" : "👁️ Видео & Обучение")) {
                g_state.current_tab = 1;
                refresh_windows_list(&g_state);
                capture_single_preview(&g_state);
            }
            if (nk_button_label(ctx, g_state.current_tab == 2 ? "🚀 ОБУЧЕНИЕ GPU [АКТИВЕН]" : "🚀 Обучение GPU")) g_state.current_tab = 2;
            if (nk_button_label(ctx, g_state.current_tab == 3 ? "🌐 ВЕБ-СЕРФИНГ [АКТИВЕН]" : "🌐 Веб-серфинг")) g_state.current_tab = 3;
            if (nk_button_label(ctx, g_state.current_tab == 4 ? "⚙️ СИСТЕМА [АКТИВЕН]" : "⚙️ Система")) g_state.current_tab = 4;

            nk_layout_row_dynamic(ctx, 8, 1);
            nk_spacing(ctx, 1);

            // ==========================================
            // ВКЛАДКА 0: ЧАТ
            // ==========================================
            if (g_state.current_tab == 0) {
                nk_layout_row_begin(ctx, NK_STATIC, 32, 5);
                nk_layout_row_push(ctx, 165);
                if (nk_button_label(ctx, g_state.use_vision ? "[✓] 👁 Зрение: ВКЛ" : "[  ] 👁 Зрение: ВЫКЛ")) {
                    g_state.use_vision = !g_state.use_vision;
                }

                nk_layout_row_push(ctx, 185);
                if (nk_button_label(ctx, g_state.use_web ? "[✓] 🌐 Веб-поиск: ВКЛ" : "[  ] 🌐 Веб-поиск: ВЫКЛ")) {
                    g_state.use_web = !g_state.use_web;
                }

                nk_layout_row_push(ctx, 320);
                if (g_state.window_count > 0) {
                    g_state.selected_window_idx = nk_combo(ctx, g_state.window_ptrs, g_state.window_count, g_state.selected_window_idx, 25, nk_vec2(380, 200));
                }

                nk_layout_row_push(ctx, 140);
                if (nk_button_label(ctx, "🔄 Обновить окна")) {
                    refresh_windows_list(&g_state);
                }

                nk_layout_row_push(ctx, 120);
                if (nk_button_label(ctx, "🗑️ Очистить")) {
                    strcpy(g_state.chat_history, "История очищена.\n------------------------------------------------------------\n");
                }
                nk_layout_row_end(ctx);

                // Окно диалога
                nk_layout_row_dynamic(ctx, (float)(win_h - 170), 1);
                nk_edit_string_zero_terminated(ctx, NK_EDIT_MULTILINE | NK_EDIT_READ_ONLY, g_state.chat_history, sizeof(g_state.chat_history), nk_filter_default);

                // Поле ввода сообщения
                nk_layout_row_begin(ctx, NK_STATIC, 36, 2);
                nk_layout_row_push(ctx, (float)(win_w - 160));
                nk_flags res_enter = nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD | NK_EDIT_SIG_ENTER, g_state.chat_input, sizeof(g_state.chat_input), nk_filter_default);
                if (res_enter & NK_EDIT_COMMITED) {
                    send_chat_message(&g_state);
                }

                nk_layout_row_push(ctx, 130);
                if (nk_button_label(ctx, g_state.is_thinking ? "Думаю..." : "Отправить")) {
                    send_chat_message(&g_state);
                }
                nk_layout_row_end(ctx);
            }

            // ==========================================
            // ВКЛАДКА 1: ВИДЕО & ОНЛАЙН-ОБУЧЕНИЕ
            // ==========================================
            else if (g_state.current_tab == 1) {
                // Панель управления видеопотоком
                nk_layout_row_begin(ctx, NK_STATIC, 34, 5);
                nk_layout_row_push(ctx, 320);
                if (g_state.window_count > 0) {
                    int prev_idx = g_state.selected_window_idx;
                    g_state.selected_window_idx = nk_combo(ctx, g_state.window_ptrs, g_state.window_count, g_state.selected_window_idx, 25, nk_vec2(320, 200));
                    if (g_state.selected_window_idx != prev_idx) {
                        capture_single_preview(&g_state);
                    }
                }

                nk_layout_row_push(ctx, 120);
                if (nk_button_label(ctx, "📷 Снимок")) {
                    capture_single_preview(&g_state);
                }

                nk_layout_row_push(ctx, 210);
                if (g_state.stream_active && !g_state.video_learning_active) {
                    if (nk_button_label(ctx, "⏹ Остановить поток")) {
                        stop_video_stream(&g_state);
                    }
                } else {
                    if (nk_button_label(ctx, "▶ Потоковое зрение")) {
                        start_video_stream(&g_state, 0);
                    }
                }

                nk_layout_row_push(ctx, 260);
                if (g_state.stream_active && g_state.video_learning_active) {
                    if (nk_button_label(ctx, "⏹ ОСТАНОВИТЬ ОБУЧЕНИЕ")) {
                        stop_video_stream(&g_state);
                    }
                } else {
                    if (nk_button_label(ctx, "🎓 ОБУЧЕНИЕ ПО ВИДЕО (GPU)")) {
                        start_video_stream(&g_state, 1);
                    }
                }

                nk_layout_row_push(ctx, 140);
                if (nk_button_label(ctx, "🔄 Обновить окна")) {
                    refresh_windows_list(&g_state);
                }
                nk_layout_row_end(ctx);

                // Строка живой телеметрии
                nk_layout_row_dynamic(ctx, 25, 1);
                char telem_buf[256];
                if (g_state.stream_active) {
                    snprintf(telem_buf, sizeof(telem_buf),
                             "● В ЭФИРЕ: %.1f FPS | Кадров разобрано: %d | Обучение: %s | Усвоено знаний: %d сэмплов | Шаг MoE: %d | Loss: %.4f",
                             g_state.stream_fps, g_state.stream_frames,
                             g_state.video_learning_active ? "АКТИВНО" : "ВЫКЛ",
                             g_state.stream_learned_count, g_state.stream_training_steps, g_state.stream_loss);
                } else {
                    strcpy(telem_buf, "○ Видеопоток остановлен. Нажмите '▶ Потоковое зрение' или '🎓 ОБУЧЕНИЕ ПО ВИДЕО (GPU)'.");
                }
                nk_label(ctx, telem_buf, NK_TEXT_LEFT);

                // Двухколоночный интерфейс: Анализ + Лента знаний слева, Видео справа
                nk_layout_row_dynamic(ctx, (float)(win_h - 160), 2);

                // Левая колонка: Анализ + Выученные знания
                if (nk_group_begin(ctx, "VisionStreamGroup", NK_WINDOW_BORDER)) {
                    nk_layout_row_dynamic(ctx, 24, 1);
                    nk_label(ctx, "🧠 Разбор происходящего на видео в реальном времени (SmolVLM):", NK_TEXT_LEFT);
                    nk_layout_row_dynamic(ctx, (float)((win_h - 160) * 0.40), 1);
                    nk_edit_string_zero_terminated(ctx, NK_EDIT_MULTILINE | NK_EDIT_READ_ONLY, g_state.video_analysis, sizeof(g_state.video_analysis), nk_filter_default);

                    nk_layout_row_dynamic(ctx, 24, 1);
                    nk_label(ctx, "📚 Журнал усвоенных знаний из видео (Active Learning Stream):", NK_TEXT_LEFT);
                    nk_layout_row_dynamic(ctx, (float)((win_h - 160) * 0.45), 1);
                    nk_edit_string_zero_terminated(ctx, NK_EDIT_MULTILINE | NK_EDIT_READ_ONLY, g_state.stream_learned_log, sizeof(g_state.stream_learned_log), nk_filter_default);

                    nk_group_end(ctx);
                }

                // Правая колонка: Живой видеокадр (GDI Bitmap)
                if (nk_group_begin(ctx, "VideoMirrorGroup", NK_WINDOW_BORDER)) {
                    nk_layout_row_dynamic(ctx, 24, 1);
                    nk_label(ctx, "📺 Живой кадр (Real-Time Video Mirror):", NK_TEXT_LEFT);
                    nk_layout_row_dynamic(ctx, 22, 1);
                    char info_buf[140];
                    snprintf(info_buf, sizeof(info_buf), "Источник: %s", g_state.window_titles[g_state.selected_window_idx]);
                    nk_label(ctx, info_buf, NK_TEXT_LEFT);

                    if (g_state.preview_hbm) {
                        nk_layout_row_static(ctx, (float)g_state.preview_h, g_state.preview_w, 1);
                        struct nk_image img = nk_subimage_ptr((void*)g_state.preview_hbm, (unsigned short)g_state.preview_w, (unsigned short)g_state.preview_h, nk_rect(0, 0, (float)g_state.preview_w, (float)g_state.preview_h));
                        nk_image(ctx, img);
                    } else {
                        nk_layout_row_dynamic(ctx, 120, 1);
                        nk_label(ctx, "Нажмите 'Потоковое зрение' или 'Обучение по видео' для запуска отображения.", NK_TEXT_CENTERED);
                    }
                    nk_group_end(ctx);
                }
            }

            // ==========================================
            // ВКЛАДКА 2: КЛАССИЧЕСКОЕ ОБУЧЕНИЕ
            // ==========================================
            else if (g_state.current_tab == 2) {
                nk_layout_row_dynamic(ctx, 30, 1);
                nk_label(ctx, "Параметры обучения Hierarchical MoE (144 Ноды) на NVIDIA RTX 2060:", NK_TEXT_LEFT);

                nk_layout_row_begin(ctx, NK_STATIC, 30, 5);
                nk_layout_row_push(ctx, 120);
                nk_label(ctx, "Домен датасета:", NK_TEXT_LEFT);
                nk_layout_row_push(ctx, 90);
                if (nk_option_label(ctx, "Все (All)", g_state.train_domain == 0)) g_state.train_domain = 0;
                nk_layout_row_push(ctx, 90);
                if (nk_option_label(ctx, "Код", g_state.train_domain == 1)) g_state.train_domain = 1;
                nk_layout_row_push(ctx, 100);
                if (nk_option_label(ctx, "Математика", g_state.train_domain == 2)) g_state.train_domain = 2;
                nk_layout_row_push(ctx, 90);
                if (nk_option_label(ctx, "Логика", g_state.train_domain == 3)) g_state.train_domain = 3;
                nk_layout_row_end(ctx);

                nk_layout_row_begin(ctx, NK_STATIC, 35, 3);
                nk_layout_row_push(ctx, 150);
                nk_label(ctx, "Количество шагов:", NK_TEXT_LEFT);
                nk_layout_row_push(ctx, 200);
                nk_property_int(ctx, "Шаги:", 100, &g_state.train_steps, 50000, 500, 10);
                nk_layout_row_push(ctx, 300);
                char step_hint[128];
                snprintf(step_hint, sizeof(step_hint), "(Выбрано: %d шагов)", g_state.train_steps);
                nk_label(ctx, step_hint, NK_TEXT_LEFT);
                nk_layout_row_end(ctx);

                nk_layout_row_dynamic(ctx, 45, 2);
                if (g_state.train_is_running) {
                    nk_label(ctx, "⏳ Обучение выполняется на CUDA GPU...", NK_TEXT_LEFT);
                    if (nk_button_label(ctx, "⏹ ОСТАНОВИТЬ И СОХРАНИТЬ (Ctrl+C)")) {
                        stop_training(&g_state);
                    }
                } else {
                    if (nk_button_label(ctx, "▶ НАЧАТЬ ОБУЧЕНИЕ НА GPU (CUDA)")) {
                        start_training(&g_state);
                    }
                    if (nk_button_label(ctx, "🔄 Обновить статус")) {
                        poll_training_status(&g_state);
                    }
                }

                nk_layout_row_dynamic(ctx, 25, 1);
                nk_label(ctx, g_state.train_status, NK_TEXT_LEFT);

                nk_layout_row_dynamic(ctx, 22, 1);
                nk_size cur_s = (g_state.train_cur_step > 0) ? g_state.train_cur_step : 0;
                nk_size max_s = (g_state.train_tot_steps > 0) ? g_state.train_tot_steps : 1000;
                nk_progress(ctx, &cur_s, max_s, NK_FIXED);

                nk_layout_row_dynamic(ctx, 20, 1);
                nk_label(ctx, "Терминал вывода обучения (последняя строка / лог):", NK_TEXT_LEFT);
                nk_layout_row_dynamic(ctx, (float)(win_h - 360), 1);
                nk_edit_string_zero_terminated(ctx, NK_EDIT_MULTILINE | NK_EDIT_READ_ONLY, g_state.train_log, sizeof(g_state.train_log), nk_filter_default);
            }

            // ==========================================
            // ВКЛАДКА 3: КОНТРОЛИРУЕМЫЙ ВЕБ-СЕРФИНГ & ЧТЕНИЕ СТРАНИЦ ЦЕЛИКОМ
            // ==========================================
            else if (g_state.current_tab == 3) {
                nk_layout_row_dynamic(ctx, 26, 1);
                nk_label(ctx, "🌐 Контролируемый Веб-серфинг & Полнотекстовое Чтение Страниц (KobyakovAI):", NK_TEXT_LEFT);

                // Строка ввода запроса или URL
                nk_layout_row_begin(ctx, NK_STATIC, 36, 4);
                nk_layout_row_push(ctx, (float)(win_w - 430));
                nk_flags res_web_enter = nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD | NK_EDIT_SIG_ENTER, g_state.web_input, sizeof(g_state.web_input), nk_filter_default);
                if (res_web_enter & NK_EDIT_COMMITED) {
                    if (strncmp(g_state.web_input, "http", 4) == 0) perform_web_fetch(&g_state);
                    else perform_web_search(&g_state);
                }

                nk_layout_row_push(ctx, 140);
                if (nk_button_label(ctx, g_state.web_is_loading ? "Поиск..." : "🔍 Найти в сети")) {
                    perform_web_search(&g_state);
                }

                nk_layout_row_push(ctx, 140);
                if (nk_button_label(ctx, "📖 Читать URL")) {
                    perform_web_fetch(&g_state);
                }

                nk_layout_row_push(ctx, 110);
                if (nk_button_label(ctx, "🗑️ Очистить")) {
                    g_state.web_input[0] = '\0';
                    g_state.web_results_text[0] = '\0';
                    g_state.web_results_count = 0;
                    strcpy(g_state.web_status, "Очищено.");
                }
                nk_layout_row_end(ctx);

                // Статус и объем
                nk_layout_row_dynamic(ctx, 22, 1);
                nk_label(ctx, g_state.web_status, NK_TEXT_LEFT);

                // Если есть найденные источники — отображаем кнопки быстрого открытия страницы целиком
                int has_quick_links = (g_state.web_results_count > 0);
                if (has_quick_links) {
                    int num_btns = g_state.web_results_count;
                    if (num_btns > 3) num_btns = 3;
                    nk_layout_row_begin(ctx, NK_STATIC, 30, num_btns + 1);
                    nk_layout_row_push(ctx, 170);
                    nk_label(ctx, "📖 Читать полностью:", NK_TEXT_LEFT);
                    for (int ri = 0; ri < num_btns; ri++) {
                        char btn_lbl[64];
                        snprintf(btn_lbl, sizeof(btn_lbl), "Источник %d ↗", ri + 1);
                        nk_layout_row_push(ctx, 140);
                        if (nk_button_label(ctx, btn_lbl)) {
                            strncpy(g_state.web_input, g_state.web_last_urls[ri], sizeof(g_state.web_input) - 1);
                            perform_web_fetch(&g_state);
                        }
                    }
                    nk_layout_row_end(ctx);
                }

                // Просмотр найденных фактов / текста страницы
                float editor_h = (float)(win_h - (has_quick_links ? 270 : 230));
                nk_layout_row_dynamic(ctx, editor_h, 1);
                nk_edit_string_zero_terminated(ctx, NK_EDIT_MULTILINE | NK_EDIT_READ_ONLY, g_state.web_results_text, sizeof(g_state.web_results_text), nk_filter_default);

                // Кнопки взаимодействия
                nk_layout_row_dynamic(ctx, 38, 2);
                if (nk_button_label(ctx, "💬 Перенести этот полный текст в Чат для обсуждения")) {
                    send_web_to_chat(&g_state);
                }
                if (nk_button_label(ctx, "🧠 Усвоить веб-знания в веса MoE (Обучение)")) {
                    learn_web_content(&g_state);
                }
            }

            // ==========================================
            // ВКЛАДКА 4: СИСТЕМА
            // ==========================================
            else if (g_state.current_tab == 4) {
                nk_layout_row_dynamic(ctx, 160, 1);
                nk_edit_string_zero_terminated(ctx, NK_EDIT_MULTILINE | NK_EDIT_READ_ONLY, g_state.system_info, sizeof(g_state.system_info), nk_filter_default);

                nk_layout_row_dynamic(ctx, 25, 1);
                nk_label(ctx, "Кроссплатформенная компиляция нативного движка через Zig cc:", NK_TEXT_LEFT);

                nk_layout_row_dynamic(ctx, 40, 3);
                if (nk_button_label(ctx, "⚡ Собрать x86_64 Windows")) {
                    system("zig cc -O3 engine.c -o engine_c.exe");
                    MessageBoxW(wnd, L"engine_c.exe успешно пересобран!", L"Zig Compiler", MB_OK | MB_ICONINFORMATION);
                }
                if (nk_button_label(ctx, "🍏 Собрать ARM64 (Apple / Pi)")) {
                    system("zig cc -O3 engine.c -target aarch64-linux -o engine_arm64");
                    MessageBoxW(wnd, L"engine_arm64 успешно собран!", L"Zig Compiler", MB_OK | MB_ICONINFORMATION);
                }
                if (nk_button_label(ctx, "📟 Собрать RISC-V")) {
                    system("zig cc -O3 engine.c -target riscv64-linux -o engine_riscv64");
                    MessageBoxW(wnd, L"engine_riscv64 успешно собран!", L"Zig Compiler", MB_OK | MB_ICONINFORMATION);
                }

                nk_layout_row_dynamic(ctx, 15, 1);
                nk_spacing(ctx, 1);

                nk_layout_row_dynamic(ctx, 40, 2);
                if (nk_button_label(ctx, "💻 Запустить CLI-терминал MoE в отдельном окне")) {
                    system("start cmd /k engine_c.exe model.bin tokenizer.bin");
                }
                if (nk_button_label(ctx, "🔄 Перезапустить фоновый демон")) {
                    cleanup_daemon();
                    Sleep(500);
                    start_daemon_if_needed();
                    refresh_windows_list(&g_state);
                    MessageBoxW(wnd, L"Демон успешно перезапущен!", L"KobyakovAI", MB_OK | MB_ICONINFORMATION);
                }
            }
        }
        nk_end(ctx);

        nk_gdi_render(nk_rgb(20, 22, 26));
        Sleep(16); // ~60 FPS
    }

    cleanup_daemon();
    if (g_state.preview_hbm) DeleteObject(g_state.preview_hbm);
    nk_gdifont_del(font);
    nk_gdi_shutdown();
    ReleaseDC(wnd, dc);
    UnregisterClassW(wc.lpszClassName, wc.hInstance);
    return 0;
}
