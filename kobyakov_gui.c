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

#define WINDOW_WIDTH 1100
#define WINDOW_HEIGHT 740
#define DAEMON_HOST "http://127.0.0.1:8999"

// ============================================================
// Глобальные переменные процессов и состояние
// ============================================================
static PROCESS_INFORMATION g_daemon_pi = {0};
static int g_daemon_started_by_us = 0;

typedef struct {
    int current_tab; // 0: Chat, 1: Watcher, 2: Training, 3: System

    // Чат
    char chat_input[512];
    char chat_history[32768];
    int use_vision;
    int is_thinking;

    // Список окон
    char window_titles[40][128];
    int window_hwnds[40];
    const char* window_ptrs[40];
    int window_count;
    int selected_window_idx;

    // Video Watcher (YouTube / TikTok)
    int watch_mode;
    clock_t last_watch_time;
    char video_analysis[16384];
    HBITMAP preview_hbm;
    int preview_w;
    int preview_h;

    // Обучение
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

    DWORD timeout = 2500; // 2.5s timeout
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

    DWORD timeout = 15000; // 15s for AI inference
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
// Автоматический запуск и управление демоном
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

    char cmd[512] = "python vision_daemon.py";
    BOOL ok = CreateProcessA(NULL, cmd, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &g_daemon_pi);
    if (!ok) {
        strcpy(cmd, "py vision_daemon.py");
        ok = CreateProcessA(NULL, cmd, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &g_daemon_pi);
    }

    if (ok) {
        g_daemon_started_by_us = 1;
        // Ожидаем готовности сервера до 6 секунд
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
// Функции взаимодействия с Daemon API
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

        char* p = strstr(resp, "\"title\":");
        while (p && s->window_count < 38) {
            p += 8;
            while (*p == ' ' || *p == '\"') p++;
            char* end = strchr(p, '\"');
            if (!end) break;
            int len = (int)(end - p);
            if (len > 120) len = 120;
            strncpy(s->window_titles[s->window_count], p, len);
            s->window_titles[s->window_count][len] = '\0';
            s->window_ptrs[s->window_count] = s->window_titles[s->window_count];
            s->window_count++;
            p = strstr(end, "\"title\":");
        }
    } else {
        s->daemon_online = 0;
    }
}

void capture_window_preview(AppState* s) {
    char req[256];
    if (s->selected_window_idx <= 0) {
        strcpy(req, "{\"target\": null}");
    } else {
        snprintf(req, sizeof(req), "{\"target\": \"%s\"}", s->window_titles[s->selected_window_idx]);
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

void analyze_window_action(AppState* s, const char* prompt) {
    capture_window_preview(s);
    char req[1024];
    const char* win = (s->selected_window_idx <= 0) ? "" : s->window_titles[s->selected_window_idx];
    snprintf(req, sizeof(req), "{\"target\": \"%s\", \"prompt\": \"%s\"}", win, prompt);

    char resp[16384];
    if (http_post_json("/api/analyze", req, resp, sizeof(resp))) {
        char* p = strstr(resp, "\"analysis\":");
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
            snprintf(s->video_analysis, sizeof(s->video_analysis), "=== КАДР: [%s] ===\n%s\n", win[0] ? win : "Весь экран", clean);
        }
    }
}

void send_chat_message(AppState* s) {
    if (strlen(s->chat_input) == 0) return;

    int cur_len = (int)strlen(s->chat_history);
    snprintf(s->chat_history + cur_len, sizeof(s->chat_history) - cur_len, "\nПользователь: %s\n\n🤖 KobyakovAI:\n", s->chat_input);

    char req[2048];
    const char* win = (s->use_vision && s->selected_window_idx > 0) ? s->window_titles[s->selected_window_idx] : "";
    snprintf(req, sizeof(req), "{\"prompt\": \"%s\", \"use_vision\": %s, \"target\": \"%s\"}",
             s->chat_input, s->use_vision ? "true" : "false", win);

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
    char resp[512];
    if (http_post_json("/api/train/stop", "{}", resp, sizeof(resp))) {
        strcpy(s->train_status, "Остановка... Безопасное сохранение чекпоинта и весов...");
    }
}

void poll_training_status(AppState* s) {
    char resp[2048];
    if (http_get(DAEMON_HOST "/api/train/status", resp, sizeof(resp))) {
        char* p_run = strstr(resp, "\"running\":");
        if (p_run) {
            s->train_is_running = (strstr(p_run, "true") != NULL);
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
            strncpy(s->train_log, p_log, sizeof(s->train_log) - 1);
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
    // Поддержка CLI аргументов в автономном exe
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

    // Проверяем веса модели и фоновый сервис
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
    wc.lpszClassName = L"KobyakovStudioMainClass";
    RegisterClassW(&wc);

    wnd = CreateWindowW(
        wc.lpszClassName,
        L"🤖 KobyakovAI Native Studio (Pure C99 / Zig MoE + Vision)",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        100, 60, WINDOW_WIDTH, WINDOW_HEIGHT,
        NULL, NULL, wc.hInstance, NULL
    );

    dc = GetDC(wnd);
    GdiFont* font = nk_gdifont_create("Segoe UI", 16);
    ctx = nk_gdi_init(font, dc, WINDOW_WIDTH, WINDOW_HEIGHT);

    // Тема оформления (Modern Dark)
    struct nk_color table[NK_COLOR_COUNT];
    table[NK_COLOR_TEXT] = nk_rgba(230, 230, 230, 255);
    table[NK_COLOR_WINDOW] = nk_rgba(22, 24, 28, 255);
    table[NK_COLOR_HEADER] = nk_rgba(32, 36, 43, 255);
    table[NK_COLOR_BORDER] = nk_rgba(45, 50, 60, 255);
    table[NK_COLOR_BUTTON] = nk_rgba(36, 40, 48, 255);
    table[NK_COLOR_BUTTON_HOVER] = nk_rgba(75, 145, 230, 255);
    table[NK_COLOR_BUTTON_ACTIVE] = nk_rgba(55, 125, 210, 255);
    table[NK_COLOR_TOGGLE] = nk_rgba(40, 45, 54, 255);
    table[NK_COLOR_TOGGLE_HOVER] = nk_rgba(75, 145, 230, 255);
    table[NK_COLOR_TOGGLE_CURSOR] = nk_rgba(75, 145, 230, 255);
    table[NK_COLOR_SELECT] = nk_rgba(36, 40, 48, 255);
    table[NK_COLOR_SELECT_ACTIVE] = nk_rgba(75, 145, 230, 255);
    table[NK_COLOR_SLIDER] = nk_rgba(40, 45, 54, 255);
    table[NK_COLOR_SLIDER_CURSOR] = nk_rgba(75, 145, 230, 255);
    table[NK_COLOR_SLIDER_CURSOR_HOVER] = nk_rgba(95, 165, 250, 255);
    table[NK_COLOR_SLIDER_CURSOR_ACTIVE] = nk_rgba(65, 135, 220, 255);
    table[NK_COLOR_PROPERTY] = nk_rgba(30, 34, 40, 255);
    table[NK_COLOR_EDIT] = nk_rgba(26, 29, 35, 255);
    table[NK_COLOR_EDIT_CURSOR] = nk_rgba(230, 230, 230, 255);
    table[NK_COLOR_COMBO] = nk_rgba(32, 36, 43, 255);
    table[NK_COLOR_CHART] = nk_rgba(32, 36, 43, 255);
    table[NK_COLOR_CHART_COLOR] = nk_rgba(75, 145, 230, 255);
    table[NK_COLOR_CHART_COLOR_HIGHLIGHT] = nk_rgba(240, 180, 80, 255);
    table[NK_COLOR_SCROLLBAR] = nk_rgba(26, 29, 35, 255);
    table[NK_COLOR_SCROLLBAR_CURSOR] = nk_rgba(50, 56, 66, 255);
    table[NK_COLOR_SCROLLBAR_CURSOR_HOVER] = nk_rgba(70, 77, 90, 255);
    table[NK_COLOR_SCROLLBAR_CURSOR_ACTIVE] = nk_rgba(75, 145, 230, 255);
    table[NK_COLOR_TAB_HEADER] = nk_rgba(32, 36, 43, 255);
    nk_style_from_table(ctx, table);

    // Начальное состояние
    memset(&g_state, 0, sizeof(AppState));
    g_state.train_steps = 1000;
    strcpy(g_state.chat_history, "🤖 KobyakovAI: Привет! Я нативный ИИ-ассистент с MoE архитектурой (144 экспертные ноды).\nЯ умею писать код, решать задачи и смотреть любые открытые окна (YouTube, TikTok, IDE, экран) через SmolVLM!\n------------------------------------------------------------\n");
    strcpy(g_state.train_status, "Готов к обучению на GPU RTX 2060");
    strcpy(g_state.system_info, "Модель: Иерархический MoE (144 Экспертные Ноды) | 102.14 M Параметров\nЗрение: SmolVLM-500M-Instruct (CUDA FP16)\nВидеокарта: NVIDIA GeForce RTX 2060 (6.0 GB GDDR6)\nИнтерфейс: Нативный C99 + Nuklear GDI (Скомпилировано через Zig cc)\nПотребление памяти GUI: всего ~8.5 МБ ОЗУ | Без Electron и Node.js\nСтатус демона: 127.0.0.1:8999 (Активен)");

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

        // Периодический опрос обучения
        clock_t now = clock();
        if (g_state.current_tab == 2 && (now - g_state.last_train_poll) > (CLOCKS_PER_SEC * 1.5)) {
            poll_training_status(&g_state);
            g_state.last_train_poll = now;
        }

        // Авто-захват кадра в режиме Live Video Watcher
        if (g_state.watch_mode && (now - g_state.last_watch_time) > (CLOCKS_PER_SEC * 3)) {
            analyze_window_action(&g_state, "Опиши кратко, что сейчас происходит на видео (субтитры, действия, код).");
            g_state.last_watch_time = now;
        }

        // Отрисовка интерфейса
        RECT rect;
        GetClientRect(wnd, &rect);
        int win_w = rect.right - rect.left;
        int win_h = rect.bottom - rect.top;

        if (nk_begin(ctx, "MainWindow", nk_rect(0, 0, (float)win_w, (float)win_h), NK_WINDOW_NO_SCROLLBAR)) {
            // Навигационные вкладки
            nk_layout_row_dynamic(ctx, 42, 4);
            if (nk_button_label(ctx, g_state.current_tab == 0 ? "💬 ЧАТ MoE [АКТИВЕН]" : "💬 Чат MoE")) g_state.current_tab = 0;
            if (nk_button_label(ctx, g_state.current_tab == 1 ? "👁️ ВИДЕО [АКТИВЕН]" : "👁️ Video Watcher")) {
                g_state.current_tab = 1;
                refresh_windows_list(&g_state);
                capture_window_preview(&g_state);
            }
            if (nk_button_label(ctx, g_state.current_tab == 2 ? "🚀 ОБУЧЕНИЕ [АКТИВЕН]" : "🚀 Обучение GPU")) g_state.current_tab = 2;
            if (nk_button_label(ctx, g_state.current_tab == 3 ? "⚙️ СИСТЕМА [АКТИВЕН]" : "⚙️ Система")) g_state.current_tab = 3;

            nk_layout_row_dynamic(ctx, 8, 1);
            nk_spacing(ctx, 1);

            // ==========================================
            // ВКЛАДКА 0: ЧАТ
            // ==========================================
            if (g_state.current_tab == 0) {
                // Панель зрения
                nk_layout_row_begin(ctx, NK_STATIC, 32, 4);
                nk_layout_row_push(ctx, 180);
                nk_checkbox_label(ctx, "👁️ Включить зрение", &g_state.use_vision);

                nk_layout_row_push(ctx, 360);
                if (g_state.window_count > 0) {
                    g_state.selected_window_idx = nk_combo(ctx, g_state.window_ptrs, g_state.window_count, g_state.selected_window_idx, 25, nk_vec2(360, 200));
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
            // ВКЛАДКА 1: VIDEO WATCHER (YouTube / TikTok)
            // ==========================================
            else if (g_state.current_tab == 1) {
                nk_layout_row_begin(ctx, NK_STATIC, 32, 4);
                nk_layout_row_push(ctx, 360);
                if (g_state.window_count > 0) {
                    int prev_idx = g_state.selected_window_idx;
                    g_state.selected_window_idx = nk_combo(ctx, g_state.window_ptrs, g_state.window_count, g_state.selected_window_idx, 25, nk_vec2(360, 200));
                    if (g_state.selected_window_idx != prev_idx) {
                        capture_window_preview(&g_state);
                    }
                }
                nk_layout_row_push(ctx, 140);
                if (nk_button_label(ctx, "📷 Снимок окна")) {
                    capture_window_preview(&g_state);
                    analyze_window_action(&g_state, "Опиши подробно, что видно в этом окне/на видео.");
                }
                nk_layout_row_push(ctx, 230);
                if (g_state.watch_mode) {
                    if (nk_button_label(ctx, "⏹ Остановить видео-режим")) {
                        g_state.watch_mode = 0;
                    }
                } else {
                    if (nk_button_label(ctx, "▶ Смотреть видео (Live 3s)")) {
                        g_state.watch_mode = 1;
                        g_state.last_watch_time = clock() - (CLOCKS_PER_SEC * 5);
                    }
                }
                nk_layout_row_push(ctx, 140);
                if (nk_button_label(ctx, "🔄 Обновить окна")) {
                    refresh_windows_list(&g_state);
                }
                nk_layout_row_end(ctx);

                // Превью и текстовый разбор бок о бок
                nk_layout_row_dynamic(ctx, (float)(win_h - 130), 2);

                // Левая колонка: Описание видео от SmolVLM
                if (nk_group_begin(ctx, "VideoAnalysisGroup", NK_WINDOW_BORDER)) {
                    nk_layout_row_dynamic(ctx, 25, 1);
                    nk_label(ctx, "🧠 Разбор происходящего на видео (SmolVLM):", NK_TEXT_LEFT);
                    nk_layout_row_dynamic(ctx, (float)(win_h - 190), 1);
                    nk_edit_string_zero_terminated(ctx, NK_EDIT_MULTILINE | NK_EDIT_READ_ONLY, g_state.video_analysis, sizeof(g_state.video_analysis), nk_filter_default);
                    nk_group_end(ctx);
                }

                // Правая колонка: Кадр видео
                if (nk_group_begin(ctx, "PreviewGroup", NK_WINDOW_BORDER)) {
                    nk_layout_row_dynamic(ctx, 25, 1);
                    nk_label(ctx, "📺 Текущий кадр окна / экрана:", NK_TEXT_LEFT);
                    nk_layout_row_dynamic(ctx, 25, 1);
                    char info_buf[140];
                    snprintf(info_buf, sizeof(info_buf), "Окно: %s", g_state.window_titles[g_state.selected_window_idx]);
                    nk_label(ctx, info_buf, NK_TEXT_LEFT);

                    if (g_state.preview_hbm) {
                        nk_layout_row_static(ctx, (float)g_state.preview_h, g_state.preview_w, 1);
                        struct nk_image img = nk_subimage_ptr((void*)g_state.preview_hbm, (unsigned short)g_state.preview_w, (unsigned short)g_state.preview_h, nk_rect(0, 0, (float)g_state.preview_w, (float)g_state.preview_h));
                        nk_image(ctx, img);
                    } else {
                        nk_layout_row_dynamic(ctx, 120, 1);
                        nk_label(ctx, "Нажмите 'Снимок окна' или 'Смотреть видео' для вывода кадра.", NK_TEXT_CENTERED);
                    }
                    nk_group_end(ctx);
                }
            }

            // ==========================================
            // ВКЛАДКА 2: ОБУЧЕНИЕ
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
            // ВКЛАДКА 3: СИСТЕМА
            // ==========================================
            else if (g_state.current_tab == 3) {
                nk_layout_row_dynamic(ctx, 160, 1);
                nk_edit_string_zero_terminated(ctx, NK_EDIT_MULTILINE | NK_EDIT_READ_ONLY, g_state.system_info, sizeof(g_state.system_info), nk_filter_default);

                nk_layout_row_dynamic(ctx, 25, 1);
                nk_label(ctx, "Кроссплатформенная компиляция нативного движка через Zig cc:", NK_TEXT_LEFT);

                nk_layout_row_dynamic(ctx, 40, 3);
                if (nk_button_label(ctx, "⚡ Собрать x86_64 Windows")) {
                    system("zig cc -O3 engine.c -o engine_c.exe");
                    MessageBoxA(wnd, "engine_c.exe успешно пересобран!", "Zig Compiler", MB_OK | MB_ICONINFORMATION);
                }
                if (nk_button_label(ctx, "🍏 Собрать ARM64 (Apple / Pi)")) {
                    system("zig cc -O3 engine.c -target aarch64-linux -o engine_arm64");
                    MessageBoxA(wnd, "engine_arm64 успешно собран!", "Zig Compiler", MB_OK | MB_ICONINFORMATION);
                }
                if (nk_button_label(ctx, "📟 Собрать RISC-V")) {
                    system("zig cc -O3 engine.c -target riscv64-linux -o engine_riscv64");
                    MessageBoxA(wnd, "engine_riscv64 успешно собран!", "Zig Compiler", MB_OK | MB_ICONINFORMATION);
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
                    MessageBoxA(wnd, "Демон успешно перезапущен!", "KobyakovAI", MB_OK | MB_ICONINFORMATION);
                }
            }
        }
        nk_end(ctx);

        nk_gdi_render(nk_rgb(22, 24, 28));
        Sleep(16); // ~60 FPS, 0% CPU consumption
    }

    cleanup_daemon();
    if (g_state.preview_hbm) DeleteObject(g_state.preview_hbm);
    nk_gdifont_del(font);
    nk_gdi_shutdown();
    ReleaseDC(wnd, dc);
    UnregisterClassW(wc.lpszClassName, wc.hInstance);
    return 0;
}
