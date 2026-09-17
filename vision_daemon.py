# -*- coding: utf-8 -*-
"""
============================================================
  👁️ KobyakovAI Vision, Brain & Active Video Learning Daemon
  Backend service for Native C/Zig GUI (KobyakovAI.exe)
  Runs on 127.0.0.1:8999 with zero external server dependencies.
  Features:
    - SmolVLM-500M GPU acceleration (RTX 2060)
    - Real-Time Continuous Video Stream (no artificial delay)
    - Active Video Learning (Online MoE Fine-Tuning from video frames)
============================================================
"""

import os
import sys
import json
import time
import signal
import queue
import subprocess
import threading
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from PIL import Image

# Force UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from screen_vision import ScreenCapture, ScreenVisionModel
from web_surfer import WebSurfer, SemanticMeaningSelector

HOST = '127.0.0.1'
PORT = 8999

# Global singletons
capture = ScreenCapture()
web_surfer = WebSurfer()
vision_model = None
vision_lock = threading.Lock()

brain_model = None
brain_enc = None
brain_lock = threading.Lock()

def get_vision_model():
    global vision_model
    with vision_lock:
        if vision_model is None:
            print('[VisionDaemon] Инициализация SmolVLM-500M на GPU...')
            vision_model = ScreenVisionModel()
        return vision_model

def get_brain_model():
    global brain_model, brain_enc
    with brain_lock:
        if brain_model is None:
            try:
                import torch
                import tiktoken
                from model import CodeLanguageModel
                device = "cuda" if torch.cuda.is_available() else "cpu"
                enc = tiktoken.get_encoding("gpt2")
                m = CodeLanguageModel(
                    vocab_size=enc.n_vocab,
                    n_embd=256, n_head=8, n_layer=2, block_size=128, dropout=0.0,
                    num_parents=4, num_sub_parents=4, num_leaf_experts=9, top_k=2, mult=2
                )
                weights_path = "moe_model_weights.pth"
                if os.path.exists(weights_path):
                    sd = torch.load(weights_path, map_location=device, weights_only=True)
                    m.load_state_dict(sd, strict=False)
                    print(f"[VisionDaemon] Загружены веса KobyakovAI MoE (144 ноды) на {device.upper()}")
                m.to(device)
                m.eval()
                brain_model = (m, device)
                brain_enc = enc
            except Exception as e:
                print(f"[VisionDaemon] Не удалось инициализировать PyTorch Brain: {e}")
                return None, None
        return brain_model, brain_enc

# ============================================================
# Активное онлайн-дообучение на основе видео (Active Video Learning)
# ============================================================
class ActiveVideoLearner:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.device = None
        self.optimizer = None
        self.scaler = None
        self.enc = None
        self.step_count = 0
        self.current_loss = 0.0

    def _ensure_init(self):
        if self.model is None:
            import torch
            import torch.optim as optim
            import tiktoken
            from model import CodeLanguageModel
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.enc = tiktoken.get_encoding("gpt2")
            self.model = CodeLanguageModel(
                vocab_size=self.enc.n_vocab,
                n_embd=256, n_head=8, n_layer=2, block_size=128, dropout=0.05,
                num_parents=4, num_sub_parents=4, num_leaf_experts=9, top_k=2, mult=2
            )
            weights_path = "moe_model_weights.pth"
            if os.path.exists(weights_path):
                try:
                    sd = torch.load(weights_path, map_location=self.device, weights_only=True)
                    self.model.load_state_dict(sd, strict=False)
                    print(f"[VideoLearner] Веса MoE загружены для онлайн-дообучения на {self.device.upper()}")
                except Exception as e:
                    print(f"[VideoLearner] Ошибка загрузки весов: {e}")
            self.model.to(self.device)
            self.model.train()
            self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.01)
            self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device == "cuda"))

    def train_sample(self, text):
        with self.lock:
            self._ensure_init()
            import torch
            tokens = self.enc.encode(text)
            if len(tokens) < 6:
                return None

            block_size = 128
            if len(tokens) < block_size + 1:
                tokens = tokens + [50256] * (block_size + 1 - len(tokens))

            chunk = tokens[:block_size + 1]
            x = torch.tensor([chunk[:block_size]], dtype=torch.long, device=self.device)
            y = torch.tensor([chunk[1:block_size + 1]], dtype=torch.long, device=self.device)

            with torch.amp.autocast("cuda", enabled=(self.device == "cuda")):
                logits, loss = self.model(x, y)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.step_count += 1
            loss_val = round(loss.item(), 4)
            self.current_loss = loss_val

            if self.step_count % 5 == 0:
                self.save_checkpoint()

            return loss_val

    def save_checkpoint(self):
        if self.model is not None:
            import torch
            torch.save(self.model.state_dict(), "moe_model_weights.pth")
            try:
                from export_weights import export_model_bin
                export_model_bin(self.model, "model.bin")
            except Exception as e:
                print(f"[VideoLearner] Ошибка экспорта model.bin: {e}")
            global brain_model
            with brain_lock:
                brain_model = None
            print(f"[VideoLearner] Веса сохранены (Шаг онлайн-обучения {self.step_count})")

active_learner = ActiveVideoLearner()

# ============================================================
# Потоковый движок реального времени (Continuous Video Engine)
# ============================================================
class ContinuousVideoEngine:
    def __init__(self, capture_obj, learner_obj):
        self.capture = capture_obj
        self.learner = learner_obj
        self.running = False
        self.learning_enabled = False
        self.target = None
        self.lock = threading.Lock()

        self.frames_captured = 0
        self.frames_analyzed = 0
        self.fps = 0.0
        self.latest_analysis = "Поток не запущен"
        self.current_loss = 0.0
        self.training_steps = 0
        self.learned_log = []
        self.last_analysis_text = ""

        self.capture_thread = None
        self.analysis_thread = None
        self.frame_queue = queue.Queue(maxsize=1)

    def start(self, target=None, hwnd=None, enable_learning=False):
        with self.lock:
            self.hwnd = hwnd if (hwnd and isinstance(hwnd, int) and hwnd > 0) else None
            self.target = target if (target and target != "Весь экран (Desktop Screen)") else None
            self.learning_enabled = enable_learning
            if self.running:
                return
            self.running = True
            self.frames_captured = 0
            self.frames_analyzed = 0
            self.latest_analysis = "Подключение к видеопотоку..."
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
            self.capture_thread.start()
            self.analysis_thread.start()
            print(f"[VideoEngine] Поток запущен. HWND: {self.hwnd}, Заголовок: {self.target}, Обучение: {self.learning_enabled}")

    def stop(self):
        with self.lock:
            if not self.running:
                return
            self.running = False
            self.latest_analysis = "Видеопоток остановлен."
        if self.learning_enabled and self.learner:
            self.learner.save_checkpoint()
        print("[VideoEngine] Поток остановлен.")

    def _capture_loop(self):
        start_t = time.time()
        fc = 0
        while self.running:
            try:
                target_key = self.hwnd if self.hwnd else self.target
                im = None
                if target_key:
                    im, win = self.capture.capture_window(target_key)
                if not im:
                    im = self.capture.capture_screen()

                # Сохраняем кадр для нативного рендеринга в C GUI (20-25 FPS)
                thumb = im.copy()
                thumb.thumbnail((480, 270), Image.Resampling.LANCZOS)
                thumb.save("preview.bmp", "BMP")

                if not self.frame_queue.full():
                    try:
                        self.frame_queue.put_nowait(im)
                    except queue.Full:
                        pass

                self.frames_captured += 1
                fc += 1
                el = time.time() - start_t
                if el >= 1.0:
                    self.fps = round(fc / el, 1)
                    fc = 0
                    start_t = time.time()

                time.sleep(0.04) # ~25 FPS
            except Exception:
                time.sleep(0.08)

    def _analysis_loop(self):
        while self.running:
            try:
                try:
                    im = self.frame_queue.get(timeout=0.4)
                except queue.Empty:
                    continue

                v_model = get_vision_model()
                if self.learning_enabled:
                    prompt = (
                        "Выдели и подробно объясни весь программный код, математику, формулы, алгоритмы или знания на этом кадре. "
                        "Если виден код, выпиши точный код. Если объяснение или субтитры, сформулируй суть."
                    )
                else:
                    prompt = "Кратко опиши, что сейчас происходит на видео (субтитры, код, действия, элементы интерфейса)."

                analysis = v_model.analyze(im, prompt=prompt)
                clean = analysis.strip()

                with self.lock:
                    self.latest_analysis = clean
                    self.frames_analyzed += 1

                # Обучение по кадрам видео
                if self.learning_enabled and len(clean) > 15:
                    if clean != self.last_analysis_text:
                        self.last_analysis_text = clean
                        train_text = f"User: Что показано на видео?\n\nKobyakovAI:\n{clean}\n"
                        loss_val = self.learner.train_sample(train_text)
                        if loss_val is not None:
                            now_str = time.strftime("%H:%M:%S")
                            snippet = clean.replace("\n", " ")[:50]
                            log_entry = f"[{now_str}] Усвоено: {snippet}... (Loss: {loss_val})"
                            with self.lock:
                                self.current_loss = loss_val
                                self.training_steps = self.learner.step_count
                                self.learned_log.append(log_entry)
                                if len(self.learned_log) > 25:
                                    self.learned_log.pop(0)

                            try:
                                with open("video_train_dataset.txt", "a", encoding="utf-8") as vf:
                                    vf.write(train_text + "\n" + ("=" * 40) + "\n")
                            except Exception:
                                pass
            except Exception as e:
                print(f"[VideoEngine] Ошибка в цикле анализа: {e}")
                time.sleep(0.2)

video_engine = ContinuousVideoEngine(capture, active_learner)

# ============================================================
# Традиционный трекер классического обучения
# ============================================================
train_process = None
train_lock = threading.Lock()
train_state = {
    'running': False,
    'step': 0,
    'total_steps': 0,
    'loss': 0.0,
    'status': 'Готов к обучению',
    'last_log': ''
}

def train_monitor_thread(proc, steps):
    global train_process, train_state
    for line in proc.stdout:
        line_str = line.strip()
        if not line_str:
            continue
        with train_lock:
            train_state['last_log'] = line_str
            if 'Step' in line_str and '/' in line_str:
                try:
                    parts = line_str.split('|')
                    step_part = parts[0].replace('Step', '').strip()
                    cur, tot = step_part.split('/')
                    train_state['step'] = int(cur.strip())
                    train_state['total_steps'] = int(tot.strip())
                    if len(parts) > 1 and 'Loss:' in parts[1]:
                        loss_val = parts[1].replace('Loss:', '').strip()
                        clean_loss = ''.join(c for c in loss_val if c.isdigit() or c == '.')
                        if clean_loss:
                            train_state['loss'] = float(clean_loss)
                    train_state['status'] = f'Шаг {train_state["step"]} / {train_state["total_steps"]}'
                except Exception:
                    pass
            elif 'сохранены' in line_str or 'зафиксированы' in line_str or 'Остановка' in line_str:
                train_state['status'] = line_str

    proc.wait()
    if os.path.exists("train_stop.flag"):
        try:
            os.remove("train_stop.flag")
        except Exception:
            pass

    with train_lock:
        train_state['running'] = False
        if 'Останов' in train_state.get('status', '') or 'Останов' in train_state.get('last_log', ''):
            train_state['status'] = f'Обучение остановлено и зафиксировано (на шаге {train_state["step"]})'
        else:
            train_state['status'] = f'Обучение завершено (сохранено на шаге {train_state["step"]})'
        train_process = None
    global brain_model
    with brain_lock:
        brain_model = None
    print('[VisionDaemon] Процесс обучения завершен. Веса перезагружены.')


# ============================================================
# HTTP API Обработчик
# ============================================================
class DaemonHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/api/windows':
            wins = capture.list_windows()
            data = [
                {'index': w.index, 'hwnd': w.hwnd, 'title': w.title, 'width': w.width, 'height': w.height}
                for w in wins
            ]
            self._send_json({'status': 'ok', 'windows': data})

        elif self.path == '/api/train/status':
            with train_lock:
                self._send_json(dict(train_state))

        elif self.path == '/api/video_stream/status':
            with video_engine.lock:
                self._send_json({
                    'status': 'ok',
                    'streaming': video_engine.running,
                    'learning': video_engine.learning_enabled,
                    'target': video_engine.target or 'Весь экран',
                    'frames_captured': video_engine.frames_captured,
                    'frames_analyzed': video_engine.frames_analyzed,
                    'fps': video_engine.fps,
                    'latest_analysis': video_engine.latest_analysis,
                    'training_steps': video_engine.training_steps,
                    'current_loss': video_engine.current_loss,
                    'learned_count': len(video_engine.learned_log),
                    'learned_log': '\n'.join(video_engine.learned_log)
                })

        elif self.path == '/api/web/status':
            self._send_json({
                'status': 'ok',
                'last_query': web_surfer.last_query,
                'last_results': web_surfer.last_results,
                'last_page_url': web_surfer.last_page_url,
                'last_page_title': web_surfer.last_page_title,
                'history': web_surfer.history
            })

        elif self.path == '/api/ping':
            self._send_json({'status': 'ok', 'message': 'KobyakovAI Daemon is active'})
        else:
            self._send_json({'error': 'Not found'}, 404)

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body_bytes = self.rfile.read(content_length) if content_length > 0 else b'{}'
        try:
            req = json.loads(body_bytes.decode('utf-8'))
        except Exception:
            req = {}

        if self.path == '/api/video_stream/start':
            target = req.get('target', None)
            hwnd = req.get('hwnd', None)
            learning = bool(req.get('enable_learning', False))
            video_engine.start(target=target, hwnd=hwnd, enable_learning=learning)
            self._send_json({'status': 'ok', 'streaming': True, 'learning': learning})

        elif self.path == '/api/video_stream/stop':
            video_engine.stop()
            self._send_json({'status': 'ok', 'streaming': False})

        elif self.path == '/api/capture':
            target = req.get('target', None)
            hwnd = req.get('hwnd', None)
            target_key = hwnd if (hwnd and isinstance(hwnd, int) and hwnd > 0) else target
            im = None
            title = 'Весь экран'
            if target_key and target_key != 'Весь экран (Desktop Screen)':
                im, win = capture.capture_window(target_key)
                if win:
                    title = win.title
            if not im:
                im = capture.capture_screen()

            thumb = im.copy()
            thumb.thumbnail((480, 270), Image.Resampling.LANCZOS)
            thumb.save('preview.bmp', 'BMP')

            self._send_json({
                'status': 'ok',
                'title': title,
                'width': thumb.width,
                'height': thumb.height,
                'path': 'preview.bmp'
            })

        elif self.path == '/api/analyze':
            target = req.get('target', None)
            hwnd = req.get('hwnd', None)
            target_key = hwnd if (hwnd and isinstance(hwnd, int) and hwnd > 0) else target
            prompt = req.get('prompt', 'Опиши подробно, что происходит на видео или в этом окне.')
            im = None
            title = 'Весь экран'
            if target_key and target_key != 'Весь экран (Desktop Screen)':
                im, win = capture.capture_window(target_key)
                if win:
                    title = win.title
            if not im:
                im = capture.capture_screen()

            v_model = get_vision_model()
            answer = v_model.analyze(im, prompt=prompt)

            self._send_json({
                'status': 'ok',
                'title': title,
                'prompt': prompt,
                'analysis': answer
            })

        elif self.path == '/api/chat':
            user_prompt = req.get('prompt', '')
            use_vision = req.get('use_vision', False)
            use_web = req.get('use_web', False)
            target = req.get('target', None)
            hwnd = req.get('hwnd', None)
            target_key = hwnd if (hwnd and isinstance(hwnd, int) and hwnd > 0) else target

            visual_context = ''
            if use_vision:
                im = None
                if target_key and target_key != 'Весь экран (Desktop Screen)':
                    im, win = capture.capture_window(target_key)
                if not im:
                    im = capture.capture_screen()
                v_model = get_vision_model()
                visual_summary = v_model.analyze(im, prompt='Кратко опиши содержимое экрана/окна, код, текст и элементы.')
                visual_context = f'[Визуальный контекст экрана (SmolVLM)]: {visual_summary}\n\n'

            semantic_reasoning_resp = ''
            web_context = ''

            # 1. Проверка прямого URL
            direct_urls = web_surfer.extract_urls(user_prompt)
            if direct_urls:
                target_url = direct_urls[0]
                print(f"[VisionDaemon] Обнаружен прямой URL: {target_url}. Чтение страницы на 100%...")
                page_text = web_surfer.fetch_url(target_url, max_chars=None)
                page_title = web_surfer.last_page_title
                relevant_blocks = SemanticMeaningSelector.select_relevant_blocks(user_prompt, page_text, top_n=3)

                reasoning_lines = [
                    f"📖 [Страница прочитана на 100%]: {page_title}",
                    f"🔗 [URL]: {target_url}",
                    f"📊 [Объем]: {web_surfer.last_page_chars} символов ({web_surfer.last_page_lines} строк)\n",
                    "🧠 [Семантический отбор разделов по смыслу]:"
                ]
                for bi, b in enumerate(relevant_blocks, 1):
                    reasoning_lines.append(f"  {bi}. [Раздел: {b['heading']} | Релевантность: {b['score']:.1f}]:")
                    reasoning_lines.append(f"     {b['text'][:400]}...")

                reasoning_lines.append("\n💭 [Логическое мышление и вывод]:")
                if relevant_blocks:
                    reasoning_lines.append(f"По запросу «{user_prompt}» суть найдена в разделе «{relevant_blocks[0]['heading']}»:\n{relevant_blocks[0]['text']}")
                    if len(relevant_blocks) > 1:
                        reasoning_lines.append(f"\nДополнительные детали:\n{relevant_blocks[1]['text']}")
                else:
                    reasoning_lines.append(page_text[:2000])

                semantic_reasoning_resp = "\n".join(reasoning_lines)

            # 2. Контролируемый веб-поиск и семантический отбор
            elif use_web and user_prompt.strip():
                print(f"[VisionDaemon] Контролируемый поиск и смысловой отбор: «{user_prompt}»...")
                deep_res = web_surfer.search_with_deep_crawl(user_prompt, max_results=3, crawl_top=True)
                search_results = deep_res.get('results', [])
                top_text = deep_res.get('top_page_text', '')
                top_title = deep_res.get('top_page_title', '')
                relevant_blocks = deep_res.get('relevant_blocks', [])

                if search_results:
                    domains = ", ".join(dict.fromkeys(r.get('domain', 'web') for r in search_results))
                    reasoning_lines = [
                        f"🔍 [Поиск в сети]: «{user_prompt}»",
                        f"🌐 [Источники]: {domains}"
                    ]
                    if top_title and top_text:
                        reasoning_lines.append(f"📖 [Первоисточник прочитан полностью]: «{top_title}» ({len(top_text)} символов)\n")
                    else:
                        reasoning_lines.append("")

                    if relevant_blocks:
                        reasoning_lines.append("🧠 [Семантический отбор по смыслу]:")
                        for bi, b in enumerate(relevant_blocks, 1):
                            reasoning_lines.append(f"• [Раздел: {b['heading']} | Смысловой вес: {b['score']:.1f}]:\n  {b['text'][:350]}")
                        reasoning_lines.append("\n💭 [Логическое мышление и ответ]:")
                        reasoning_lines.append(f"На основе семантического анализа первоисточника по запросу «{user_prompt}»:")
                        reasoning_lines.append(relevant_blocks[0]['text'])
                        if len(relevant_blocks) > 1 and len(relevant_blocks[1]['text']) > 50:
                            reasoning_lines.append(f"\nСущественные детали:\n{relevant_blocks[1]['text']}")
                    else:
                        snippets_text = "\n".join(f"• {r['snippet']}" for r in search_results)
                        reasoning_lines.append("Факты из проверенных источников:\n" + snippets_text)

                    semantic_reasoning_resp = "\n".join(reasoning_lines)

            # Если сформирован структурированный ответ с семантическим отбором
            if semantic_reasoning_resp:
                resp = semantic_reasoning_resp
            else:
                # Полноценная генерация MoE без лимитов (до 512 токенов)
                full_prompt = f"{visual_context}User: {user_prompt}\n\nKobyakovAI:\n"
                resp = ""
                bm, b_enc = get_brain_model()
                if bm is not None and b_enc is not None:
                    try:
                        import torch
                        m, dev = bm
                        input_ids = b_enc.encode(full_prompt)
                        x = torch.tensor([input_ids[-100:]], dtype=torch.long, device=dev)
                        out_tokens = []
                        max_gen_tokens = 512
                        with torch.no_grad():
                            for _ in range(max_gen_tokens):
                                cond_x = x[:, -128:]
                                with torch.amp.autocast(device_type="cuda", enabled=(dev == "cuda")):
                                    logits, _ = m(cond_x)
                                cur_logits = logits[0, -1, :].clone()

                                for pt in set(out_tokens[-35:]):
                                    if cur_logits[pt] > 0:
                                        cur_logits[pt] /= 1.4
                                    else:
                                        cur_logits[pt] *= 1.4

                                probs = torch.softmax(cur_logits / 0.65, dim=-1)
                                top_k_probs, top_k_idx = torch.topk(probs, 40)
                                top_k_probs = top_k_probs / torch.sum(top_k_probs)
                                next_tok = top_k_idx[torch.multinomial(top_k_probs, 1)].item()

                                if next_tok in (50256, 12982):
                                    break
                                out_tokens.append(next_tok)
                                x = torch.cat([x, torch.tensor([[next_tok]], device=dev)], dim=1)

                                if len(out_tokens) >= 8 and out_tokens[-3:] == out_tokens[-6:-3]:
                                    break

                        resp = b_enc.decode(out_tokens).strip()
                    except Exception as e:
                        print(f"[VisionDaemon] Ошибка ин-мемори инференса: {e}")

                # Фолбэк на C engine
                if not resp:
                    try:
                        proc = subprocess.run(
                            ['engine_c.exe', 'model.bin', 'tokenizer.bin'],
                            input=user_prompt.encode('utf-8'),
                            capture_output=True,
                            timeout=10
                        )
                        raw_out = proc.stdout.decode('utf-8', errors='replace')
                        if 'KobyakovAI:' in raw_out:
                            resp = raw_out.split('KobyakovAI:', 1)[-1].strip()
                            if '[' in resp and 'tok/s' in resp:
                                resp = resp.split('[Generated')[0].strip()
                        else:
                            resp = raw_out.strip()
                    except Exception:
                        pass

            if not resp:
                resp = "..." 

            self._send_json({
                'status': 'ok',
                'visual_context': visual_context,
                'response': resp
            })

        elif self.path == '/api/web/search':
            q = req.get('query', '')
            max_res = int(req.get('max_results', 4))
            deep = bool(req.get('deep_crawl', False))
            if deep:
                deep_res = web_surfer.search_with_deep_crawl(q, max_results=max_res, crawl_top=True)
                self._send_json({
                    'status': 'ok',
                    'query': q,
                    'results': deep_res['results'],
                    'top_page_text': deep_res['top_page_text'],
                    'top_page_title': deep_res['top_page_title'],
                    'top_page_url': deep_res['top_page_url']
                })
            else:
                results = web_surfer.search(q, max_results=max_res)
                self._send_json({'status': 'ok', 'query': q, 'results': results})

        elif self.path == '/api/web/fetch':
            url = req.get('url', '')
            content = web_surfer.fetch_url(url, max_chars=None)
            self._send_json({
                'status': 'ok',
                'url': url,
                'title': web_surfer.last_page_title,
                'chars': web_surfer.last_page_chars,
                'lines': web_surfer.last_page_lines,
                'content': content
            })

        elif self.path == '/api/web/learn':
            text = req.get('text', '')
            source = req.get('source', 'Web')
            loss_val = None
            if text and len(text) > 20:
                train_text = f"User: Что известно о {source}?\n\nKobyakovAI:\n{text[:350]}\n"
                loss_val = active_learner.train_sample(train_text)
            self._send_json({'status': 'ok', 'loss': loss_val, 'steps': active_learner.step_count})

        elif self.path == '/api/train/start':
            global train_process
            with train_lock:
                if train_process and train_process.poll() is None:
                    self._send_json({'status': 'already_running', 'message': 'Обучение уже идет'})
                    return

                steps = int(req.get('steps', 1000))
                domain = req.get('domain', 'all')

                cmd = [sys.executable, '-u', 'train_gpu.py', '--steps', str(steps), '--domain', domain]
                train_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
                train_state['running'] = True
                train_state['step'] = 0
                train_state['total_steps'] = steps
                train_state['loss'] = 0.0
                train_state['status'] = f'Запуск обучения ({steps} шагов, {domain})...'

                t = threading.Thread(target=train_monitor_thread, args=(train_process, steps), daemon=True)
                t.start()

                self._send_json({'status': 'started', 'steps': steps, 'domain': domain})

        elif self.path == '/api/train/stop':
            # 1. Записываем файл-флаг остановки (мгновенно считывается train_gpu.py на следующем шаге)
            try:
                with open('train_stop.flag', 'w', encoding='utf-8') as f:
                    f.write('stop')
                print('[VisionDaemon] Создан файл-флаг train_stop.flag')
            except Exception as e:
                print(f'[VisionDaemon] Ошибка создания флага остановки: {e}')

            with train_lock:
                if train_process and train_process.poll() is None:
                    print('[VisionDaemon] Посылаем сигнал остановки процессу обучения...')
                    try:
                        if hasattr(signal, 'CTRL_BREAK_EVENT'):
                            train_process.send_signal(signal.CTRL_BREAK_EVENT)
                        elif hasattr(signal, 'CTRL_C_EVENT'):
                            train_process.send_signal(signal.CTRL_C_EVENT)
                    except Exception as e:
                        print(f'[VisionDaemon] Ошибка отправки сигнала: {e}')

                    train_state['status'] = 'Остановка... Сохранение весов...'

                    # Фоновый watchdog для гарантии завершения (до 6 секунд на сохранение чекпоинта)
                    def stop_watchdog(p):
                        for _ in range(30):
                            time.sleep(0.2)
                            if p.poll() is not None:
                                return
                        if p.poll() is None:
                            print('[VisionDaemon] Процесс не завершился в срок, принудительное закрытие...')
                            try:
                                p.terminate()
                                time.sleep(0.5)
                                if p.poll() is None:
                                    p.kill()
                            except Exception:
                                pass
                    threading.Thread(target=stop_watchdog, args=(train_process,), daemon=True).start()

                    self._send_json({'status': 'stopping', 'message': 'Сигнал остановки отправлен, идет сохранение весов...'})
                else:
                    self._send_json({'status': 'not_running', 'message': 'Обучение не запущено'})

        else:
            self._send_json({'error': 'Not found'}, 404)


def run_daemon():
    print('=' * 65)
    print(f'  🚀 KobyakovAI Vision, Brain & Active Learning Daemon: http://{HOST}:{PORT}')
    print('  Ready to serve KobyakovAI.exe Studio GUI')
    print('=' * 65)
    # Фоновый прогрев весов MoE модели для мгновенного первого ответа
    threading.Thread(target=get_brain_model, daemon=True).start()
    server = ThreadingHTTPServer((HOST, PORT), DaemonHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n[VisionDaemon] Сервер остановлен.')

if __name__ == '__main__':
    run_daemon()
