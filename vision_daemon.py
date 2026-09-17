# -*- coding: utf-8 -*-
"""
============================================================
  👁️ KobyakovAI Vision & Brain Daemon (Local HTTP API)
  Backend service for Native C/Zig GUI (kobyakov_gui.exe)
  Runs on 127.0.0.1:8999 with zero external server dependencies.
============================================================
"""

import os
import sys
import json
import time
import signal
import subprocess
import threading
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
from PIL import Image

# Force UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from screen_vision import ScreenCapture, ScreenVisionModel

HOST = '127.0.0.1'
PORT = 8999

# Global singletons
capture = ScreenCapture()
vision_model = None
vision_lock = threading.Lock()

brain_model = None
brain_enc = None
brain_lock = threading.Lock()

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

# Training state tracking
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

def get_vision_model():
    global vision_model
    with vision_lock:
        if vision_model is None:
            print('[VisionDaemon] Инициализация SmolVLM-500M на GPU...')
            vision_model = ScreenVisionModel()
        return vision_model

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
                        # strip ansi color codes if any
                        clean_loss = ''.join(c for c in loss_val if c.isdigit() or c == '.')
                        if clean_loss:
                            train_state['loss'] = float(clean_loss)
                    train_state['status'] = f'Шаг {train_state["step"]} / {train_state["total_steps"]}'
                except Exception:
                    pass
            elif 'сохранены' in line_str or 'зафиксированы' in line_str:
                train_state['status'] = line_str

    proc.wait()
    with train_lock:
        train_state['running'] = False
        train_state['status'] = f'Обучение завершено (сохранено на шаге {train_state["step"]})'
        train_process = None
    global brain_model
    with brain_lock:
        brain_model = None  # Сброс кэша весов для мгновенной перезагрузки
    print('[VisionDaemon] Процесс обучения завершен. Веса перезагружены.')


class DaemonHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress verbose HTTP logging
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

        if self.path == '/api/capture':
            target = req.get('target', None)
            im = None
            title = 'Весь экран'
            if target:
                im, win = capture.capture_window(target)
                if win:
                    title = win.title
            if not im:
                im = capture.capture_screen()

            # Save preview bitmap (BMP for zero-dependency native Windows GDI rendering in C GUI)
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
            prompt = req.get('prompt', 'Опиши подробно, что происходит на видео или в этом окне.')
            im = None
            title = 'Весь экран'
            if target:
                im, win = capture.capture_window(target)
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
            target = req.get('target', None)

            visual_context = ''
            if use_vision:
                im = None
                if target:
                    im, win = capture.capture_window(target)
                if not im:
                    im = capture.capture_screen()
                v_model = get_vision_model()
                visual_summary = v_model.analyze(im, prompt='Кратко опиши содержимое экрана/окна, код, текст и элементы.')
                visual_context = f'[Визуальный контекст экрана (SmolVLM)]: {visual_summary}\n\n'

            full_prompt = f'{visual_context}User: {user_prompt}\n\nKobyakovAI:\n'
            resp = ""

            # 1. Быстрый инференс в памяти через PyTorch MoE
            bm, b_enc = get_brain_model()
            if bm is not None and b_enc is not None:
                try:
                    import torch
                    m, dev = bm
                    input_ids = b_enc.encode(full_prompt)
                    x = torch.tensor([input_ids], dtype=torch.long, device=dev)
                    out_tokens = []
                    with torch.no_grad():
                        for _ in range(120):
                            cond_x = x[:, -128:]
                            with torch.amp.autocast(device_type="cuda", enabled=(dev == "cuda")):
                                logits, _ = m(cond_x)
                            next_tok = torch.argmax(logits[0, -1, :]).item()
                            if next_tok in (50256, 12982): # <|endoftext|> or User:
                                break
                            out_tokens.append(next_tok)
                            x = torch.cat([x, torch.tensor([[next_tok]], device=dev)], dim=1)
                    resp = b_enc.decode(out_tokens).strip()
                except Exception as e:
                    print(f"[VisionDaemon] Ошибка ин-мемори инференса: {e}")

            # 2. Фолбэк на native C engine если инференс в памяти не сработал
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

        elif self.path == '/api/train/start':
            global train_process
            with train_lock:
                if train_process and train_process.poll() is None:
                    self._send_json({'status': 'already_running', 'message': 'Обучение уже идет'})
                    return

                steps = int(req.get('steps', 1000))
                domain = req.get('domain', 'all')
                
                cmd = [sys.executable, 'train_gpu.py', '--steps', str(steps), '--domain', domain]
                train_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
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
            with train_lock:
                if train_process and train_process.poll() is None:
                    print('[VisionDaemon] Посылаем сигнал SIGINT (Ctrl+C) процессу обучения...')
                    train_process.send_signal(signal.CTRL_C_EVENT)
                    self._send_json({'status': 'stopping', 'message': 'Сигнал остановки отправлен, идет сохранение весов...'})
                else:
                    self._send_json({'status': 'not_running', 'message': 'Обучение не запущено'})

        else:
            self._send_json({'error': 'Not found'}, 404)


def run_daemon():
    print('=' * 65)
    print(f'  🚀 KobyakovAI Vision & Brain Daemon running on http://{HOST}:{PORT}')
    print('  Ready to serve Native C/Zig GUI (kobyakov_gui.exe)')
    print('=' * 65)
    server = ThreadingHTTPServer((HOST, PORT), DaemonHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\n[VisionDaemon] Сервер остановлен.')

if __name__ == '__main__':
    run_daemon()
