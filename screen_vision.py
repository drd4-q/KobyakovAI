# -*- coding: utf-8 -*-
"""
============================================================
  👁️ KobyakovAI Vision Module: Screen and Window Intelligence
  Capabilities:
  - Capture Entire Desktop Screen
  - List Active Application Windows (HWND, Titles, Dimensions)
  - Capture Specific Selected Window (even background / partially occluded)
  - Local Vision-Language Model Analysis (SmolVLM-500M on CUDA GPU / CPU)
============================================================
"""

import os
import sys
import argparse
import ctypes
from ctypes import wintypes
from typing import List, Optional, Tuple
from PIL import Image

# Force UTF-8 encoding on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Windows GDI / User32 API Setup
user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

# Configure 64-bit function signatures
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetWindowRect.restype = wintypes.BOOL
user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
user32.PrintWindow.restype = wintypes.BOOL
user32.GetSystemMetrics.argtypes = [ctypes.c_int]
user32.GetSystemMetrics.restype = ctypes.c_int
user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.ReleaseDC.restype = ctypes.c_int

gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
gdi32.BitBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD]
gdi32.BitBlt.restype = wintypes.BOOL
gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteObject.restype = wintypes.BOOL
gdi32.DeleteDC.argtypes = [wintypes.HDC]
gdi32.DeleteDC.restype = wintypes.BOOL

class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ('biSize', wintypes.DWORD),
        ('biWidth', wintypes.LONG),
        ('biHeight', wintypes.LONG),
        ('biPlanes', wintypes.WORD),
        ('biBitCount', wintypes.WORD),
        ('biCompression', wintypes.DWORD),
        ('biSizeImage', wintypes.DWORD),
        ('biXPelsPerMeter', wintypes.LONG),
        ('biYPelsPerMeter', wintypes.LONG),
        ('biClrUsed', wintypes.DWORD),
        ('biClrImportant', wintypes.DWORD)
    ]

gdi32.GetDIBits.argtypes = [
    wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
    ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT
]
gdi32.GetDIBits.restype = ctypes.c_int


class WindowInfo:
    def __init__(self, index: int, hwnd: int, title: str, width: int, height: int):
        self.index = index
        self.hwnd = hwnd
        self.title = title
        self.width = width
        self.height = height

    def __repr__(self):
        return f'[{self.index}] HWND {self.hwnd} | {self.width}x{self.height} | {self.title}'


class ScreenCapture:
    """
    Нативный модуль захвата экрана и отдельных окон в Windows без внешних зависимостей.
    """
    def __init__(self):
        self._ensure_user_desktop()

    def _ensure_user_desktop(self):
        """Гарантирует подключение к интерактивному пользовательскому рабочему столу (default)."""
        try:
            hdesk = user32.OpenDesktopW('default', 0, False, 0x01FF)
            if hdesk:
                user32.SetThreadDesktop(hdesk)
        except Exception:
            pass

    def list_windows(self) -> List[WindowInfo]:
        """Возвращает список всех видимых окон приложений."""
        self._ensure_user_desktop()
        raw_windows = []

        def callback(hwnd, lparam):
            if user32.IsWindowVisible(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buf, length + 1)
                    title = buf.value.strip()
                    if title and title not in ['Program Manager', 'Windows Input Experience']:
                        rect = wintypes.RECT()
                        user32.GetWindowRect(hwnd, ctypes.byref(rect))
                        w = rect.right - rect.left
                        h = rect.bottom - rect.top
                        if w > 80 and h > 80:
                            raw_windows.append((hwnd, title, w, h))
            return True

        cb = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)(callback)
        user32.EnumWindows(cb, 0)

        # Remove duplicate titles and assign indices
        windows = []
        seen = set()
        idx = 1
        for hwnd, title, w, h in raw_windows:
            key = (title, w, h)
            if key not in seen:
                seen.add(key)
                windows.append(WindowInfo(idx, hwnd, title, w, h))
                idx += 1
        return windows

    def capture_window(self, target) -> Tuple[Optional[Image.Image], Optional[WindowInfo]]:
        """
        Захватывает конкретное окно с поддержкой HWND и аппаратного ускорения Chrome/YouTube:
        - target может быть HWND (int), строкой HWND, индексом окна или строкой заголовка.
        - Использует высокоскоростной DWM Screen BitBlt (захватывает 60 FPS аппаратное видео в Chrome/YouTube без черных экранов)
        - С автоматическим фолбэком на PrintWindow(PW_RENDERFULLCONTENT).
        """
        self._ensure_user_desktop()
        windows = self.list_windows()
        selected: Optional[WindowInfo] = None

        # 1. Поиск по HWND или индексу
        if isinstance(target, int):
            for w in windows:
                if w.hwnd == target or w.index == target:
                    selected = w
                    break
            # Если окно не найдено в списке, но HWND валиден в Windows
            if not selected and user32.IsWindow(target):
                rect = wintypes.RECT()
                user32.GetWindowRect(target, ctypes.byref(rect))
                w = rect.right - rect.left
                h = rect.bottom - rect.top
                length = user32.GetWindowTextLengthW(target)
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(target, buf, length + 1)
                selected = WindowInfo(0, target, buf.value, w, h)

        elif isinstance(target, str):
            target_strip = target.strip()
            if target_strip.isdigit():
                num = int(target_strip)
                for w in windows:
                    if w.hwnd == num or w.index == num:
                        selected = w
                        break
                if not selected and user32.IsWindow(num):
                    rect = wintypes.RECT()
                    user32.GetWindowRect(num, ctypes.byref(rect))
                    w = rect.right - rect.left
                    h = rect.bottom - rect.top
                    length = user32.GetWindowTextLengthW(num)
                    buf = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(num, buf, length + 1)
                    selected = WindowInfo(0, num, buf.value, w, h)

            if not selected:
                target_lower = target_strip.lower()
                # 1. Точное или частичное совпадение
                for w in windows:
                    if target_lower in w.title.lower():
                        selected = w
                        break
                # 2. Нечеткое совпадение по ключевым словам (YouTube, Chrome, TikTok, Discord)
                if not selected:
                    keywords = [k for k in ["youtube", "chrome", "tiktok", "discord", "code", "visual studio"] if k in target_lower]
                    if keywords:
                        for w in windows:
                            if any(k in w.title.lower() for k in keywords):
                                selected = w
                                break

        if not selected:
            return None, None

        hwnd = selected.hwnd
        tw = selected.width
        th = selected.height
        if tw <= 10 or th <= 10:
            return None, selected

        im = None

        # Способ 1: Высокоскоростной DWM Screen BitBlt (захватывает live видео YouTube/Chrome на 60 FPS)
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        wx = rect.left
        wy = rect.top
        ww = rect.right - rect.left
        wh = rect.bottom - rect.top

        if not user32.IsIconic(hwnd) and ww > 40 and wh > 40:
            try:
                hdc_screen = user32.GetDC(None)
                hdc_mem_screen = gdi32.CreateCompatibleDC(hdc_screen)
                hbm_screen = gdi32.CreateCompatibleBitmap(hdc_screen, ww, wh)
                old_bm_s = gdi32.SelectObject(hdc_mem_screen, hbm_screen)

                # CAPTUREBLT | SRCCOPY
                gdi32.BitBlt(hdc_mem_screen, 0, 0, ww, wh, hdc_screen, wx, wy, 0x00CC0020 | 0x40000000)

                bmi = BITMAPINFOHEADER()
                bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
                bmi.biWidth = ww
                bmi.biHeight = -wh
                bmi.biPlanes = 1
                bmi.biBitCount = 32
                bmi.biCompression = 0
                buf_size = ww * wh * 4
                buf = ctypes.create_string_buffer(buf_size)
                gdi32.GetDIBits(hdc_mem_screen, hbm_screen, 0, wh, buf, ctypes.byref(bmi), 0)

                cand_im = Image.frombuffer('RGBA', (ww, wh), buf, 'raw', 'BGRA', 0, 1).convert('RGB')

                gdi32.SelectObject(hdc_mem_screen, old_bm_s)
                gdi32.DeleteObject(hbm_screen)
                gdi32.DeleteDC(hdc_mem_screen)
                user32.ReleaseDC(None, hdc_screen)

                im = cand_im
            except Exception:
                im = None

        # Способ 2: PrintWindow (если окно свернуто или перекрыто)
        if im is None:
            try:
                hdc_wnd = user32.GetDC(hwnd)
                hdc_mem = gdi32.CreateCompatibleDC(hdc_wnd)
                hbm = gdi32.CreateCompatibleBitmap(hdc_wnd, tw, th)
                old_bm = gdi32.SelectObject(hdc_mem, hbm)

                res = user32.PrintWindow(hwnd, hdc_mem, 2) # PW_RENDERFULLCONTENT
                if not res:
                    user32.PrintWindow(hwnd, hdc_mem, 0)

                bmi = BITMAPINFOHEADER()
                bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
                bmi.biWidth = tw
                bmi.biHeight = -th
                bmi.biPlanes = 1
                bmi.biBitCount = 32
                bmi.biCompression = 0

                buf_size = tw * th * 4
                buf = ctypes.create_string_buffer(buf_size)
                gdi32.GetDIBits(hdc_mem, hbm, 0, th, buf, ctypes.byref(bmi), 0)

                im = Image.frombuffer('RGBA', (tw, th), buf, 'raw', 'BGRA', 0, 1).convert('RGB')

                gdi32.SelectObject(hdc_mem, old_bm)
                gdi32.DeleteObject(hbm)
                gdi32.DeleteDC(hdc_mem)
                user32.ReleaseDC(hwnd, hdc_wnd)
            except Exception:
                pass

        return im, selected

    def capture_screen(self) -> Image.Image:
        """Захватывает весь экран рабочего стола."""
        self._ensure_user_desktop()
        w = user32.GetSystemMetrics(0) # SM_CXSCREEN
        h = user32.GetSystemMetrics(1) # SM_CYSCREEN

        hdc_screen = user32.GetDC(None)
        hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)
        hbm = gdi32.CreateCompatibleBitmap(hdc_screen, w, h)
        old_bm = gdi32.SelectObject(hdc_mem, hbm)

        # SRCCOPY | CAPTUREBLT
        gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_screen, 0, 0, 0x00CC0020 | 0x40000000)

        bmi = BITMAPINFOHEADER()
        bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.biWidth = w
        bmi.biHeight = -h
        bmi.biPlanes = 1
        bmi.biBitCount = 32
        bmi.biCompression = 0

        buf_size = w * h * 4
        buf = ctypes.create_string_buffer(buf_size)
        gdi32.GetDIBits(hdc_mem, hbm, 0, h, buf, ctypes.byref(bmi), 0)

        im = Image.frombuffer('RGBA', (w, h), buf, 'raw', 'BGRA', 0, 1)
        im = im.convert('RGB')

        gdi32.SelectObject(hdc_mem, old_bm)
        gdi32.DeleteObject(hbm)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(None, hdc_screen)

        return im


class ScreenVisionModel:
    """
    Модель компьютерного зрения для анализа экрана и окон.
    Использует SmolVLM-500M-Instruct (ускорение CUDA FP16 на RTX 2060).
    """
    def __init__(self, model_id: str = 'HuggingFaceTB/SmolVLM-500M-Instruct', device: Optional[str] = None):
        import torch
        from transformers import AutoProcessor, Idefics3ForConditionalGeneration

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f'[👁️ Vision] Инициализация модели {model_id} на {self.device.upper()}...')
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.model = Idefics3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()
        print('[👁️ Vision] Модель готова к анализу!')

    def analyze(self, image: Image.Image, prompt: str = 'Подробно опиши, что изображено в этом окне/на экране.', max_new_tokens: int = 250) -> str:
        """Анализирует изображение и отвечает на текстовый вопрос/промпт."""
        import torch

        # Оптимизируем разрешение для быстрого инференса
        img_copy = image.copy()
        img_copy.thumbnail((1024, 1024))

        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image'},
                    {'type': 'text', 'text': prompt}
                ]
            }
        ]

        formatted_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=formatted_prompt, images=[img_copy], return_tensors='pt').to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Очищаем системный префикс диалога
        if 'Assistant:' in output_text:
            output_text = output_text.split('Assistant:', 1)[-1].strip()
        elif 'assistant\n' in output_text:
            output_text = output_text.split('assistant\n', 1)[-1].strip()

        return output_text


def interactive_mode(capture: ScreenCapture, vision: ScreenVisionModel):
    """Интерактивный шелл для непрерывного просмотра экрана/окон."""
    print('=' * 65)
    print('  👁️ KobyakovAI - Screen & Window Vision Shell')
    print('  Команды:')
    print('  :list          - показать список открытых окон')
    print('  :window <id>   - выбрать окно по номеру или названию')
    print('  :screen        - переключиться на захват всего экрана')
    print('  :save <file>   - сохранить текущий кадр в PNG')
    print('  :exit          - выход')
    print('=' * 65)

    current_target = None # None means full screen
    current_title = 'Весь экран (Desktop Screen)'

    while True:
        try:
            user_input = input(f'\n\033[1m[👁️ {current_title}]\033[0m \nВопрос: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nВыход.')
            break

        if not user_input:
            continue
        if user_input.lower() in [':exit', 'exit', 'quit']:
            print('Выход.')
            break

        if user_input == ':list':
            wins = capture.list_windows()
            print('\nОткрытые окна:')
            for w in wins:
                print(f'  {w}')
            continue

        if user_input.startswith(':window'):
            arg = user_input[7:].strip()
            if not arg:
                print('Укажите номер или название окна, например: :window 1 или :window Discord')
                continue
            im, win = capture.capture_window(arg)
            if win:
                current_target = win.hwnd
                current_title = f'{win.title} ({win.width}x{win.height})'
                print(f'\033[92m[OK] Выбрано окно:\033[0m {current_title}')
            else:
                print(f'\033[91m[Ошибка] Окно "{arg}" не найдено.\033[0m Используйте :list')
            continue

        if user_input == ':screen':
            current_target = None
            current_title = 'Весь экран (Desktop Screen)'
            print('\033[92m[OK] Переключено на весь экран.\033[0m')
            continue

        if user_input.startswith(':save'):
            filename = user_input[5:].strip() or 'capture.png'
            if current_target is None:
                im = capture.capture_screen()
            else:
                im, _ = capture.capture_window(current_target)
            if im:
                im.save(filename)
                print(f'\033[92m[OK] Снимок сохранён в {filename}\033[0m')
            continue

        # Захват свежего кадра
        print('📸 Захватываю кадр...')
        if current_target is None:
            im = capture.capture_screen()
        else:
            im, _ = capture.capture_window(current_target)
            if not im:
                print('\033[93m[Предупреждение] Окно недоступно, захватываю весь экран.\033[0m')
                im = capture.capture_screen()

        # Анализ
        print('🧠 Анализирую изображение...')
        answer = vision.analyze(im, prompt=user_input)
        print(f'\n\033[96m🤖 KobyakovAI:\033[0m\n{answer}')


def main():
    parser = argparse.ArgumentParser(description='KobyakovAI Screen & Window Vision Module')
    parser.add_argument('--list', action='store_true', help='Показать список всех активных окон')
    parser.add_argument('--window', type=str, default=None, help='Номер, HWND или часть названия окна для захвата')
    parser.add_argument('--screen', action='store_true', help='Захватить весь экран рабочего стола')
    parser.add_argument('--prompt', type=str, default='Опиши подробно, что видно в этом окне/на экране.', help='Вопрос или задание модели')
    parser.add_argument('--save', type=str, default=None, help='Сохранить снимок в файл (например, screen.png)')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'], help='Устройство (cuda/cpu)')
    parser.add_argument('-i', '--interactive', action='store_true', help='Интерактивный режим вопросов и ответов')

    args = parser.parse_args()

    capture = ScreenCapture()

    if args.list:
        windows = capture.list_windows()
        print('=' * 65)
        print(f'  Активные окна ({len(windows)}):')
        print('=' * 65)
        for w in windows:
            print(f'  {w}')
        print('=' * 65)
        return

    # If interactive mode is requested
    if args.interactive:
        vision = ScreenVisionModel(device=args.device)
        interactive_mode(capture, vision)
        return

    # Single-shot execution
    im = None
    info_title = 'Весь экран'
    if args.window:
        im, win = capture.capture_window(args.window)
        if win:
            info_title = win.title
            print(f'Захвачено окно: {win.title} ({win.width}x{win.height})')
        else:
            print(f'Окно "{args.window}" не найдено. Список окон:')
            for w in capture.list_windows()[:10]:
                print(f'  {w}')
            return
    else:
        im = capture.capture_screen()
        print(f'Захвачен весь экран ({im.width}x{im.height})')

    if args.save:
        im.save(args.save)
        print(f'Снимок сохранён в {args.save}')

    vision = ScreenVisionModel(device=args.device)
    print(f'\nВопрос: {args.prompt}')
    print('Анализ...')
    answer = vision.analyze(im, prompt=args.prompt)
    print(f'\n🤖 \033[96mKobyakovAI Vision:\033[0m\n{answer}\n')


if __name__ == '__main__':
    main()
