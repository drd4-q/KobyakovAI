import os
import sys
import torch
import tiktoken
import random
from datasets import load_dataset, load_from_disk

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

# Каталог Google Диска по умолчанию
DEFAULT_GDRIVE_DATASETS = "G:/My Drive/datasets"

DATASET_CONFIGS = {
    "magicoder": {
        "local": "magicoder",
        "hf": "ise-uiuc/Magicoder-OSS-Instruct-75K",
        "domains": ["code", "all"]
    },
    "code_alpaca": {
        "local": "code_alpaca",
        "hf": "HuggingFaceH4/CodeAlpaca_20K",
        "domains": ["code", "all"]
    },
    "gsm8k": {
        "local": "gsm8k",
        "hf": "openai/gsm8k",
        "subset": "main",
        "domains": ["math", "reasoning", "logic", "all"]
    },
    "math_instruct": {
        "local": "math_instruct",
        "hf": "TIGER-Lab/MathInstruct",
        "domains": ["math", "reasoning", "all"]
    },
    "openhermes": {
        "local": "openhermes",
        "hf": "teknium/OpenHermes-2.5",
        "domains": ["reasoning", "logic", "chat", "all"]
    },
    "physics": {
        "local": "physics",
        "hf": "camel-ai/physics",
        "domains": ["physics", "all"]
    }
}

IDENTITY_EXAMPLES = [
    "User: who are you\n\nKobyakovAI:\nI am KobyakovAI, an AI assistant with a 144-node MoE architecture specialized in code, logic, and mathematics.",
    "User: Who are you?\n\nKobyakovAI:\nI am KobyakovAI, a hierarchical MoE neural intelligence designed for programming and problem solving.",
    "User: Кто ты?\n\nKobyakovAI:\nЯ KobyakovAI — нативный искусственный интеллект с MoE архитектурой (144 ноды) для написания кода, математики и анализа видео.",
    "User: What is your name?\n\nKobyakovAI:\nMy name is KobyakovAI.",
    "User: Привет\n\nKobyakovAI:\nПривет! Я KobyakovAI. Чем могу помочь: написать код, решить задачу или разобрать видео?",
    "User: hello\n\nKobyakovAI:\nHello! How can I help you today?",

    # Базовая арифметика и математика
    "User: 2+2\n\nKobyakovAI:\n4",
    "User: 2 + 2\n\nKobyakovAI:\n2 + 2 = 4",
    "User: 2=2\n\nKobyakovAI:\nTrue (2 равно 2)",
    "User: 2 = 2\n\nKobyakovAI:\nTrue (выражение истинно: 2 равно 2)",
    "User: сколько будет 2+2\n\nKobyakovAI:\n2 + 2 = 4",
    "User: сколько будет 2 + 2?\n\nKobyakovAI:\n2 + 2 = 4",
    "User: 5*5\n\nKobyakovAI:\n25",
    "User: 5 * 5\n\nKobyakovAI:\n5 * 5 = 25",
    "User: 10/2\n\nKobyakovAI:\n5",
    "User: 10 / 2\n\nKobyakovAI:\n5",
    "User: 100 - 37\n\nKobyakovAI:\n63",
    "User: 7 * 8\n\nKobyakovAI:\n56",
    "User: 2^3\n\nKobyakovAI:\n8 (2 в кубе = 8)",
    "User: 2 ** 3\n\nKobyakovAI:\n8",
    "User: sqrt(16)\n\nKobyakovAI:\n4",
    "User: 10 % 3\n\nKobyakovAI:\n1 (остаток от деления 10 на 3 равен 1)",
    "User: python 2+2\n\nKobyakovAI:\n```python\nresult = 2 + 2\nprint(result)  # 4\n```",
    "User: random number\n\nKobyakovAI:\n```python\nimport random\nprint(random.randint(1, 100))\n```",
    "User: random math number with float\n\nKobyakovAI:\n```python\nimport random\nprint(random.uniform(0.0, 100.0))\n```",
    "User: random word from vocabulary\n\nKobyakovAI:\n```python\nimport random\nwords = ['apple', 'banana', 'matrix', 'neural', 'quantum']\nprint(random.choice(words))\n```",
    "User: bash script for output something\n\nKobyakovAI:\n```bash\n#!/bin/bash\necho 'Hello from KobyakovAI!'\n```",

    # Код и Hello World
    "User: hello world\n\nKobyakovAI:\n```python\nprint('Hello, World!')\n```",
    "User: Hello World\n\nKobyakovAI:\n```python\nprint('Hello, World!')\n```",
    "User: hello world python\n\nKobyakovAI:\n```python\nprint('Hello, World!')\n```",
    "User: hello world python code\n\nKobyakovAI:\n```python\ndef hello_world():\n    print('Hello, World!')\n\nhello_world()\n```",
    "User: Python Hello world code\n\nKobyakovAI:\n```python\nprint('Hello, World!')\n```",
    "User: write code for hello world\n\nKobyakovAI:\n```python\nprint('Hello, World!')\n```",
    "User: Write a python function to print hello world\n\nKobyakovAI:\n```python\ndef hello_world():\n    print('Hello, World!')\n```",
    "User: Write a python function to add two numbers\n\nKobyakovAI:\n```python\ndef add(a, b):\n    return a + b\n```",
    "User: напиши hello world на python\n\nKobyakovAI:\n```python\nprint('Привет, Мир!')\n```",
    "User: Write hello world in C\n\nKobyakovAI:\n```c\n#include <stdio.h>\n\nint main() {\n    printf(\"Hello, World!\\n\");\n    return 0;\n}\n```",
    "User: Write hello world in Zig\n\nKobyakovAI:\n```zig\nconst std = @import(\"std\");\npub fn main() !void {\n    const stdout = std.io.getStdOut().writer();\n    try stdout.print(\"Hello, World!\\n\", .{});\n}\n```"
]

def format_item(item):
    """Преобразует пример датасета в единый формат обучения для KobyakovAI."""
    # 1. CodeAlpaca (prompt / completion) - идеален для block_size 128
    if "prompt" in item and "completion" in item:
        p = item["prompt"].strip()
        c = item["completion"].strip()
        return f"User: {p}\n\nKobyakovAI:\n{c}"

    # 2. Формат диалогов (OpenHermes / ShareGPT)
    if "conversations" in item and isinstance(item["conversations"], list):
        turns = []
        for turn in item["conversations"]:
            sender = "User" if turn.get("from") in ["human", "user", "system"] else "KobyakovAI"
            turns.append(f"{sender}: {turn.get('value', '').strip()}")
        return "\n\n".join(turns)
    
    # 3. GSM8K (question / answer)
    if "question" in item and "answer" in item:
        return f"User: {item['question'].strip()}\n\nKobyakovAI:\n{item['answer'].strip()}"
    
    # 4. CodeAlpaca / MathInstruct (instruction / input / output)
    if "instruction" in item and "output" in item:
        inp = f"\n{item['input'].strip()}" if item.get("input") else ""
        return f"User: {item['instruction'].strip()}{inp}\n\nKobyakovAI:\n{item['output'].strip()}"

    # 5. Magicoder (problem / solution)
    if "problem" in item and "solution" in item:
        p = item['problem'].strip()
        if len(p) > 250:
            p = p[:250] + "..."
        return f"User: {p}\n\nKobyakovAI:\n{item['solution'].strip()}"
    
    # 6. Camel-AI Physics (message_1 / message_2)
    if "message_1" in item and "message_2" in item:
        return f"User: {item['message_1'].strip()}\n\nKobyakovAI:\n{item['message_2'].strip()}"

    # 7. Fallback
    parts = [str(v).strip() for k, v in item.items() if isinstance(v, str) and str(v).strip()]
    return "\n\n".join(parts)

def load_single_dataset(name, cfg, gdrive_dir, split="train"):
    """Загружает датасет либо локально с Google Диска, либо потоково с HuggingFace."""
    local_path = os.path.join(gdrive_dir, cfg["local"])
    if os.path.exists(local_path):
        print(f"  [G-Drive] Загрузка локального датасета: {name} из {local_path}")
        ds = load_from_disk(local_path)
        if hasattr(ds, "keys") and split in ds:
            return ds[split]
        return ds
    else:
        print(f"  [Stream] Локальная копия не найдена, подключение потока HF: {cfg['hf']}")
        subset = cfg.get("subset")
        if subset:
            return load_dataset(cfg["hf"], subset, split=split, streaming=True)
        return load_dataset(cfg["hf"], split=split, streaming=True)

class StreamingDataLoader:
    def __init__(self, domain="all", gdrive_dir=DEFAULT_GDRIVE_DATASETS, block_size=128, batch_size=16, device='cpu'):
        self.domain = domain
        self.gdrive_dir = gdrive_dir
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.buffer = []
        self.target_tokens = (block_size + 1) * batch_size

        # Выбираем датасеты, соответствующие домену
        self.active_datasets = []
        for name, cfg in DATASET_CONFIGS.items():
            if domain == "all" or domain in cfg["domains"]:
                try:
                    ds = load_single_dataset(name, cfg, gdrive_dir)
                    self.active_datasets.append((name, ds))
                except Exception as e:
                    print(f"  Предупреждение: ошибка подключения {name}: {e}")

        if not self.active_datasets:
            print("  Внимание: нет доступных датасетов, откат к CodeAlpaca...")
            ds = load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train", streaming=True)
            self.active_datasets.append(("code_alpaca", ds))

        self.iterators = [(name, iter(ds)) for name, ds in self.active_datasets]

    def _get_next_item(self):
        while True:
            name, it = random.choice(self.iterators)
            try:
                item = next(it)
                return item
            except StopIteration:
                # Перезапускаем датасет
                for idx, (n, _) in enumerate(self.iterators):
                    if n == name:
                        for d_name, ds in self.active_datasets:
                            if d_name == name:
                                self.iterators[idx] = (name, iter(ds))
                                break

    def get_batch(self):
        """Формирует батч токенов из датасетов Google Диска."""
        while len(self.buffer) < self.target_tokens:
            if random.random() < 0.10:
                text = random.choice(IDENTITY_EXAMPLES)
            else:
                item = self._get_next_item()
                text = format_item(item)
            if text and text.strip():
                tokens = enc.encode(text)
                self.buffer.extend(tokens)
                self.buffer.extend(enc.encode("\n\n"))

        chunk = self.buffer[:self.target_tokens]
        self.buffer = self.buffer[self.target_tokens:]

        data = torch.tensor(chunk, dtype=torch.long)
        data = data.view(self.batch_size, self.block_size + 1)
        x = data[:, :self.block_size]
        y = data[:, 1:self.block_size + 1]

        return x.to(self.device), y.to(self.device)
