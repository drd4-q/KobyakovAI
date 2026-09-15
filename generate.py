import torch
import tiktoken
import os
import time
from model import CodeLanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\n============================================================")
print(f"  🧠 СПАРС МОДЕЛЬ MoE (Sparse Mixture of Experts)")
print(f"  Устройство: {device.upper()}")
print(f"============================================================")

print("Загрузка токенизатора...")
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

print("Инициализация MoE Transformer (4 эксперта, Top-2 выбор)...")
model = CodeLanguageModel(
    vocab_size=vocab_size,
    n_embd=512,
    n_head=8,
    n_layer=8,
    block_size=128,
    dropout=0.0,
    num_experts=4,
    top_k=2
)
model = model.to(device)

checkpoint_path = "moe_model_weights.pth"

print(f"Загрузка весов ({checkpoint_path})...")
if os.path.exists(checkpoint_path):
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print("✅ Веса MoE успешно загружены.")
    except Exception as e:
        print(f"⚠️ Ошибка загрузки чекпоинта: {e}")
else:
    print("⚠️  ВНИМАНИЕ: Файл moe_model_weights.pth не найден! Модель запустится с необученными весами.")
    print("    Запустите 'python train.py' для обучения MoE модели.")

if device == 'cpu':
    try:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("  [Включен режим экономии ОЗУ — INT8]")
    except Exception as e:
        print(f"  [Не удалось квантовать модель: {e}]")

model.eval()

print("\n============================================================")
print(" РОЛЬ: Мульти-экспертный Ассистент (MoE Transformer)")
print(" Введите 'exit' для выхода.")
print("============================================================\n")

thinking_mode = True

while True:
    try:
        user_input = input("\nВы (Промпт): ")
        if user_input.lower() in ['exit', 'quit', 'выход']:
            print("До свидания! 👋")
            break
        if not user_input.strip():
            continue

        if thinking_mode:
            prompt = f"Question: {user_input}\n<think>\n"
        else:
            prompt = f"Question: {user_input}\nAnswer:\n"

        tokens = enc.encode(prompt)
        if len(tokens) > model.block_size:
            tokens = tokens[-model.block_size:]
            
        context = torch.tensor([tokens], dtype=torch.long, device=device)

        print("\n🤖 MoE ИИ: ", end="", flush=True)

        with torch.no_grad():
            in_thought = thinking_mode
            if in_thought:
                print("\033[90m", end="") # Серый цвет для <think>

            full_text = ""
            for _ in range(250):
                idx_cond = context[:, -model.block_size:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :]
                probs = torch.nn.functional.softmax(logits / 0.8, dim=-1)
                
                # Top-k сэмплирование
                v, _ = torch.topk(probs, 50)
                probs[probs < v[:, [-1]]] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                idx_next = torch.multinomial(probs, num_samples=1)
                context = torch.cat((context, idx_next), dim=1)

                token_text = enc.decode([idx_next.item()])
                full_text += token_text
                print(token_text, end="", flush=True)
                time.sleep(0.01)

                if in_thought and "</think>" in full_text:
                    in_thought = False
                    print("\033[0m", end="")

        print("\033[0m")
        print("-" * 50)

    except KeyboardInterrupt:
        print("\nГенерация прервана.")
        continue
