import os
import sys
import struct
import signal

# Force UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.optim as optim
import tiktoken
from model import CodeLanguageModel
from data_loader import StreamingDataLoader
from export_weights import export_model_bin

def train_cuda(
    domain="code",
    steps=1000,
    batch_size=16,
    block_size=128,
    lr=3e-4,
    gdrive_dir=None,
    output_bin="model.bin"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("  🤖 KobyakovAI - CUDA GPU TRAINING PIPELINE -> NATIVE C ENGINE")
    print(f"  Device: {device.upper()}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print("=" * 60)

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    n_embd = 256
    n_head = 8
    n_layer = 2
    num_parents = 4
    num_sub_parents = 4
    num_leaf_experts = 9
    top_k = 2
    mult = 2
    total_leaf_nodes = num_parents * num_sub_parents * num_leaf_experts

    print(f"Initializing Hierarchical MoE ({n_layer} layers, {num_parents} parents x {num_sub_parents} sub x {num_leaf_experts} leaves = {total_leaf_nodes} nodes)...")
    model = CodeLanguageModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=0.1,
        num_parents=num_parents,
        num_sub_parents=num_sub_parents,
        num_leaf_experts=num_leaf_experts,
        top_k=top_k,
        mult=mult
    )
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {num_params:.2f} M")

    checkpoint_pth = "moe_model_weights.pth"
    if os.path.exists(checkpoint_pth):
        try:
            model.load_state_dict(torch.load(checkpoint_pth, map_location=device, weights_only=True), strict=False)
            print(f"Loaded existing checkpoint from {checkpoint_pth}")
        except Exception as e:
            print(f"Starting fresh: {e}")

    loader_dir = gdrive_dir if gdrive_dir else "G:/My Drive/datasets"
    loader = StreamingDataLoader(domain=domain, gdrive_dir=loader_dir, block_size=block_size, batch_size=batch_size, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Удаляем устаревший флаг остановки перед началом
    if os.path.exists("train_stop.flag"):
        try:
            os.remove("train_stop.flag")
        except Exception:
            pass

    stop_requested = False
    def on_stop_signal(sig=None, frame=None):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, on_stop_signal)
    signal.signal(signal.SIGTERM, on_stop_signal)
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, on_stop_signal)

    print(f"\nTraining for {steps} steps on {device.upper()}... (Кнопка 'Остановить' в GUI или Ctrl+C сохранит веса)", flush=True)
    model.train()

    completed_steps = 0
    try:
        for step in range(1, steps + 1):
            if stop_requested or os.path.exists("train_stop.flag"):
                try:
                    if os.path.exists("train_stop.flag"):
                        os.remove("train_stop.flag")
                except Exception:
                    pass
                print(f"\n\n\033[93m[⚠️ Остановка по запросу] Обучение прервано на шаге {completed_steps} из {steps}!\033[0m", flush=True)
                print("\033[93mИдет сохранение накопленного прогресса в веса...\033[0m", flush=True)
                break

            x, y = loader.get_batch()

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            completed_steps = step

            if step == 1 or step % 50 == 0 or step == steps:
                print(f"Step {step:4d} / {steps} | Loss: \033[92m{loss.item():.4f}\033[0m", flush=True)

            # Периодическое автосохранение каждые 250 шагов
            if step % 250 == 0 and step < steps:
                torch.save(model.state_dict(), checkpoint_pth)
                model.eval()
                export_model_bin(model, output_bin)
                model.train()
                print(f"  \033[90m[Автосохранение чекпоинта и model.bin на шаге {step}]\033[0m", flush=True)

        if not stop_requested and not os.path.exists("train_stop.flag") and completed_steps == steps:
            print("\nTraining completed!", flush=True)
    except KeyboardInterrupt:
        print(f"\n\n\033[93m[⚠️ Остановка по Ctrl+C] Обучение прервано пользователем на шаге {completed_steps} из {steps}!\033[0m", flush=True)
        print("\033[93mИдет сохранение накопленного прогресса в веса...\033[0m", flush=True)

    # Save PyTorch checkpoint and export to C engine
    if completed_steps > 0:
        # Временно блокируем повторный сигнал во время записи на диск
        prev_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
        prev_sigterm = signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            torch.save(model.state_dict(), checkpoint_pth)
            print(f"\033[92m[✓] PyTorch веса сохранены в {checkpoint_pth} (пройдено шагов: {completed_steps})\033[0m", flush=True)

            # Export directly to native C model.bin
            print(f"Экспорт весов в нативный C-движок ({output_bin})...", flush=True)
            model.eval()
            export_model_bin(model, output_bin)
            print(f"\033[92m[✓] Все {completed_steps} шагов успешно зафиксированы в model.bin для C-движка!\033[0m", flush=True)
            print(f"\033[92m[✓] Обучение завершено и безопасно сохранено.\033[0m", flush=True)
        finally:
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)
    else:
        print("Обучение остановлено до первого шага, сохранение пропущено.", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="code", choices=["code", "math", "reasoning", "logic", "physics", "chat", "all"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gdrive_dir", type=str, default="G:/My Drive/datasets", help="Путь к папке датасетов на Google Диске (по умолчанию 'G:/My Drive/datasets')")
    args = parser.parse_args()

    train_cuda(domain=args.domain, steps=args.steps, lr=args.lr, gdrive_dir=args.gdrive_dir)
