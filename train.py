import torch
import os
import argparse
from model import CodeLanguageModel
from data_loader import StreamingDataLoader, vocab_size

def main():
    parser = argparse.ArgumentParser(description="Тренировка нейросети с архитектурой Sparse MoE")
    parser.add_argument("--domain", type=str, default="all", choices=["all", "code", "math", "physics", "chat"],
                        help="Выбор домена для обучения (по умолчанию: all — все сферы)")
    parser.add_argument("--num_experts", type=int, default=4, help="Количество экспертов в MoE слоях (по умолчанию: 4)")
    parser.add_argument("--top_k", type=int, default=2, help="Количество выбираемых экспертов на токен (по умолчанию: 2)")
    args = parser.parse_args()

    domain = args.domain
    checkpoint_path = "moe_model_weights.pth"

    print(f"============================================================")
    print(f" ЗАПУСК ОБУЧЕНИЯ MoE МОДЕЛИ (Домен: {domain.upper()})")
    print(f" Экспертов: {args.num_experts} | Top-K выбор: {args.top_k}")
    print(f"============================================================")

    batch_size = 16
    block_size = 128
    learning_rate = 3e-4
    max_iters = 50000
    eval_interval = 100

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используемое устройство: {device}")

    print("Инициализация Streaming Data Loader...")
    loader = StreamingDataLoader(domain=domain, block_size=block_size, batch_size=batch_size, device=device)

    # Инициализация MoE Transformer модели
    model = CodeLanguageModel(
        vocab_size=vocab_size,
        n_layer=8,
        n_embd=512,
        n_head=8,
        block_size=block_size,
        num_experts=args.num_experts,
        top_k=args.top_k
    )
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Количество параметров MoE модели: {num_params:.2f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if os.path.exists(checkpoint_path):
        print(f"Найден чекпоинт {checkpoint_path}. Загружаем веса...")
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
            print("Чекпоинт успешно загружен! Продолжаем обучение MoE.")
        except Exception as e:
            print(f"Предупреждение при загрузке (возможно изменилась архитектура): {e}")
            print("Начинаем обучение заново.")
    else:
        print(f"Чекпоинт {checkpoint_path} не найден. Начинаем обучение MoE с нуля.")

    use_amp = (device == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    print("\nНачало обучения MoE... (Нажмите Ctrl+C для безопасной остановки)")

    try:
        for iter_num in range(max_iters):
            x, y = loader.get_batch()

            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if iter_num % eval_interval == 0:
                print(f"Итерация {iter_num}: Ошибка (Loss) {loss.item():.4f}")
                torch.save(model.state_dict(), checkpoint_path)

    except KeyboardInterrupt:
        print("\nОбучение прервано пользователем!")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Финальная MoE модель сохранена в {checkpoint_path}")

if __name__ == "__main__":
    main()
