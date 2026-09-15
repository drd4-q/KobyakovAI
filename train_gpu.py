import os
import sys
import struct

# Force UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.optim as optim
import tiktoken
from model import CodeLanguageModel
from data_loader import StreamingDataLoader

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
    n_layer = 6
    num_experts = 8
    top_k = 2

    print(f"Initializing MoE model ({n_layer} layers, {n_embd} dim, {num_experts} experts, top-{top_k})...")
    model = CodeLanguageModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=0.1,
        num_experts=num_experts,
        top_k=top_k
    )
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total parameters: {num_params:.2f} M")

    # If model.bin exists, we can start from it or train fresh
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

    print(f"\nTraining for {steps} steps on {device.upper()}...")
    model.train()

    for step in range(1, steps + 1):
        x, y = loader.get_batch()

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 50 == 0 or step == steps:
            print(f"Step {step:4d} / {steps} | Loss: \033[92m{loss.item():.4f}\033[0m")

    print("\nTraining completed!")
    # Save PyTorch checkpoint
    torch.save(model.state_dict(), checkpoint_pth)
    print(f"Saved PyTorch weights to {checkpoint_pth}")

    # Export directly to native C model.bin
    print(f"Exporting directly to native C format ({output_bin})...")
    model.eval()
    with open(output_bin, "wb") as f:
        header = struct.pack("iiiiiii", vocab_size, block_size, n_embd, n_layer, n_head, num_experts, top_k)
        f.write(header)
        f.write(model.token_embedding_table.weight.detach().cpu().numpy().astype("float32").tobytes())
        f.write(model.position_embedding_table.weight.detach().cpu().numpy().astype("float32").tobytes())

        for block in model.blocks:
            f.write(block.ln1.weight.detach().cpu().numpy().astype("float32").tobytes())
            f.write(block.ln1.bias.detach().cpu().numpy().astype("float32").tobytes())

            wq = torch.cat([h.query.weight for h in block.sa.heads], dim=0)
            wk = torch.cat([h.key.weight for h in block.sa.heads], dim=0)
            wv = torch.cat([h.value.weight for h in block.sa.heads], dim=0)
            wo = block.sa.proj.weight
            f.write(wq.detach().cpu().numpy().astype("float32").tobytes())
            f.write(wk.detach().cpu().numpy().astype("float32").tobytes())
            f.write(wv.detach().cpu().numpy().astype("float32").tobytes())
            f.write(wo.detach().cpu().numpy().astype("float32").tobytes())

            f.write(block.ln2.weight.detach().cpu().numpy().astype("float32").tobytes())
            f.write(block.ln2.bias.detach().cpu().numpy().astype("float32").tobytes())

            f.write(block.moe.gate.weight.detach().cpu().numpy().astype("float32").tobytes())
            if block.moe.gate.bias is not None:
                f.write(block.moe.gate.bias.detach().cpu().numpy().astype("float32").tobytes())
            else:
                f.write(torch.zeros(num_experts, dtype=torch.float32).numpy().tobytes())

            for exp in block.moe.experts:
                w1 = exp.net[0].weight
                b1 = exp.net[0].bias
                w2 = exp.net[2].weight
                b2 = exp.net[2].bias
                f.write(w1.detach().cpu().numpy().astype("float32").tobytes())
                f.write(b1.detach().cpu().numpy().astype("float32").tobytes())
                f.write(w2.detach().cpu().numpy().astype("float32").tobytes())
                f.write(b2.detach().cpu().numpy().astype("float32").tobytes())

        f.write(model.ln_f.weight.detach().cpu().numpy().astype("float32").tobytes())
        f.write(model.ln_f.bias.detach().cpu().numpy().astype("float32").tobytes())
        f.write(model.lm_head.weight.detach().cpu().numpy().astype("float32").tobytes())

    print(f"Done! {output_bin} updated for engine_c.exe!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="all", choices=["code", "math", "reasoning", "logic", "physics", "chat", "all"])
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gdrive_dir", type=str, default="G:/My Drive/datasets", help="Путь к папке датасетов на Google Диске (по умолчанию 'G:/My Drive/datasets')")
    args = parser.parse_args()

    train_cuda(domain=args.domain, steps=args.steps, lr=args.lr, gdrive_dir=args.gdrive_dir)
