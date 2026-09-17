import os
import sys
import time

# Force UTF-8 on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.nn.functional as F
import tiktoken
from model import CodeLanguageModel

def run_gpu_shell(weights_path="moe_model_weights.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("  🤖 KobyakovAI - GPU INTERACTIVE SHELL (CUDA Accelerated)")
    print(f"  Architecture: Hierarchical MoE (144 Nodes) | Device: {device.upper()}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print("  Type your prompt and press Enter.")
    print("  Type 'exit' or 'quit' to exit.")
    print("=" * 60 + "\n")

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    n_embd = 256
    n_head = 8
    n_layer = 2
    block_size = 128
    num_parents = 4
    num_sub_parents = 4
    num_leaf_experts = 9
    top_k = 2
    mult = 2

    print("Loading Hierarchical MoE weights onto GPU...")
    model = CodeLanguageModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=0.0,
        num_parents=num_parents,
        num_sub_parents=num_sub_parents,
        num_leaf_experts=num_leaf_experts,
        top_k=top_k,
        mult=mult
    )

    if os.path.exists(weights_path):
        try:
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print(f"Weights successfully loaded from {weights_path}!\n")
        except Exception as e:
            print(f"Warning: could not load {weights_path}: {e}\n")
    else:
        print(f"Warning: {weights_path} not found. Running with random weights.\n")

    model.to(device)
    model.eval()

    def generate_for_prompt(user_input):
        prompt = f"User: {user_input}\n\nKobyakovAI:\n"
        input_tokens = enc.encode(prompt)
        x = torch.tensor([input_tokens], dtype=torch.long, device=device)

        print("\n🤖 \033[96mKobyakovAI:\033[0m")
        sys.stdout.flush()

        start_time = time.time()
        generated_tokens = 0
        max_new_tokens = 150

        with torch.no_grad():
            for _ in range(max_new_tokens):
                cond_x = x[:, -block_size:]
                with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
                    logits, _ = model(cond_x)

                next_token_logits = logits[0, -1, :]
                
                # Greedy decoding for sharp, accurate code
                next_token = torch.argmax(next_token_logits).item()

                # Stop conditions: <|endoftext|> (50256) or next turn 'User' (12982)
                if next_token == 50256 or next_token == 12982:
                    break

                piece = enc.decode([next_token])
                if "User:" in piece:
                    break

                print(piece, end="", flush=True)
                generated_tokens += 1
                x = torch.cat([x, torch.tensor([[next_token]], device=device)], dim=1)

        elapsed = time.time() - start_time
        tps = generated_tokens / elapsed if elapsed > 0.001 else 0.0
        print(f"\n\033[90m[Generated {generated_tokens} tokens in {elapsed:.2f}s ({tps:.1f} tok/s) on {device.upper()}]\033[0m")
        print("-" * 60)

    # Single-shot prompt from CLI
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        generate_for_prompt(" ".join(sys.argv[1:]))
        return

    while True:
        try:
            user_input = input("\n\033[1mUser:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        generate_for_prompt(user_input)

if __name__ == "__main__":
    run_gpu_shell()
