import struct
import os
import sys

# Force UTF-8 on Windows stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import tiktoken
from model import CodeLanguageModel

def export_tokenizer(filename="tokenizer.bin"):
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Exporting tokenizer GPT-2 ({vocab_size} tokens) to {filename}...")
    
    with open(filename, "wb") as f:
        f.write(struct.pack("i", vocab_size))
        for i in range(vocab_size):
            try:
                b = enc.decode_single_token_bytes(i)
            except KeyError:
                b = b""
            f.write(struct.pack("i", len(b)))
            f.write(b)
    print(f"Tokenizer saved ({os.path.getsize(filename) / 1024:.1f} KB).")

def export_model(weights_path="moe_model_weights.pth", output_bin="model.bin"):
    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    
    # Model configuration
    n_embd = 256
    n_head = 8
    n_layer = 6
    block_size = 128
    num_experts = 8
    top_k = 2
    dropout = 0.0

    print("Initializing MoE model structure...")
    model = CodeLanguageModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout,
        num_experts=num_experts,
        top_k=top_k
    )

    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}...")
        try:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            print("Weights loaded.")
        except Exception as e:
            print(f"Init fresh weights: {e}")
    else:
        print("No weights file found, initializing new random weights.")

    model.eval()

    print(f"Exporting model weights to {output_bin}...")
    with open(output_bin, "wb") as f:
        # Header: 7 int32s
        header = struct.pack(
            "iiiiiii",
            vocab_size,
            block_size,
            n_embd,
            n_layer,
            n_head,
            num_experts,
            top_k
        )
        f.write(header)

        # 1. token embedding table: [vocab_size, n_embd]
        f.write(model.token_embedding_table.weight.detach().numpy().astype("float32").tobytes())

        # 2. position embedding table: [block_size, n_embd]
        f.write(model.position_embedding_table.weight.detach().numpy().astype("float32").tobytes())

        # 3. Transformer blocks
        for block in model.blocks:
            # LayerNorm 1
            f.write(block.ln1.weight.detach().numpy().astype("float32").tobytes())
            f.write(block.ln1.bias.detach().numpy().astype("float32").tobytes())

            # Multi-Head Attention Q, K, V, O
            wq = torch.cat([h.query.weight for h in block.sa.heads], dim=0) # [n_embd, n_embd]
            wk = torch.cat([h.key.weight for h in block.sa.heads], dim=0)
            wv = torch.cat([h.value.weight for h in block.sa.heads], dim=0)
            wo = block.sa.proj.weight
            f.write(wq.detach().numpy().astype("float32").tobytes())
            f.write(wk.detach().numpy().astype("float32").tobytes())
            f.write(wv.detach().numpy().astype("float32").tobytes())
            f.write(wo.detach().numpy().astype("float32").tobytes())

            # LayerNorm 2
            f.write(block.ln2.weight.detach().numpy().astype("float32").tobytes())
            f.write(block.ln2.bias.detach().numpy().astype("float32").tobytes())

            # MoE Gate: [num_experts, n_embd]
            f.write(block.moe.gate.weight.detach().numpy().astype("float32").tobytes())
            if block.moe.gate.bias is not None:
                f.write(block.moe.gate.bias.detach().numpy().astype("float32").tobytes())
            else:
                f.write(torch.zeros(num_experts, dtype=torch.float32).numpy().tobytes())

            # MoE Experts
            for exp in block.moe.experts:
                w1 = exp.net[0].weight # [4*n_embd, n_embd]
                b1 = exp.net[0].bias   # [4*n_embd]
                w2 = exp.net[2].weight # [n_embd, 4*n_embd]
                b2 = exp.net[2].bias   # [n_embd]
                f.write(w1.detach().numpy().astype("float32").tobytes())
                f.write(b1.detach().numpy().astype("float32").tobytes())
                f.write(w2.detach().numpy().astype("float32").tobytes())
                f.write(b2.detach().numpy().astype("float32").tobytes())

        # 4. Final LayerNorm
        f.write(model.ln_f.weight.detach().numpy().astype("float32").tobytes())
        f.write(model.ln_f.bias.detach().numpy().astype("float32").tobytes())

        # 5. LM Head: [vocab_size, n_embd]
        f.write(model.lm_head.weight.detach().numpy().astype("float32").tobytes())

    total_mb = os.path.getsize(output_bin) / (1024 * 1024)
    print(f"Export completed successfully! File {output_bin} ({total_mb:.2f} MB)")

if __name__ == "__main__":
    export_tokenizer("tokenizer.bin")
    export_model("moe_model_weights.pth", "model.bin")
