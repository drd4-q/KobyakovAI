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

def export_model_bin(model, output_bin="model.bin"):
    """
    Экспортирует модель с иерархической структурой MoE (144 ноды)
    в нативный плоский бинарный формат для C-движка (engine.c).
    """
    block = model.blocks[0]
    moe = block.moe

    vocab_size = model.token_embedding_table.weight.shape[0]
    n_embd = model.token_embedding_table.weight.shape[1]
    block_size = model.block_size
    n_layer = len(model.blocks)
    n_head = block.sa.heads[0].key.weight.shape[0] * len(block.sa.heads) // n_embd
    # or n_head = len(block.sa.heads)
    n_head = len(block.sa.heads)
    num_parents = moe.num_parents
    num_sub_parents = moe.num_sub_parents
    num_leaf_experts = moe.num_leaf_experts
    top_k = moe.parents[0].top_k
    mult = 2 # hidden_dim = mult * n_embd

    print(f"Exporting Hierarchical MoE ({n_layer} layers, {num_parents} parents x {num_sub_parents} sub x {num_leaf_experts} leaves = {num_parents*num_sub_parents*num_leaf_experts} nodes) to {output_bin}...")

    with open(output_bin, "wb") as f:
        # Header: 10 int32s
        header = struct.pack(
            "iiiiiiiiii",
            vocab_size,
            block_size,
            n_embd,
            n_layer,
            n_head,
            num_parents,
            num_sub_parents,
            num_leaf_experts,
            top_k,
            mult
        )
        f.write(header)

        # 1. token embedding table: [vocab_size, n_embd]
        f.write(model.token_embedding_table.weight.detach().cpu().numpy().astype("float32").tobytes())

        # 2. position embedding table: [block_size, n_embd]
        f.write(model.position_embedding_table.weight.detach().cpu().numpy().astype("float32").tobytes())

        # 3. Transformer blocks
        for b_idx, blk in enumerate(model.blocks):
            # LayerNorm 1
            f.write(blk.ln1.weight.detach().cpu().numpy().astype("float32").tobytes())
            f.write(blk.ln1.bias.detach().cpu().numpy().astype("float32").tobytes())

            # Multi-Head Attention Q, K, V, O
            wq = torch.cat([h.query.weight for h in blk.sa.heads], dim=0) # [n_embd, n_embd]
            wk = torch.cat([h.key.weight for h in blk.sa.heads], dim=0)   # [n_embd, n_embd]
            wv = torch.cat([h.value.weight for h in blk.sa.heads], dim=0) # [n_embd, n_embd]
            wo = blk.sa.proj.weight                                       # [n_embd, n_embd]
            f.write(wq.detach().cpu().numpy().astype("float32").tobytes())
            f.write(wk.detach().cpu().numpy().astype("float32").tobytes())
            f.write(wv.detach().cpu().numpy().astype("float32").tobytes())
            f.write(wo.detach().cpu().numpy().astype("float32").tobytes())

            # LayerNorm 2
            f.write(blk.ln2.weight.detach().cpu().numpy().astype("float32").tobytes())
            f.write(blk.ln2.bias.detach().cpu().numpy().astype("float32").tobytes())

            # Hierarchical MoE:
            # a) Master Accuracy Gate [num_parents, n_embd]
            f.write(blk.moe.accuracy_gate.weight.detach().cpu().numpy().astype("float32").tobytes())
            if blk.moe.accuracy_gate.bias is not None:
                f.write(blk.moe.accuracy_gate.bias.detach().cpu().numpy().astype("float32").tobytes())
            else:
                f.write(torch.zeros(num_parents, dtype=torch.float32).numpy().tobytes())

            # b) 4 Parents
            for parent in blk.moe.parents:
                # Parent Gate [num_sub_parents, n_embd]
                f.write(parent.gate.weight.detach().cpu().numpy().astype("float32").tobytes())
                if parent.gate.bias is not None:
                    f.write(parent.gate.bias.detach().cpu().numpy().astype("float32").tobytes())
                else:
                    f.write(torch.zeros(num_sub_parents, dtype=torch.float32).numpy().tobytes())

                # 4 Sub-parents
                for sub in parent.sub_parents:
                    # Sub-parent Gate [num_leaf_experts, n_embd]
                    f.write(sub.gate.weight.detach().cpu().numpy().astype("float32").tobytes())
                    if sub.gate.bias is not None:
                        f.write(sub.gate.bias.detach().cpu().numpy().astype("float32").tobytes())
                    else:
                        f.write(torch.zeros(num_leaf_experts, dtype=torch.float32).numpy().tobytes())

                    # 9 Leaf Experts
                    for exp in sub.experts:
                        w1 = exp.net[0].weight # [mult*n_embd, n_embd]
                        b1 = exp.net[0].bias   # [mult*n_embd]
                        w2 = exp.net[2].weight # [n_embd, mult*n_embd]
                        b2 = exp.net[2].bias   # [n_embd]
                        f.write(w1.detach().cpu().numpy().astype("float32").tobytes())
                        f.write(b1.detach().cpu().numpy().astype("float32").tobytes())
                        f.write(w2.detach().cpu().numpy().astype("float32").tobytes())
                        f.write(b2.detach().cpu().numpy().astype("float32").tobytes())

        # 4. Final LayerNorm
        f.write(model.ln_f.weight.detach().cpu().numpy().astype("float32").tobytes())
        f.write(model.ln_f.bias.detach().cpu().numpy().astype("float32").tobytes())

        # 5. LM Head: [vocab_size, n_embd]
        f.write(model.lm_head.weight.detach().cpu().numpy().astype("float32").tobytes())

    total_mb = os.path.getsize(output_bin) / (1024 * 1024)
    print(f"Done! {output_bin} ({total_mb:.2f} MB) is ready for C engine.")

def export_model(weights_path="moe_model_weights.pth", output_bin="model.bin"):
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

    print("Initializing Hierarchical MoE model (144 nodes)...")
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
    export_model_bin(model, output_bin)

if __name__ == "__main__":
    export_tokenizer("tokenizer.bin")
    export_model("moe_model_weights.pth", "model.bin")
