import torch
import tiktoken
from model import CodeLanguageModel

enc = tiktoken.get_encoding('gpt2')
model = CodeLanguageModel(
    vocab_size=enc.n_vocab,
    n_embd=256,
    n_head=8,
    n_layer=6,
    block_size=128,
    dropout=0.0,
    num_experts=8,
    top_k=2
)
model.load_state_dict(torch.load('moe_model_weights.pth', map_location='cpu', weights_only=True), strict=False)
model.eval()

prompts = [
    "User: hello world python\n\nKobyakovAI:\n",
    "User: who are you\n\nKobyakovAI:\n",
    "User: What is the command to print \"Hello World\" twelve times in Python?\n\nKobyakovAI:\n"
]

for p in prompts:
    print("=" * 50)
    print("PROMPT:", p.strip())
    tokens = enc.encode(p)
    x = torch.tensor([tokens], dtype=torch.long)
    generated = []
    with torch.no_grad():
        for _ in range(60):
            logits, _ = model(x[:, -128:])
            # greedy next token
            next_tok = torch.argmax(logits[0, -1, :]).item()
            if next_tok == 50256:
                break
            generated.append(next_tok)
            x = torch.cat([x, torch.tensor([[next_tok]])], dim=1)
    print("RESPONSE:")
    print(enc.decode(generated))
