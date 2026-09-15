import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """ Одиночный механизм самовнимания (Self-Attention) """
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # Вычисляем веса внимания (attention scores)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Маскируем будущее
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """ Несколько голов самовнимания, работающих параллельно """
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ Полносвязная сеть эксперта """
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class SparseMoE(nn.Module):
    """ Слой Sparse Mixture of Experts (MoE) с динамическим Top-K роутингом """
    def __init__(self, n_embd, num_experts=4, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(n_embd, num_experts)
        self.experts = nn.ModuleList([FeedForward(n_embd, dropout) for _ in range(num_experts)])

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C) # (B*T, C)
        
        # Логиты шлюза (роутера)
        gate_logits = self.gate(x_flat) # (B*T, num_experts)
        weights = F.softmax(gate_logits, dim=-1)
        
        # Топ-K экспертов для каждого токена
        topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=-1) # (B*T, top_k)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8) # Нормализация
        
        out = torch.zeros_like(x_flat)
        
        # Направление токенов экспертам
        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i)
            token_idx, k_idx = torch.where(mask)
            
            if len(token_idx) > 0:
                expert_input = x_flat[token_idx]
                expert_output = expert(expert_input)
                routing_weight = topk_weights[token_idx, k_idx].unsqueeze(-1)
                out.index_add_(0, token_idx, expert_output * routing_weight)
                
        return out.view(B, T, C)

class Block(nn.Module):
    """ Блок трансформера: Multi-Head Attention + Sparse MoE """
    def __init__(self, n_embd, n_head, block_size, dropout, num_experts=4, top_k=2):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.moe = SparseMoE(n_embd, num_experts=num_experts, top_k=top_k, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x

class CodeLanguageModel(nn.Module):
    """ Модель Transformer с архитектурой Sparse Mixture-of-Experts """
    def __init__(self, vocab_size, n_embd=512, n_head=8, n_layer=8, block_size=128, dropout=0.1, num_experts=4, top_k=2):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head=n_head, block_size=block_size, dropout=dropout, num_experts=num_experts, top_k=top_k)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B*T, C)
            targets_reshaped = targets.view(B*T)
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
