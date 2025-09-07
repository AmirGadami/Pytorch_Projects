# 🧠 Attention Mechanisms in PyTorch

This repository contains minimal PyTorch implementations of the core attention mechanisms used in Transformer architectures.  
The goal is **clarity and learning** — the code is small enough to step through by hand, but faithful to the original *Attention Is All You Need* paper.

---

## ✨ Implemented Modules

### 1. SelfAttention
```python
class SelfAttention(nn.Module)
```
- Standard self-attention over a sequence of token embeddings.  
- Input: encodings for a single sequence.  
- Output: contextualized embeddings for each token.  
- Formula:  
  Attention(Q,K,V) = softmax(QKᵀ / √dₖ) V  

---

### 2. MaskedSelfAttention
```python
class MaskedSelfAttention(nn.Module)
```
- Extension of self-attention with **masking** support.  
- Handles:  
  - Padding masks → ignore `[PAD]` tokens.  
  - Causal masks → prevent attending to future tokens.  

---

### 3. Attention
```python
class Attention(nn.Module)
```
- A general attention layer where **queries, keys, and values may come from different sources**.  
- Covers both:  
  - Self-attention (Q=K=V)  
  - Cross-attention (Q from decoder, K/V from encoder)  

---

### 4. MultiHeadAttention (Full Implementation)
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=2, num_heads=2):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, d_model = q.size()
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        sims = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            sims = sims.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(sims, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.w_o(out)
```

---

## 📂 Project Structure
```
attention/
├── attention.py     # Implementations of attention classes
└── README.md        # Documentation
```

---

## 🚀 Quick Start
```python
import torch
from attention import SelfAttention, MaskedSelfAttention, Attention, MultiHeadAttention

x = torch.randn(1, 3, 4)  # batch=1, seq_len=3, embed_dim=4

# SelfAttention
sa = SelfAttention(d_model=4)
print("SelfAttention:", sa(x[0]).shape)

# MaskedSelfAttention
mask = torch.tensor([[0, -float("inf"), -float("inf")], 
                     [0, 0, -float("inf")], 
                     [0, 0, 0]])
msa = MaskedSelfAttention(d_model=4)
print("MaskedSelfAttention:", msa(x[0], mask=mask).shape)

# MultiHeadAttention
mha = MultiHeadAttention(d_model=4, num_heads=2)
print("MultiHeadAttention:", mha(x, x, x).shape)
```

---

## 🧠 Concepts Recap
- **Query (Q):** What am I looking for?  
- **Key (K):** What do I offer?  
- **Value (V):** What information do I carry?  
- **Softmax(QKᵀ):** Attention weights (who to focus on).  
- **Weighted sum with V:** Contextualized representation.  
- **Multi-head:** Several attention mechanisms in parallel capture different relations.  

---

## 📖 References
- Vaswani et al. (2017). [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)  
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)  
