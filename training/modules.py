import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        transpose = False
        if x.ndim == 3: # [B, C, T] -> [B, T, C]
            transpose = True
            x = x.transpose(1, 2)
        
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        
        if transpose:
            x = x.transpose(1, 2)
        return x

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, D]
        return x + self.pe[:, :x.size(1), :]

class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):
        # x: [B, C, T] or [B, C]
        # s: [B, S_dim]
        
        transpose = False
        if x.ndim == 3:
            transpose = True
            x = x.transpose(1, 2) # [B, T, C]

        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        
        h = self.fc(s)
        gamma, beta = h.chunk(2, dim=1)
        
        if transpose:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        else:
            gamma = gamma.unsqueeze(1) # [B, 1, C] if x was [B, C] originally but normalized
            # Adjust dims if needed based on usage

        x = (1 + gamma) * x + beta
        
        if transpose:
            x = x.transpose(1, 2)
        return x

class RoPE(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        # x: [B, T, D]
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1) # [T, D_rope]
        cos = emb.cos()
        sin = emb.sin()
        
        # Match dimensions for broadcasting
        # cos/sin: [T, D_rope] -> [1, T, D_rope]
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        
        # If x has more channels than RoPE dim (e.g. D=192 vs D_rope=48)
        # only rotate the first D_rope channels
        d_rope = cos.shape[-1]
        x_rope = x[..., :d_rope]
        x_pass = x[..., d_rope:]
        
        x_rotated = (x_rope * cos) + (rotate_half(x_rope) * sin)
        return torch.cat((x_rotated, x_pass), dim=-1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

