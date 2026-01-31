import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .modules import LayerNorm, AdaLayerNorm, SinusoidalPositionalEmbedding, PositionalEncoding, RoPE

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, n_heads, style_dim, n_langs=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lang_embedding = nn.Embedding(n_langs, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)
        self.rope = RoPE(embed_dim // n_heads)
        self.style_proj = nn.Linear(style_dim, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_dim*4, 
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, text_ids, style, lang_ids=None, mask=None):
        # text_ids: [B, T]
        # style: [B, S]
        # lang_ids: [B]
        x = self.embedding(text_ids)
        if lang_ids is not None:
            l = self.lang_embedding(lang_ids).unsqueeze(1)
            x = x + l
        
        x = self.pos_enc(x)
        s = self.style_proj(style).unsqueeze(1)
        x = x + s 
        
        x = self.rope(x)
        
        if mask is not None:
            # mask is [B, 1, T] usually 1 for valid, 0 for pad
            # Transformer expects src_key_padding_mask as [B, T] where True is PADDED (ignored)
            # if mask is 1 for keep, 0 for ignore:
            padding_mask = (mask.squeeze(1) == 0)
        else:
            padding_mask = None

        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return x

class ConvNeXtBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dwconv = nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.norm = LayerNorm(channels)
        self.pwconv1 = nn.Conv1d(channels, channels * 4, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv1d(channels * 4, channels, kernel_size=1)

    def forward(self, x):
        # x: [B, T, C]
        residual = x
        y = x.transpose(1, 2)
        y = self.dwconv(y)
        y = self.norm(y)
        y = self.pwconv1(y)
        y = self.act(y)
        y = self.pwconv2(y)
        y = y.transpose(1, 2)
        return residual + y

class ConvNeXtTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, n_blocks=8, style_dim=128, n_langs=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lang_embedding = nn.Embedding(n_langs, embed_dim)
        self.style_proj = nn.Linear(style_dim, embed_dim)
        self.convnext = nn.ModuleList([ConvNeXtBlock1D(embed_dim) for _ in range(n_blocks)])

    def forward(self, text_ids, style, lang_ids=None, mask=None):
        x = self.embedding(text_ids)
        if lang_ids is not None:
            x = x + self.lang_embedding(lang_ids).unsqueeze(1)
        x = x + self.style_proj(style).unsqueeze(1)
        for blk in self.convnext:
            x = blk(x)
        return x

class DurationPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, style_dim, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim + style_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.norm1 = LayerNorm(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2)
        self.norm2 = LayerNorm(hidden_dim)
        self.proj = nn.Conv1d(hidden_dim, 1, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_emb, style, mask=None):
        # text_emb: [B, T, C]
        # style: [B, S]
        x = text_emb.transpose(1, 2) # [B, C, T]
        s = style.unsqueeze(-1).expand(-1, -1, x.size(2)) # [B, S, T]
        x = torch.cat([x, s], dim=1)
        
        x = self.conv1(x)
        x = F.relu(self.norm1(x))
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = F.relu(self.norm2(x))
        x = self.dropout(x)
        
        x = self.proj(x) # [B, 1, T]
        
        if mask is not None:
            x = x * mask
            
        return x.squeeze(1) # [B, T] - log duration

class VectorEstimator(nn.Module):
    def __init__(self, in_dim, dim, style_dim, n_layers=4, n_heads=4, dropout=0.1):
        super().__init__()
        self.time_emb = SinusoidalPositionalEmbedding(dim)
        self.pos_enc = PositionalEncoding(dim)
        self.rope = RoPE(dim // n_heads)
        
        # Local context: 1D Conv before global attention
        self.input_conv = nn.Conv1d(in_dim, dim, kernel_size=5, padding=2)
        self.input_proj = nn.Linear(dim, dim) 
        
        self.style_proj = nn.Linear(style_dim, dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            batch_first=True,
            norm_first=True,
            dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.final_proj = nn.Linear(dim, in_dim) 

    def forward(self, noisy_latent, text_emb, style, time_step, mask=None, text_mask=None):
        # noisy_latent: [B, T_latent, 80]
        # Align/Project
        x = noisy_latent.transpose(1, 2) # [B, 80, T]
        x = F.gelu(self.input_conv(x))
        x = x.transpose(1, 2) # [B, T, 192]
        
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.rope(x)
        
        t_emb = self.time_emb(time_step) # [B, D]
        s = self.style_proj(style).unsqueeze(1)
        
        x = x + s + t_emb.unsqueeze(1) 
        
        # Cross-attention over text_emb
        output = self.decoder(x, text_emb) 
        return self.final_proj(output)

class Supertonic(nn.Module):
    def __init__(self, vocab_size=2000, embed_dim=192, hidden_dim=192, style_dim=128, n_mels=80, encoder_type='transformer', n_langs=10):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == 'convnext':
            embed_dim = 256
            hidden_dim = 256
            blocks = int(os.environ.get("CONVNEXT_BLOCKS", 8))
            self.text_encoder = ConvNeXtTextEncoder(vocab_size, embed_dim=embed_dim, n_blocks=blocks, style_dim=style_dim, n_langs=n_langs)
        else:
            self.text_encoder = TextEncoder(vocab_size, embed_dim, hidden_dim, 4, 4, style_dim, n_langs=n_langs)
        self.duration_predictor = DurationPredictor(hidden_dim, hidden_dim, style_dim)
        self.vector_estimator = VectorEstimator(n_mels, hidden_dim, style_dim)
        # Vocoder is separate (HiFiGAN usually) and often frozen or loaded separately
    
    @torch.no_grad()
    def inference(self, text_ids, style, lang_ids=None, n_steps=10, temperature=1.0, fixed_duration=None):
        device = text_ids.device
        
        # 1. Encode Text
        # text_ids: [1, T_text]
        text_emb = self.text_encoder(text_ids, style, lang_ids=lang_ids)
        
        # 2. Predict Duration
        if fixed_duration is not None:
            # Manual override (e.g. 5-10 frames per token)
            dur_int = torch.full((1, text_emb.size(1)), fixed_duration, dtype=torch.long, device=device)
        else:
            log_dur = self.duration_predictor(text_emb, style)
            dur = torch.exp(log_dur) # [1, T_text]
            dur_int = torch.ceil(dur).long()
            dur_int = torch.clamp(dur_int, min=1)
        
        # Upsample
        # Repeat each token embedding `dur_int` times
        upsampled_emb = []
        for i in range(text_emb.size(1)):
            reps = dur_int[0, i].item()
            upsampled_emb.append(text_emb[:, i:i+1].expand(-1, reps, -1))
        
        cond = torch.cat(upsampled_emb, dim=1) # [1, T_mel, D]
        length = cond.size(1)
        
        # 3. Flow Matching Sampling (Euler)
        # Start from Noise x0
        x = torch.randn(1, length, 80, device=device) * temperature
        
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = i / n_steps
            t = torch.tensor([t_val], device=device).float()
            
            # Predict velocity
            # VectorEstimator expects [B, T, D] inputs? 
            # Our definition: vector_estimator(noisy_latent, text_emb, style, t)
            # noisy_latent should be [1, T, 80]
            # text_emb aligned should be [1, T, D] -> We calculated `cond`
            
            # Note: VectorEstimator projects x(80) -> 192 inside.
            v = self.vector_estimator(x, cond, style, t)
            
            # Update: x_{t+1} = x_t + v * dt
            x = x + v * dt
            
        return x.transpose(1, 2) # Return [1, 80, T] for vocoder
