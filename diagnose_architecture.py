"""
Diagnostic script to compare ONNX and PyTorch architectures
"""

import onnx
import torch
from training.model import Supertonic

print("="*70)
print("ARCHITECTURE COMPARISON")
print("="*70)

# Load ONNX
text_encoder = onnx.load('onnx/text_encoder.onnx')

print("\n1. ONNX Text Encoder Structure:")
print("-" * 70)
weight_names = [w.name for w in text_encoder.graph.initializer]
print(f"Total weights: {len(weight_names)}")
print("\nKey layers:")
for name in weight_names[:10]:
    dims = [d for w in text_encoder.graph.initializer if w.name == name for d in w.dims]
    print(f"  {name:55s} {dims}")

# Try to infer ONNX embedding dim
onnx_embed_dim = None
for w in text_encoder.graph.initializer:
    if 'char_embedder.weight' in w.name or 'embedding.weight' in w.name:
        if len(w.dims) == 2:
            onnx_embed_dim = int(w.dims[1])
            break

print("\n2. PyTorch Model Structure:")
print("-" * 70)

# Try different dimensions
for embed_dim in [192, 256]:
    try:
        model = Supertonic(vocab_size=2000, embed_dim=embed_dim, hidden_dim=embed_dim, style_dim=128)
        state = model.state_dict()
        print(f"\nWith embed_dim={embed_dim}:")
        print(f"  Total parameters: {len(state)}")
        print(f"  Text encoder layers:")
        for k in list(state.keys())[:10]:
            print(f"    {k:55s} {list(state[k].shape)}")
        break
    except Exception as e:
        print(f"  embed_dim={embed_dim} failed: {e}")

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)

onnx_has_convnext = any('convnext' in name for name in weight_names)
pytorch_uses_transformer = any('encoder.layers' in k for k in state.keys())
embed_dim = next((p.shape[-1] for k,p in model.state_dict().items() if k.endswith('embedding.weight')), None)

if onnx_has_convnext and pytorch_uses_transformer:
    print("❌ CRITICAL MISMATCH:")
    print("  - ONNX model uses ConvNeXt architecture")
    print("  - PyTorch model uses Transformer architecture")
    print("  - These are INCOMPATIBLE!")
    print("\n  You have two options:")
    print("    A) Rewrite PyTorch model to match ONNX ConvNeXt structure")
    print("    B) Use ONNX inference directly (no PyTorch training)")
else:
    print("✓ Architectures might be compatible")

print("\nONNX embedding dimension:", onnx_embed_dim if onnx_embed_dim is not None else "unknown")
print("PyTorch embedding dimension:", embed_dim)

if embed_dim != 256:
    print("❌ Dimension mismatch! Change embed_dim to 256 in training scripts if you plan to load ONNX weights.")
