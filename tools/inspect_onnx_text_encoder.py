#!/usr/bin/env python3
"""
Inspect ONNX Text Encoder structure to determine ConvNeXt block count and embedding dimension.

Usage:
  python tools/inspect_onnx_text_encoder.py --path onnx/text_encoder.onnx

Outputs:
  - List of ConvNeXt block indices detected (0-based)
  - Total count of ConvNeXt blocks
  - Detected embedding dimension from char embedding matrix, if present
"""
import argparse
import onnx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='onnx/text_encoder.onnx', help='Path to ONNX text encoder file')
    args = parser.parse_args()

    print(f"Loading ONNX: {args.path}")
    model = onnx.load(args.path)

    # Detect ConvNeXt blocks
    blks = set()
    for w in model.graph.initializer:
        name = w.name
        if 'convnext.convnext.' in name:
            try:
                idx = int(name.split('convnext.convnext.')[1].split('.')[0])
                blks.add(idx)
            except Exception:
                pass

    # Try to infer embedding dimension from char embedding
    onnx_embed_dim = None
    embed_key = None
    for w in model.graph.initializer:
        if 'char_embedder.weight' in w.name or w.name.endswith('embedding.weight'):
            if len(w.dims) == 2:
                onnx_embed_dim = int(w.dims[1])
                embed_key = w.name
                break

    print("\n=== ONNX Text Encoder Inspection ===")
    print(f"ConvNeXt blocks found: {sorted(list(blks))} | count = {len(blks)}")
    if embed_key is not None:
        print(f"Embedding key: {embed_key}")
    print(f"Embedding dim (if detected): {onnx_embed_dim if onnx_embed_dim is not None else 'unknown'}")

    # Extra: show a few relevant initializers for manual verification
    print("\nSample initializers:")
    shown = 0
    for w in model.graph.initializer:
        if any(k in w.name for k in ['char_embedder.weight', 'convnext.convnext.0.dwconv.weight', 'convnext.convnext.0.pwconv1.weight']):
            print(f"  {w.name:80s} {list(w.dims)}")
            shown += 1
        if shown >= 5:
            break


if __name__ == '__main__':
    main()
