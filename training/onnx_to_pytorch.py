"""
ONNX to PyTorch Weight Converter for Supertonic

This script converts the pretrained English Supertonic ONNX weights
to PyTorch state_dict format for transfer learning to Ukrainian/Russian.
"""

import torch
import onnx
import numpy as np
from collections import OrderedDict
import os


def load_onnx_weights(onnx_path):
    """Load weights from ONNX model file"""
    model = onnx.load(onnx_path)
    
    # Extract initializers (weights and biases)
    weights = {}
    for initializer in model.graph.initializer:
        name = initializer.name
        data = onnx.numpy_helper.to_array(initializer)
        weights[name] = torch.from_numpy(data)
    
    return weights


def map_text_encoder_weights(onnx_weights, vocab_size_new=2000):
    """
    Map ONNX text encoder weights to PyTorch TextEncoder/ConvNeXtTextEncoder

    Supports ConvNeXt-based text encoder with embed_dim=256.
    Also expands character embedding rows to vocab_size_new.
    """
    pytorch_state = OrderedDict()

    # 1) Detect and map character embeddings
    # Common ONNX key pattern for Supertonic: '...text_embedder.char_embedder.weight'
    embed_key = None
    for k in onnx_weights.keys():
        if 'char_embedder.weight' in k or k.endswith('embedding.weight'):
            embed_key = k
            break
    if embed_key is not None:
        orig_embed = onnx_weights[embed_key]
        if orig_embed.ndim == 2:
            embed_dim = orig_embed.shape[1]
            new_embed = torch.randn(vocab_size_new, embed_dim) * 0.02
            rows_to_copy = min(orig_embed.shape[0], vocab_size_new)
            new_embed[:rows_to_copy] = orig_embed[:rows_to_copy]
            pytorch_state['embedding.weight'] = new_embed
            print(f"✓ Char embeddings mapped: {orig_embed.shape} → {new_embed.shape}")

    # 2) Detect ConvNeXt blocks in ONNX and map
    has_convnext = any('convnext' in k for k in onnx_weights.keys())
    if has_convnext:
        # Expected ONNX patterns like: '...convnext.convnext.<i>.dwconv.weight', '...norm.norm.weight',
        # '...pwconv1.weight', '...pwconv2.weight', etc.
        for k, v in onnx_weights.items():
            if 'convnext.convnext.' in k:
                try:
                    # Extract block index
                    parts = k.split('convnext.convnext.')[-1].split('.')
                    blk_idx = int(parts[0])
                    tail = '.'.join(parts[1:])

                    # Depthwise conv
                    if tail.startswith('dwconv.weight'):
                        pytorch_state[f'convnext.{blk_idx}.dwconv.weight'] = v
                    elif tail.startswith('dwconv.bias'):
                        pytorch_state[f'convnext.{blk_idx}.dwconv.bias'] = v
                    # Norm
                    elif 'norm.norm.weight' in k:
                        pytorch_state[f'convnext.{blk_idx}.norm.gamma'] = v
                    elif 'norm.norm.bias' in k:
                        pytorch_state[f'convnext.{blk_idx}.norm.beta'] = v
                    # Pointwise convs
                    elif tail.startswith('pwconv1.weight'):
                        pytorch_state[f'convnext.{blk_idx}.pwconv1.weight'] = v
                    elif tail.startswith('pwconv1.bias'):
                        pytorch_state[f'convnext.{blk_idx}.pwconv1.bias'] = v
                    elif tail.startswith('pwconv2.weight'):
                        pytorch_state[f'convnext.{blk_idx}.pwconv2.weight'] = v
                    elif tail.startswith('pwconv2.bias'):
                        pytorch_state[f'convnext.{blk_idx}.pwconv2.bias'] = v
                except Exception:
                    continue
        return pytorch_state

    # 3) Fallback: legacy transformer mapping (kept for compatibility)
    for key, value in onnx_weights.items():
        if 'text_encoder.layers' in key or 'text_encoder.transformer' in key:
            pt_key = key.replace('text_encoder.', '')
            pytorch_state[pt_key] = value
    for key, value in onnx_weights.items():
        if 'norm' in key.lower() or 'pos' in key.lower():
            if 'text_encoder' in key:
                pt_key = key.replace('text_encoder.', '')
                pytorch_state[pt_key] = value
    return pytorch_state


def map_duration_predictor_weights(onnx_weights):
    """Map ONNX duration predictor weights to PyTorch"""
    pytorch_state = OrderedDict()
    
    for key, value in onnx_weights.items():
        if 'duration_predictor' in key or 'dur_pred' in key:
            pt_key = key.replace('duration_predictor.', '').replace('dur_pred.', '')
            pytorch_state[pt_key] = value
    
    return pytorch_state


def map_vector_estimator_weights(onnx_weights):
    """Map ONNX flow matching / vector estimator weights to PyTorch"""
    pytorch_state = OrderedDict()
    
    for key, value in onnx_weights.items():
        if 'vector_estimator' in key or 'flow' in key or 'decoder' in key:
            pt_key = key.replace('vector_estimator.', '').replace('decoder.', '')
            pytorch_state[pt_key] = value
    
    return pytorch_state


def convert_onnx_to_pytorch(onnx_dir, output_path='pretrained_weights.pt'):
    """
    Main conversion function
    
    Args:
        onnx_dir: Directory containing ONNX model files
        output_path: Where to save converted PyTorch weights
    """
    # Supertonic v2 has multiple ONNX files
    # Based on HuggingFace repo structure
    text_encoder_path = os.path.join(onnx_dir, 'text_encoder.onnx')
    duration_pred_path = os.path.join(onnx_dir, 'duration_predictor.onnx')
    vector_est_path = os.path.join(onnx_dir, 'vector_estimator.onnx')
    
    # Alternative: single monolithic ONNX file
    monolithic_path = os.path.join(onnx_dir, 'supertonic.onnx')
    
    all_weights = {}
    
    # Try loading individual components first
    if os.path.exists(text_encoder_path):
        print(f"Loading {text_encoder_path}...")
        te_weights = load_onnx_weights(text_encoder_path)
        all_weights['text_encoder'] = map_text_encoder_weights(te_weights)
    
    if os.path.exists(duration_pred_path):
        print(f"Loading {duration_pred_path}...")
        dp_weights = load_onnx_weights(duration_pred_path)
        all_weights['duration_predictor'] = map_duration_predictor_weights(dp_weights)
    
    if os.path.exists(vector_est_path):
        print(f"Loading {vector_est_path}...")
        ve_weights = load_onnx_weights(vector_est_path)
        all_weights['vector_estimator'] = map_vector_estimator_weights(ve_weights)
    
    # If individual files don't exist, try monolithic
    if not all_weights and os.path.exists(monolithic_path):
        print(f"Loading monolithic {monolithic_path}...")
        onnx_weights = load_onnx_weights(monolithic_path)
        
        all_weights['text_encoder'] = map_text_encoder_weights(onnx_weights)
        all_weights['duration_predictor'] = map_duration_predictor_weights(onnx_weights)
        all_weights['vector_estimator'] = map_vector_estimator_weights(onnx_weights)
    
    if not all_weights:
        raise FileNotFoundError(f"No ONNX files found in {onnx_dir}")
    
    # Save converted weights
    torch.save(all_weights, output_path)
    print(f"\n✅ Saved converted weights to {output_path}")
    print(f"Components: {list(all_weights.keys())}")
    
    return all_weights


def load_pretrained_into_model(model, pretrained_path, freeze_encoder=False):
    """
    Load pretrained weights into Supertonic model
    
    Args:
        model: PyTorch Supertonic model instance
        pretrained_path: Path to converted weights .pt file
        freeze_encoder: Whether to freeze text encoder (fine-tune only new embeddings)
    """
    weights = torch.load(pretrained_path, map_location='cpu')
    
    # Load text encoder
    if 'text_encoder' in weights:
        missing, unexpected = model.text_encoder.load_state_dict(weights['text_encoder'], strict=False)
        print(f"Text Encoder loaded:")
        print(f"  Missing keys: {len(missing)}")
        if missing:
            print("   ", missing[:20], '...' if len(missing) > 20 else '')
        print(f"  Unexpected keys: {len(unexpected)}")
        if unexpected:
            print("   ", unexpected[:20], '...' if len(unexpected) > 20 else '')
        
        if freeze_encoder:
            for name, param in model.text_encoder.named_parameters():
                if 'embedding' not in name:
                    param.requires_grad = False
            print("  Froze pretrained layers, kept embeddings trainable")
    
    # Load duration predictor
    if 'duration_predictor' in weights:
        model.duration_predictor.load_state_dict(weights['duration_predictor'], strict=False)
        print("Duration Predictor loaded")
    
    # Load vector estimator (flow matching)
    if 'vector_estimator' in weights:
        model.vector_estimator.load_state_dict(weights['vector_estimator'], strict=False)
        print("Vector Estimator loaded")
    
    return model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert Supertonic ONNX to PyTorch')
    parser.add_argument('--onnx-dir', type=str, required=True,
                       help='Directory containing ONNX model files')
    parser.add_argument('--output', type=str, default='pretrained_weights.pt',
                       help='Output path for converted weights')
    
    args = parser.parse_args()
    
    convert_onnx_to_pytorch(args.onnx_dir, args.output)
