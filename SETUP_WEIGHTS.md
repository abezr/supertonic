# Setup Instructions - Pretrained ONNX Weights

Due to WSL command execution issues, please run these steps manually in your WSL terminal:

## Step 1: Download ONNX Models

```bash
cd /mnt/d/study/ai/supertonic

# Option A: Use Python script (recommended)
./venv/bin/python download_onnx.py

# Option B: Manual download with curl
mkdir -p onnx
cd onnx
curl -L "https://huggingface.co/Supertone/supertonic-2/resolve/main/onnx/text_encoder.onnx" -o text_encoder.onnx
curl -L "https://huggingface.co/Supertone/supertonic-2/resolve/main/onnx/duration_predictor.onnx" -o duration_predictor.onnx
curl -L "https://huggingface.co/Supertone/supertonic-2/resolve/main/onnx/vector_estimator.onnx" -o vector_estimator.onnx
curl -L "https://huggingface.co/Supertone/supertonic-2/resolve/main/onnx/vocoder.onnx" -o vocoder.onnx
curl -L "https://huggingface.co/Supertone/supertonic-2/resolve/main/onnx/unicode_indexer.json" -o unicode_indexer.json
cd ..
```

## Step 2: Convert ONNX to PyTorch

```bash
./venv/bin/python training/onnx_to_pytorch.py --onnx-dir onnx --output pretrained_weights.pt
```

## Step 3: Delete Old Checkpoint

```bash
rm training_checkpoint.pt
```

## Step 4: Start Training

```bash
./run_training.sh
```

## Expected Output

When training starts, you should see:

```
============================================================
🔄 Loading pretrained ONNX weights from pretrained_weights.pt
============================================================
Text Encoder loaded:
  Missing keys: 1839 (expected: new Cyrillic embeddings)
  Unexpected keys: 0
Duration Predictor loaded
Vector Estimator loaded
✅ Pretrained weights loaded - starting fine-tuning

Epoch 1/100...
```

## What This Fixes

- ✅ **Before:** Random weights → gibberish at epoch 21
- ✅ **After:** Pretrained English → intelligible speech by epoch 5-10
- ⏳ **Remaining:** Griffin-Lim vocoder (address after confirming weight loading works)
