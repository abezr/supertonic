# Next Steps - Install ONNX and Convert Weights

**Status:** ✅ ONNX models downloaded successfully!

**Remaining steps:**

## 1. Stop Current Training
Press `Ctrl+C` in your training terminal (it's running from random initialization)

## 2. Install ONNX Package
```bash
./venv/bin/pip install onnx
```

## 3. Convert Weights
```bash
./venv/bin/python training/onnx_to_pytorch.py --onnx-dir onnx --output pretrained_weights.pt
```

**Expected output:**
```
✓ Expanded char embeddings: torch.Size([161, 192]) → torch.Size([2000, 192])
✓ Expanded lang embeddings: torch.Size([5, 192]) → torch.Size([7, 192])
Text Encoder loaded
Duration Predictor loaded
Vector Estimator loaded
✅ Saved converted weights to pretrained_weights.pt
```

## 4. Verify
```bash
ls -lh pretrained_weights.pt
```

Should show a file around 50-100MB

## 5. Restart Training
```bash
./run_training.sh
```

**Expected output:**
```
============================================================
🔄 Loading pretrained ONNX weights from pretrained_weights.pt
============================================================
Text Encoder loaded:
  Missing keys: 1839 (expected: new Cyrillic embeddings)
Duration Predictor loaded
Vector Estimator loaded
✅ Pretrained weights loaded - starting fine-tuning

Epoch 0 | Batch 0 | Loss ~3.5 (much lower than 15.24!)
```

The loss should start much lower (~3-5) instead of 15+ because the model already knows basic acoustic modeling!
