# Supertonic TTS Training Diagnosis Report

## Executive Summary

The trained model is producing low-quality output due to a combination of **dataset issues**, **insufficient training**, and **suboptimal model architecture**. The diagnostic script has identified 4 key issues ranked by severity.

---

## Diagnosis Results

### 1. Training Progress
- **Epochs completed**: 47 out of 100 planned
- **Model parameters**: 11.2M total
  - Text encoder: 4.99M
  - Duration predictor: 493K
  - Vector estimator: 5.72M
- **Learning rate**: 2e-4
- **Batch size**: 32

### 2. Dataset Quality Issues

| Metric | Value | Status |
|--------|-------|--------|
| Total samples | 2,105 | OK |
| Unique texts | 118 | **CRITICAL** |
| Duplicates | 1,987 (94%) | **CRITICAL** |
| Windows paths | 55 | **WARNING** |
| Unix paths | 2,050 | OK |

**Key Finding**: The dataset has only 118 unique sentences repeated ~18 times each. This extreme repetition prevents the model from learning diverse phonetic patterns and generalizing to new text.

### 3. Root Causes Identified

#### [HIGH] Mixed Windows/Unix Paths in Dataset
- **Impact**: Some samples may fail to load, causing training on partial data
- **Evidence**: 55 Windows-style paths vs 2050 Unix-style paths
- **Fix**: Run `python tools/migrate_dataset_to_wsl.py`

#### [MEDIUM] Small ConvNeXt Encoder (6-8 blocks)
- **Impact**: Insufficient model capacity for complex phoneme-to-speech mapping
- **Current**: 6-8 ConvNeXt blocks
- **Recommended**: 12-24 blocks (modern TTS standard)
- **Fix**: Set `CONVNEXT_BLOCKS=12` environment variable

#### [MEDIUM] Training Stopped Early (Epoch 47)
- **Impact**: Model hasn't converged yet
- **Current**: 47 epochs
- **Recommended**: 100-500 epochs for TTS convergence
- **Fix**: Continue training or restart with more epochs

#### [LOW] Low Flow Matching Steps (10-20)
- **Impact**: Lower quality during inference
- **Current**: 10-20 steps
- **Recommended**: 50-100 steps
- **Fix**: Increase `n_steps` in `run_inference.py`

---

## Why the Model Produces "Sounds Like" Speech

### Primary Causes:

1. **Insufficient Unique Training Data (94% duplicates)**
   - The model has only seen 118 unique sentences
   - It cannot learn proper phoneme representations from such limited variety
   - Result: Mumbled, "sounds like" speech that doesn't form real words

2. **Small Model Capacity**
   - 11.2M parameters is relatively small for TTS
   - The ConvNeXt encoder with 6-8 blocks lacks the depth to learn complex mappings
   - Result: Poor attention alignment, garbled output

3. **Early Stopping**
   - 47 epochs is insufficient for convergence
   - TTS models typically need 100+ epochs to learn duration and alignment
   - Result: Duration predictor hasn't learned proper timing (fast tempo)

4. **Mixed Path Formats**
   - 55 samples with Windows paths may fail to load on Linux/WSL
   - Training on partial data further reduces effective dataset size
   - Result: Inconsistent learning

---

## Recommended Fixes

### Immediate Actions (Do These First)

#### 1. Fix Dataset Paths
```bash
python tools/migrate_dataset_to_wsl.py
```

#### 2. Generate More Diverse Training Data
The current dataset has only 118 unique sentences. Generate at least 500-1000 unique sentences:

```bash
# Edit generate_dataset.py to create more diverse text
# Then regenerate:
python generate_dataset.py
```

**Target**: 5,000+ samples with 1,000+ unique texts (5x repetition max)

### Training Improvements

#### 3. Increase Model Capacity
Edit `training/train_high_vram.py`:
```python
model = Supertonic(
    vocab_size=2000,
    embed_dim=384,     # Increase from 256
    hidden_dim=384,    # Increase from 256
    style_dim=128
)
```

Or set environment variable:
```bash
export CONVNEXT_BLOCKS=12
```

#### 4. Use Phased Training
The `phases_loss_gate.json` already defines a good training schedule:
- Phase A: Stabilize with frozen lower blocks (LR=2e-5)
- Phase B: Progressive unfreezing (LR=3e-5)
- Phase C: Full fine-tuning (LR=3e-5)

#### 5. Train Longer
- Minimum: 100 epochs
- Recommended: 200-300 epochs for fine-tuning
- Monitor loss - should drop below 5.0 for good quality

### Inference Improvements

#### 6. Increase Flow Matching Steps
Edit `run_inference.py`:
```python
mel = model.inference(text_tensor, style, lang_ids=lang_ids, n_steps=50, fixed_duration=duration_val)
```

#### 7. Use Learned Duration Predictor
Instead of `fixed_duration=15`, use the learned duration predictor:
```python
mel = model.inference(text_tensor, style, lang_ids=lang_ids, n_steps=50, fixed_duration=None)
```

#### 8. Replace GriffinLim with HiFiGAN
GriffinLim produces lower quality audio. Train or use a pretrained HiFiGAN vocoder for better results.

---

## Expected Timeline for Quality Improvements

| Action | Time | Expected Improvement |
|--------|------|---------------------|
| Fix paths + Continue training to 100 epochs | 2-4 hours | Moderate improvement |
| Add diverse data (500+ unique texts) | 1-2 days | Significant improvement |
| Increase model capacity + retrain | 4-8 hours | Better clarity |
| Use HiFiGAN vocoder | 2-4 hours (training) | Much better audio quality |
| Full pipeline (all fixes) | 2-3 days | Production quality |

---

## Monitoring Training Progress

Watch for these metrics during training:

1. **Loss value**: Should decrease from ~15-20 to <5.0
2. **Duration prediction**: Check if predicted durations match actual mel lengths
3. **Attention alignment**: Visualize attention maps - should be diagonal
4. **Generated samples**: Test every 10 epochs to monitor improvement

---

## Conclusion

The model's poor quality is primarily due to **insufficient unique training data** (only 118 unique sentences) and **early stopping** (47 epochs). The pretrained weights are being loaded correctly, so the foundation is there. 

**Priority fixes**:
1. Generate more diverse training data (most important)
2. Continue training to at least 100 epochs
3. Fix mixed path formats
4. Increase model capacity if VRAM allows

With these changes, you should see significant improvement in output quality within 2-3 days of training.
