#!/bin/bash
# Setup script to download and convert Supertonic ONNX weights

set -e

echo "=============================================="
echo "Supertonic ONNX Weight Setup"
echo "=============================================="

# 1. Create onnx directory
echo "Creating onnx directory..."
mkdir -p onnx
cd onnx

# 2. Download ONNX models from HuggingFace
echo ""
echo "Downloading ONNX models from HuggingFace..."
echo "(This may take a few minutes)"

BASE_URL="https://huggingface.co/Supertone/supertonic-2/resolve/main/onnx"

# Download text encoder
if [ ! -f "text_encoder.onnx" ]; then
    echo "  - Downloading text_encoder.onnx..."
    curl -L "${BASE_URL}/text_encoder.onnx" -o text_encoder.onnx
fi

# Download duration predictor
if [ ! -f "duration_predictor.onnx" ]; then
    echo "  - Downloading duration_predictor.onnx..."
    curl -L "${BASE_URL}/duration_predictor.onnx" -o duration_predictor.onnx
fi

# Download vector estimator (flow matching)
if [ ! -f "vector_estimator.onnx" ]; then
    echo "  - Downloading vector_estimator.onnx..."
    curl -L "${BASE_URL}/vector_estimator.onnx" -o vector_estimator.onnx
fi

# Download vocoder
if [ ! -f "vocoder.onnx" ]; then
    echo "  - Downloading vocoder.onnx..."
    curl -L "${BASE_URL}/vocoder.onnx" -o vocoder.onnx
fi

# Download unicode indexer
if [ ! -f "unicode_indexer.json" ]; then
    echo "  - Downloading unicode_indexer.json..."
    curl -L "${BASE_URL}/unicode_indexer.json" -o unicode_indexer.json
fi

cd ..

# 3. Convert ONNX to PyTorch
echo ""
echo "Converting ONNX weights to PyTorch..."
./venv/bin/python training/onnx_to_pytorch.py --onnx-dir onnx --output pretrained_weights.pt

echo ""
echo "=============================================="
echo "✅ Setup complete!"
echo "=============================================="
echo ""
echo "Pretrained weights saved to: pretrained_weights.pt"
echo ""
echo "Next steps:"
echo "  1. Delete existing checkpoint (to start fresh):"
echo "     rm training_checkpoint.pt"
echo ""
echo "  2. Run training:"
echo "     ./run_training.sh"
echo ""
echo "The training will now load English weights and fine-tune for Ukrainian/Russian!"
echo "=============================================="
