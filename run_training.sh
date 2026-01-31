#!/bin/bash

# Define project root
PROJECT_ROOT="/mnt/d/study/ai/supertonic"
cd "$PROJECT_ROOT"

# Activating Virtual Environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found! Please run setup_venv.sh first."
    exit 1
fi

# Ensure CUDA paths are visible if installed manually
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Run
echo "Running Supertonic Training..."
echo "Selected Script: $1"

if [ -z "$1" ]; then
    # Default to low vram
    python -m training.train_low_vram
else
    python -m "$1"
fi
