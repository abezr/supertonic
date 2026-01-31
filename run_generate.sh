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

# Run
echo "Running Data Generation (Distillation)..."
echo "Target: Local Docker (localhost:7860)"
# Added --force to ensure old non-diverse data is replaced by new unique sentences
python generate_dataset.py --local --force "$@"
