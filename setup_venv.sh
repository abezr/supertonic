#!/bin/bash
# Setup script for WSL environment
# This creates a virtual environment and installs required packages

echo "Setting up Python virtual environment for WSL..."

# Use python3 explicitly (Linux Python, not Windows)
# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install python3-full:"
    echo "  sudo apt update && sudo apt install python3-full python3-venv"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment (detect platform)
echo "Activating virtual environment..."
if [ -f "venv/Scripts/activate" ]; then
    # Windows (Git Bash, MSYS, etc.)
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    # Linux/Mac/WSL
    source venv/bin/activate
else
    echo "Warning: Could not find activation script. Continuing anyway..."
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install gradio_client
echo "Installing gradio_client..."
pip install gradio_client

echo ""
echo "Setup complete! To use the virtual environment:"
if [ -f "venv/Scripts/activate" ]; then
    echo "  source venv/Scripts/activate  # Windows"
    echo "  python generate_dataset.py"
    echo ""
    echo "Or run directly:"
    echo "  ./venv/Scripts/python generate_dataset.py"
else
    echo "  source venv/bin/activate  # Linux/Mac/WSL"
    echo "  python generate_dataset.py"
    echo ""
    echo "Or run directly:"
    echo "  ./venv/bin/python generate_dataset.py"
fi
