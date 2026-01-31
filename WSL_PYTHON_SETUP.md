# WSL Python Environment Setup Guide

This document provides a comprehensive guide for setting up the Python environment in WSL (Windows Subsystem for Linux) for the Supertonic project, including solutions to common issues encountered during setup.

## Table of Contents

- [Problem Overview](#problem-overview)
- [Root Causes](#root-causes)
- [Solution Overview](#solution-overview)
- [Step-by-Step Setup](#step-by-step-setup)
- [Quick Reference Commands](#quick-reference-commands)
- [Troubleshooting](#troubleshooting)
- [Common Issues](#common-issues)

---

## Problem Overview

When attempting to run the training script in WSL using `python -m training.train_low_vram`, users may encounter multiple issues:

1. **Import Errors**: Missing dependencies like `torch`, `torchaudio`, and other training-specific packages
2. **Virtual Environment Corruption**: Mixed Windows/Linux file system structures causing import failures
3. **Script Compatibility**: The existing [`setup_venv.sh`](setup_venv.sh:1) script is designed for Windows/Git Bash, not WSL
4. **Incomplete Dependencies**: PyTorch and related packages are not listed in [`py/requirements.txt`](py/requirements.txt:1) but are required for training

### Example Error Messages

```
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'torchaudio'
ImportError: cannot import name 'load' from 'torch' (unknown location)
```

---

## Root Causes

### 1. Virtual Environment Corruption

When a virtual environment is created in Windows (using Git Bash or PowerShell) and then accessed from WSL, the environment becomes corrupted due to:

- **Mixed Path Separators**: Windows uses `\` while Linux uses `/`
- **Different Executable Formats**: Windows `.exe` vs Linux binaries
- **Symlink Issues**: WSL may not properly handle Windows-style symlinks
- **File System Differences**: NTFS vs ext4 filesystem characteristics

### 2. Missing PyTorch Dependencies

The training scripts in the [`training/`](training/) directory require PyTorch and related packages that are not included in the base [`py/requirements.txt`](py/requirements.txt:1):

- `torch` - Core PyTorch library
- `torchaudio` - Audio processing for PyTorch
- `torchvision` - Computer vision utilities (optional, for some models)

### 3. Script Incompatibility

The [`setup_venv.sh`](setup_venv.sh:1) script uses Windows-specific commands and paths that don't work in WSL:

```bash
# Windows/Git Bash specific commands that fail in WSL
python -m venv venv
source venv/Scripts/activate  # Wrong path for WSL
```

---

## Solution Overview

The solution involves:

1. **Remove the corrupted virtual environment** completely
2. **Create a new clean virtual environment** using WSL-compatible commands
3. **Install PyTorch** with the appropriate version (CPU or GPU)
4. **Install project dependencies** from [`py/requirements.txt`](py/requirements.txt:1)
5. **Verify the setup** by running the training script

---

## Step-by-Step Setup

### Prerequisites

Ensure you have the following installed in WSL:

- Python 3.8 or higher
- pip (Python package installer)
- Virtualenv (optional, but recommended)

Check your Python version:
```bash
python3 --version
```

### Step 1: Remove Corrupted Virtual Environment

If you have an existing virtual environment that was created in Windows or is corrupted, remove it:

```bash
# Navigate to project root
cd /mnt/d/study/ai/supertonic

# Remove the corrupted venv directory
rm -rf venv
```

**Important**: Do not try to repair a corrupted virtual environment. Always start fresh.

### Step 2: Create a New Virtual Environment

Create a new virtual environment using WSL-compatible commands:

```bash
# Create virtual environment
python3 -m venv venv

# Activate the virtual environment (WSL/Linux path)
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt, indicating the virtual environment is active.

### Step 3: Upgrade pip

Always upgrade pip to the latest version before installing packages:

```bash
pip install --upgrade pip
```

### Step 4: Install PyTorch

Choose the appropriate PyTorch installation based on your hardware:

#### Option A: CPU-only Installation (Recommended for Testing)

If you don't have a GPU or want to test without GPU acceleration:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Option B: GPU Installation (CUDA)

If you have an NVIDIA GPU and CUDA installed:

```bash
# For CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Note**: Check your CUDA version with `nvcc --version` before installing.

#### Option C: GPU Installation (ROCm)

If you have an AMD GPU:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

### Step 5: Install Project Dependencies

Install the remaining dependencies from [`py/requirements.txt`](py/requirements.txt:1):

```bash
pip install -r py/requirements.txt
```

### Step 6: Verify Installation

Test that PyTorch is installed correctly:

```bash
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch version: 2.x.x+cpu (or 2.x.x+cu121 for GPU)
CUDA available: False (or True for GPU)
```

### Step 7: Run Training Script

Now you can run the training script:

```bash
python -m training.train_low_vram
```

Or for high VRAM systems:

```bash
python -m training.train_high_vram
```

---

## Quick Reference Commands

### Initial Setup (One-time)

```bash
# Navigate to project
cd /mnt/d/study/ai/supertonic

# Remove old venv (if exists)
rm -rf venv

# Create new venv
python3 -m venv venv

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
pip install -r py/requirements.txt
```

### Starting a New Session

```bash
# Navigate to project
cd /mnt/d/study/ai/supertonic

# Activate virtual environment
source venv/bin/activate

# Run training
python -m training.train_low_vram
```

### Deactivating Virtual Environment

```bash
deactivate
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Cause**: PyTorch is not installed or the virtual environment is not activated.

**Solution**:
1. Ensure the virtual environment is activated: `source venv/bin/activate`
2. Check if torch is installed: `pip list | grep torch`
3. If not installed, follow Step 4 in the setup guide

### Issue: "ImportError: cannot import name 'load' from 'torch'"

**Cause**: Corrupted virtual environment with mixed Windows/Linux structures.

**Solution**:
1. Deactivate the virtual environment: `deactivate`
2. Remove the corrupted venv: `rm -rf venv`
3. Follow the complete setup guide from Step 1

### Issue: "Command 'python' not found"

**Cause**: Python is not installed or not in PATH.

**Solution**:
1. Check if python3 is available: `python3 --version`
2. If not, install Python 3 in WSL:
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-venv
   ```

### Issue: "pip install fails with permission error"

**Cause**: Trying to install packages system-wide instead of in virtual environment.

**Solution**:
1. Ensure virtual environment is activated: `source venv/bin/activate`
2. Check you're in the virtual environment: `which pip` should show path to venv

### Issue: "CUDA out of memory" during training

**Cause**: GPU memory is insufficient for the model or batch size.

**Solution**:
1. Use the low VRAM training script: `python -m training.train_low_vram`
2. Reduce batch size in the training configuration
3. Use CPU-only PyTorch if GPU memory is severely limited

### Issue: "Slow training performance"

**Cause**: Using CPU instead of GPU, or GPU not properly utilized.

**Solution**:
1. Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
2. If False, install GPU version of PyTorch (see Step 4, Option B)
3. Check GPU utilization with `nvidia-smi` during training

---

## Common Issues

### 1. Virtual Environment Location

**Problem**: Creating virtual environment in Windows filesystem (`/mnt/c/` or `/mnt/d/`) can be slow due to filesystem overhead.

**Solution**: For better performance, create the virtual environment in the WSL home directory:

```bash
# Create venv in home directory
python3 -m venv ~/supertonic-venv

# Activate it
source ~/supertonic-venv/bin/activate

# Install dependencies
pip install -r /mnt/d/study/ai/supertonic/py/requirements.txt
```

### 2. Path Issues Between Windows and WSL

**Problem**: Scripts or paths that work in Windows may not work in WSL.

**Solution**: Always use WSL paths when working in WSL:
- Windows: `D:\study\ai\supertonic`
- WSL: `/mnt/d/study/ai/supertonic`

### 3. Package Version Conflicts

**Problem**: Different packages requiring different versions of the same dependency.

**Solution**:
1. Use a fresh virtual environment
2. Install packages in the order specified in this guide
3. If conflicts persist, use `pip install package==version` to specify exact versions

### 4. WSL Version Compatibility

**Problem**: Some packages may not work properly in WSL 1 vs WSL 2.

**Solution**:
- WSL 2 is recommended for better compatibility and performance
- Check your WSL version: `wsl --list --verbose`
- Upgrade to WSL 2 if needed: `wsl --set-version <distro-name> 2`

### 5. Network Issues During Installation

**Problem**: Slow or failed package downloads due to network issues.

**Solution**:
1. Use a different PyTorch mirror if the default is slow
2. Increase pip timeout: `pip install --timeout=1000 <package>`
3. Use a VPN if you're in a region with restricted access

---

## Additional Resources

### PyTorch Installation Guide
- Official documentation: https://pytorch.org/get-started/locally/
- Version compatibility matrix: https://pytorch.org/get-started/previous-versions/

### WSL Documentation
- WSL installation: https://learn.microsoft.com/en-us/windows/wsl/install
- WSL 2 vs WSL 1: https://learn.microsoft.com/en-us/windows/wsl/compare-versions

### Project-Specific Documentation
- Main README: [`README.md`](README.md)
- Training scripts: [`training/`](training/)
- Python dependencies: [`py/requirements.txt`](py/requirements.txt)

---

## Summary

The key to a successful WSL Python environment setup for this project is:

1. **Always create virtual environments in WSL**, not in Windows
2. **Remove and recreate** corrupted environments rather than trying to fix them
3. **Install PyTorch separately** from the main requirements file
4. **Choose the right PyTorch variant** (CPU vs GPU) for your hardware
5. **Verify installations** before running training scripts

Following this guide will ensure a clean, working Python environment in WSL for training the Supertonic model.
