# WSL Setup Guide

## Problem
You're encountering two issues:
1. **Externally-managed-environment error**: Modern Linux distributions protect system Python
2. **Python path issue**: Windows pyenv-win shims are interfering with WSL's Python

## Solution

### Step 1: Fix Python PATH (One-time setup)

The error `/mnt/c/Python311/Lib/site-packages/pyenv-win/shims/python: cannot execute` means your PATH includes Windows Python paths. 

**Option A: Fix in your current shell session**
```bash
# Remove Windows Python paths from PATH temporarily
export PATH=$(echo $PATH | tr ':' '\n' | grep -v '/mnt/c' | tr '\n' ':' | sed 's/:$//')
```

**Option B: Fix permanently in your ~/.bashrc or ~/.zshrc**
```bash
# Add this to ~/.bashrc (or ~/.zshrc if using zsh)
# Remove Windows Python paths from PATH
export PATH=$(echo $PATH | tr ':' '\n' | grep -v '/mnt/c' | grep -v 'pyenv-win' | tr '\n' ':' | sed 's/:$//')

# Or more simply, ensure Linux Python comes first
export PATH="/usr/bin:/usr/local/bin:$PATH"
```

Then reload your shell:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

### Step 2: Set up Virtual Environment

Run the setup script:
```bash
bash setup_venv.sh
```

This will:
- Create a virtual environment using Linux Python (`python3`)
- Install `gradio_client` and dependencies
- Set everything up properly

### Step 3: Run the Script

**Option A: Use the run script**
```bash
bash run_generate.sh
```

**Option B: Manual activation**
```bash
source venv/bin/activate
python generate_dataset.py
deactivate  # when done
```

**Note:** For WSL, use `./venv/bin/python` instead of `python` to ensure the correct Python interpreter is used.

**Option C: Direct execution (without activation)**
```bash
./venv/bin/python generate_dataset.py
```

## Verify Python is Correct

Before running setup, verify you're using Linux Python:
```bash
which python3
# Should show: /usr/bin/python3 (or similar Linux path)
# NOT: /mnt/c/... (Windows path)

python3 --version
# Should show Python 3.x.x
```

## Troubleshooting

### If python3 is not found:
```bash
sudo apt update
sudo apt install python3-full python3-venv
```

### If you still see Windows Python paths:
Check your shell configuration files:
```bash
grep -r "pyenv-win\|/mnt/c.*Python" ~/.bashrc ~/.bash_profile ~/.profile ~/.zshrc 2>/dev/null
```

Remove or comment out any lines that add Windows Python to PATH.
