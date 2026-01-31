#!/usr/bin/env python3
"""
Dataset migration script for WSL performance optimization.

Copies dataset from mounted Windows paths (/mnt/d/...) to WSL native filesystem
for significantly faster I/O during training.

Usage:
    # Migrate from Windows D: drive to WSL native home directory
    python tools/migrate_dataset_to_wsl.py --source /mnt/d/study/ai/supertonic/dataset_marina

    # Or specify custom destination
    python tools/migrate_dataset_to_wsl.py --source /mnt/d/study/ai/supertonic/dataset_marina \
                                           --dest ~/datasets/dataset_marina

After migration:
    1. Update filelist.txt paths (this script does it automatically)
    2. Use the new dataset path in your training commands

Benefits:
    - 5-10x faster file I/O compared to /mnt/d/ paths
    - Better compatibility with PyTorch DataLoader
    - Reduced training overhead from path translation
"""

import argparse
import shutil
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Migrate dataset from Windows mount to WSL native path"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="/mnt/d/study/ai/supertonic/dataset_marina",
        help="Source dataset directory (typically on /mnt/d/ or /mnt/c/)",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=None,
        help="Destination directory (default: ~/datasets/dataset_marina)",
    )
    parser.add_argument(
        "--filelist",
        type=str,
        default=None,
        help="Path to filelist.txt (default: SOURCE/filelist.txt)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without copying files",
    )
    return parser.parse_args()


def get_wsl_native_path():
    """Get the WSL native home directory path."""
    return Path.home() / "datasets"


def copy_dataset(source_dir: Path, dest_dir: Path, dry_run: bool = False):
    """Copy dataset files from source to destination."""
    if dry_run:
        print(f"[DRY RUN] Would copy {source_dir} -> {dest_dir}")
        return

    print(f"Creating destination directory: {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Get all .wav files
    wav_files = sorted(source_dir.glob("*.wav"))
    print(f"Found {len(wav_files)} .wav files to copy")

    # Copy files with progress
    for i, wav_file in enumerate(wav_files, 1):
        dest_file = dest_dir / wav_file.name
        shutil.copy2(wav_file, dest_file)
        if i % 10 == 0 or i == len(wav_files):
            print(f"  Copied {i}/{len(wav_files)} files...")

    print(f"✅ Copied {len(wav_files)} files to {dest_dir}")


def update_filelist(
    source_filelist: Path, dest_dir: Path, dest_filelist: Path, dry_run: bool = False
):
    """Update filelist.txt with new paths."""
    if not source_filelist.exists():
        print(f"⚠️  Source filelist not found: {source_filelist}")
        print("   Skipping filelist update.")
        return

    if dry_run:
        print(f"[DRY RUN] Would update filelist: {source_filelist} -> {dest_filelist}")
        return

    print(f"\nUpdating filelist: {source_filelist} -> {dest_filelist}")

    # Read original filelist
    with open(source_filelist, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Transform paths
    new_lines = []
    for line in lines:
        line = line.strip()
        if not line or "|" not in line:
            new_lines.append(line)
            continue

        parts = line.split("|", 1)
        old_path = parts[0]
        text = parts[1] if len(parts) > 1 else ""

        # Extract just the filename
        filename = Path(old_path).name
        new_path = str(dest_dir / filename)

        new_lines.append(f"{new_path}|{text}")

    # Write new filelist
    with open(dest_filelist, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

    print(f"✅ Updated filelist with {len(new_lines)} entries")


def print_next_steps(dest_dir: Path, dest_filelist: Path):
    """Print instructions for using the migrated dataset."""
    print("\n" + "=" * 60)
    print("📋 NEXT STEPS")
    print("=" * 60)
    print(f"\nDataset migrated to: {dest_dir}")
    print(f"Updated filelist: {dest_filelist}")
    print("\nTo use the migrated dataset in training:")
    print("  1. Update your training script to use:")
    print(f"     dataset_path = '{dest_dir}'")
    print(f"     filelist_path = '{dest_filelist}'")
    print("\n  2. Or set environment variable:")
    print(f"     export DATASET_PATH={dest_dir}")
    print("\n  3. Run training with optimized settings:")
    print("     COMPILE_MODEL=1 BATCH_SIZE=4 GRAD_ACCUM_STEPS=6 \\")
    print("       python tools/finetune_convnext.py")
    print("\n" + "=" * 60)


def main():
    args = parse_args()

    # Resolve paths
    source_dir = Path(args.source).expanduser().resolve()

    if args.dest:
        dest_dir = Path(args.dest).expanduser().resolve()
    else:
        dest_dir = get_wsl_native_path() / source_dir.name

    if args.filelist:
        source_filelist = Path(args.filelist).expanduser().resolve()
    else:
        source_filelist = source_dir / "filelist.txt"

    dest_filelist = dest_dir / "filelist.txt"

    # Validate source
    if not source_dir.exists():
        print(f"❌ Error: Source directory does not exist: {source_dir}")
        sys.exit(1)

    print("=" * 60)
    print("🚀 WSL Dataset Migration Tool")
    print("=" * 60)
    print(f"\nSource:      {source_dir}")
    print(f"Destination: {dest_dir}")
    print(f"Filelist:    {source_filelist}")

    if args.dry_run:
        print("\n[DRY RUN MODE - No files will be modified]")

    # Check if destination already exists
    if dest_dir.exists() and not args.dry_run:
        response = input(f"\n⚠️  Destination exists: {dest_dir}\nOverwrite? (y/N): ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)
        shutil.rmtree(dest_dir)

    # Copy dataset
    print("\n" + "-" * 60)
    copy_dataset(source_dir, dest_dir, dry_run=args.dry_run)

    # Update filelist
    update_filelist(source_filelist, dest_dir, dest_filelist, dry_run=args.dry_run)

    # Print next steps
    if not args.dry_run:
        print_next_steps(dest_dir, dest_filelist)
    else:
        print("\n[DRY RUN COMPLETE]")
        print("Run without --dry-run to perform the actual migration.")


if __name__ == "__main__":
    main()
