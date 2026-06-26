#!/bin/bash
# Install dependencies
apt-get update && apt-get install -y cmake g++ libeigen3-dev git-lfs

# Initialize Git LFS and pull if .git exists, otherwise warn
if [ -d ".git" ]; then
    echo "[BUILD] Git directory found. Pulling LFS files..."
    git lfs install
    git lfs pull
else
    echo "[WARNING] .git directory not found. Skipping git lfs pull."
    echo "[WARNING] Ensure model_checkpoint.bin is present."
fi

# Build the shared library
mkdir -p build
cd build
cmake ..
make alien_llm_lib
cd ..
