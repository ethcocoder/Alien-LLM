#!/bin/bash
# Install dependencies
apt-get update && apt-get install -y cmake g++ libeigen3-dev

# Build the shared library
mkdir -p build
cd build
cmake ..
make alien_llm_lib
cd ..
