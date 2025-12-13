#!/bin/bash
# Build script for qg Python package
# This script sets up the compiler environment and builds the wheel

export CC=g++
export CXX=g++

# Set include directory for building from sdist
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export RABITQ_INCLUDE_DIR="$PROJECT_ROOT/include"

echo "Setting RABITQ_INCLUDE_DIR=$RABITQ_INCLUDE_DIR"

# Build the package
python -m build

