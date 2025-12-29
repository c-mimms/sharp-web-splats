#!/bin/bash
set -e

echo "Starting SHARP Splats Setup..."

# 1. Initialize Submodules
echo "Initializing submodules..."
git submodule update --init --recursive

# 2. Setup Python Environment
echo "Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 3. Setup Node/PlayCanvas
echo "Installing Node dependencies..."
npm install

echo "Setup complete!"
echo "To start the server, run:"
echo "source .venv/bin/activate && python app.py"
