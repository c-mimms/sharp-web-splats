# SHARP Splats

Generate 3D Gaussian Splats from photos using Apple's SHARP model and view them in VR/AR with PlayCanvas.

## Prerequisites
- macOS (tested on Apple Silicon)
- Python 3.13+
- Node.js & npm

## Installation

1. **Clone the repository with submodules**:
   ```bash
   git clone --recursive [your-repo-url]
   cd splats
   ```

2. **Run the setup script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
   This script will:
   - Initialize the SHARP Git submodule.
   - Create a Python virtual environment and install dependencies.
   - Install NPM packages and restore PlayCanvas core files to `static/`.

## Running the Server

1. **Activate the environment and start Flask**:
   ```bash
   source .venv/bin/activate
   python app.py
   ```
2. **Open the web app**:
   Navigate to `https://localhost:8080`.
   *Note: Use HTTPS as WebXR requires a secure context.*

## Features
- **Photo to Splat**: Upload any photo to generate a 3D Gaussian Splat in seconds.
- **VR/AR Ready**: Full support for WebXR headsets (Meta Quest, Vision Pro) and mobile AR.
- **Compressed Samples**: Includes pre-generated samples in the SuperSplat (`.sog`) format for fast loading.
- **Interactive Controls**: Orbit, pan, and zoom controls for smooth inspection.

## Acknowledgements
- [Apple SHARP](https://github.com/apple/ml-sharp) for the Gaussian Splat inference.
- [PlayCanvas](https://playcanvas.com) for the 3D rendering engine and Web Components.
- [Splat Transform](https://github.com/playcanvas/splat-transform) for PLY compression.
