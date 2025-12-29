# SHARP Splats

Generate 3D Gaussian Splats from photos using Apple's SHARP model and view them in VR/AR with PlayCanvas.

## Prerequisites
- macOS (tested on Apple Silicon)
- Python 3.13+
- Node.js & npm

## Installation

1. **Clone the repository**:
   ```bash
   git clone [your-repo-url]
   cd sharp-web-splats
   ```

2. **Run the setup script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
   This script will:
   - Initialize and update the SHARP Git submodule.
   - Create a Python virtual environment and install dependencies.
   - Install NPM packages.

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
- **Photo to Splat**: Upload any photo to generate a 3D Gaussian Splat using SHARP. Splats are compressed using SuperSplat (`.sog`) format for better performance on VR/AR devices.
- **VR/AR Ready**: View the splat in VR/AR using WebXR. Tested on Quest 3S.
- **Compressed Samples**: Includes pre-generated samples in the SuperSplat (`.sog`) format.
- **Interactive Controls**: Orbit, pan, and zoom controls for smooth inspection.

## Acknowledgements
- [Apple SHARP](https://github.com/apple/ml-sharp) for the Gaussian Splat inference.
- [PlayCanvas](https://playcanvas.com) for the 3D rendering engine and Web Components.
- [Splat Transform](https://github.com/playcanvas/splat-transform) for PLY compression.
