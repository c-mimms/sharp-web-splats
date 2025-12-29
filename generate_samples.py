
import os
import torch
import logging
from pathlib import Path
from PIL import Image
from sharp.models import PredictorParams, create_predictor
from sharp.utils import io
from sharp.utils import logging as logging_utils
from sharp.utils.gaussians import save_ply
from app import predict_image, load_model, device

# Setup logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("generate_samples")

SAMPLES_DIR = Path("static/samples")

def generate_samples():
    if not SAMPLES_DIR.exists():
        LOGGER.error(f"Samples directory {SAMPLES_DIR} not found.")
        return

    # Load model (reusing logic from app.py)
    load_model()

    # Iterate over images
    extensions = {".jpg", ".jpeg", ".png"}
    for file_path in SAMPLES_DIR.iterdir():
        if file_path.suffix.lower() in extensions:
            ply_path = file_path.with_suffix(".ply")
            
            if ply_path.exists():
                 LOGGER.info(f"Skipping {file_path.name}, PLY already exists.")
                 continue

            LOGGER.info(f"Processing {file_path.name}...")
            
            try:
                # Load image
                image, _, f_px = io.load_rgb(file_path)
                height, width = image.shape[:2]

                # Inference
                gaussians = predict_image(image, f_px)
                
                # Save PLY
                save_ply(gaussians, f_px, (height, width), ply_path)
                LOGGER.info(f"Generated {ply_path.name}")
                
            except Exception as e:
                LOGGER.error(f"Failed to process {file_path.name}: {e}")

if __name__ == "__main__":
    generate_samples()
