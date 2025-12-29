#!/usr/bin/env python3
"""
Compress all PLY files in static/samples to .sog format using splat-transform
"""
import os
import subprocess
from pathlib import Path

SAMPLES_DIR = Path(__file__).parent / "static" / "samples"

def compress_ply_file(ply_path):
    """Compress a PLY file to .sog format using splat-transform"""
    sog_path = ply_path.with_suffix('.sog')
    
    # Skip if already compressed
    if sog_path.exists():
        print(f"Skipping {ply_path.name} (already compressed as .sog)")
        return sog_path
    
    print(f"Compressing {ply_path.name} to .sog...")
    
    try:
        # Run splat-transform
        result = subprocess.run(
            ['splat-transform', str(ply_path), str(sog_path)],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"  ✓ Created {sog_path.name}")
        print(f"  Original: {ply_path.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"  Compressed: {sog_path.stat().st_size / 1024 / 1024:.2f} MB")
        print(f"  Ratio: {sog_path.stat().st_size / ply_path.stat().st_size * 100:.1f}%")
        return sog_path
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error compressing {ply_path.name}: {e.stderr}")
        return None
    except FileNotFoundError:
        print("  ✗ splat-transform not found. Install it with:")
        print("    npm install -g @playcanvas/splat-transform")
        return None

def main():
    print("Compressing PLY files to .sog format in static/samples...")
    print()
    
    # Find all PLY files (but not already compressed ones)
    ply_files = [f for f in SAMPLES_DIR.glob("*.ply") if not f.name.endswith('.compressed')]
    
    if not ply_files:
        print("No PLY files found to compress.")
        return
    
    print(f"Found {len(ply_files)} PLY files to compress:")
    for ply_file in ply_files:
        compress_ply_file(ply_file)
        print()
    
    print("Done!")

if __name__ == "__main__":
    main()
