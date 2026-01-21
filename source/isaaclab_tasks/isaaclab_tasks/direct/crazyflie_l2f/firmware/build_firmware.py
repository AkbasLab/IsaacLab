#!/usr/bin/env python3
"""
Firmware Build Script for Crazyflie RL Controller

This script builds Crazyflie firmware with a trained RL policy checkpoint.
The output is cf2.bin which can be flashed using cfloader.

The build uses Docker for reproducibility. The Docker image contains:
- gcc-arm-none-eabi cross-compiler for ARM Cortex-M4
- Bitcraze crazyflie-firmware
- rl_tools neural network inference headers
- Controller source files (from learning_to_fly_controller)

Usage:
    python build_firmware.py --checkpoint path/to/actor.h --output path/to/output/
    
Output:
    cf2.bin - Flash with: cfloader flash cf2.bin stm32-fw

Requirements:
    - Docker installed and running
    - Trained policy exported to rl_tools checkpoint format (actor.h)
"""

import argparse
import subprocess
import sys
from pathlib import Path


DOCKER_IMAGE = "crazyflie-l2f-firmware"


def check_docker():
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode != 0:
            print("Error: Docker is not running.")
            print("Please start Docker Desktop and try again.")
            return False
        return True
    except FileNotFoundError:
        print("Error: Docker is not installed.")
        print("Please install Docker: https://docs.docker.com/get-docker/")
        return False
    except subprocess.TimeoutExpired:
        print("Error: Docker command timed out.")
        return False


def docker_image_exists(image_name: str) -> bool:
    """Check if a Docker image exists locally."""
    result = subprocess.run(
        ["docker", "images", "-q", image_name],
        capture_output=True,
        text=True
    )
    return bool(result.stdout.strip())


def build_docker_image(dockerfile_dir: Path, image_name: str) -> bool:
    """Build the Docker image for firmware compilation."""
    print(f"Building Docker image: {image_name}")
    print(f"This may take several minutes on first run...")
    print(f"  (cloning crazyflie-firmware, rl_tools, etc.)")
    
    result = subprocess.run(
        ["docker", "build", "-t", image_name, str(dockerfile_dir)],
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error: Failed to build Docker image")
        return False
    
    print(f"Docker image built successfully: {image_name}")
    return True


def convert_path_for_docker(path: Path) -> str:
    """Convert a Windows path to Docker-compatible format."""
    path_str = str(path.resolve()).replace("\\", "/")
    
    # On Windows, convert C:/path to /c/path for Docker
    if sys.platform == "win32" and len(path_str) >= 2 and path_str[1] == ":":
        path_str = "/" + path_str[0].lower() + path_str[2:]
    
    return path_str


def build_firmware(
    checkpoint_path: Path,
    output_dir: Path,
    image_name: str = DOCKER_IMAGE
) -> bool:
    """
    Build Crazyflie firmware with the given checkpoint.
    
    The Docker container:
    1. Mounts actor.h at /controller/data/actor.h
    2. Runs make to compile firmware with OOT controller
    3. Copies cf2.bin to /output
    
    Args:
        checkpoint_path: Path to actor.h checkpoint file
        output_dir: Directory to output cf2.bin
        image_name: Docker image name
        
    Returns:
        True if build succeeded, False otherwise
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert paths for Docker
    checkpoint_mount = convert_path_for_docker(checkpoint_path)
    output_mount = convert_path_for_docker(output_dir)
    
    print(f"\n=== Building Crazyflie Firmware ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    
    # Run Docker container
    # Mount: actor.h -> /controller/data/actor.h (read-only)
    # Mount: output dir -> /output
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{checkpoint_mount}:/controller/data/actor.h:ro",
        "-v", f"{output_mount}:/output",
        image_name
    ]
    
    print(f"\nRunning: docker run ...")
    result = subprocess.run(cmd, text=True)
    
    if result.returncode != 0:
        print("\nError: Firmware build failed")
        print("Check the build output above for errors.")
        return False
    
    # Verify output
    firmware_path = output_dir / "cf2.bin"
    if firmware_path.exists():
        size_bytes = firmware_path.stat().st_size
        print(f"\n=== Build Successful ===")
        print(f"Firmware: {firmware_path}")
        print(f"Size: {size_bytes:,} bytes ({size_bytes/1024:.1f} KB)")
        print(f"\nTo flash to Crazyflie:")
        print(f"  cfloader flash {firmware_path} stm32-fw")
        return True
    else:
        print(f"\nError: Expected output not found: {firmware_path}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Build Crazyflie firmware with RL policy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build firmware with trained policy
    python build_firmware.py -c checkpoints/actor.h -o firmware/
    
    # Rebuild Docker image (if controller code changed)
    python build_firmware.py -c actor.h -o firmware/ --rebuild
    
Output:
    The build produces cf2.bin which can be flashed to Crazyflie using:
    
        cfloader flash firmware/cf2.bin stm32-fw
    
    Or using the Crazyflie client GUI.
"""
    )
    
    parser.add_argument(
        "-c", "--checkpoint",
        type=Path,
        required=True,
        help="Path to actor.h checkpoint file"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output directory for cf2.bin"
    )
    
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild Docker image"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        default=DOCKER_IMAGE,
        help=f"Docker image name (default: {DOCKER_IMAGE})"
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1
    
    # Check Docker
    if not check_docker():
        return 1
    
    # Build Docker image if needed
    script_dir = Path(__file__).parent
    if not docker_image_exists(args.image) or args.rebuild:
        if not build_docker_image(script_dir, args.image):
            return 1
    else:
        print(f"Using existing Docker image: {args.image}")
    
    # Build firmware
    success = build_firmware(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        image_name=args.image
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
