"""
config.py

Configuration for MetaCinema Cloud Pipeline
(FLUX + CogVideoX)
"""

from pathlib import Path
import torch

# =====================================================
# DIRECTORIES
# =====================================================

ROOT = Path(__file__).resolve().parent

INPUT_DIR = ROOT / "input"

OUTPUT_DIR = ROOT / "output"

IMAGE_DIR = OUTPUT_DIR / "images"

VIDEO_DIR = OUTPUT_DIR / "videos"

FINAL_DIR = OUTPUT_DIR / "final"

for folder in [
    INPUT_DIR,
    OUTPUT_DIR,
    IMAGE_DIR,
    VIDEO_DIR,
    FINAL_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# =====================================================
# INPUT FILES
# =====================================================

SCENE_PROMPTS_JSON = INPUT_DIR / "scene_prompts.json"

# =====================================================
# MODELS
# =====================================================

# Image generation
FLUX_MODEL = "black-forest-labs/FLUX.1-dev"

# Image → Video
COGVIDEO_MODEL = "THUDM/CogVideoX-5b-I2V"

# =====================================================
# DEVICE
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# =====================================================
# IMAGE GENERATION
# =====================================================

# =====================================================
# IMAGE GENERATION
# =====================================================

IMAGE_MODEL = "stabilityai/sdxl-turbo"

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

# SDXL Turbo works best with very few steps
IMAGE_STEPS = 4

IMAGE_GUIDANCE = 0.0

NEGATIVE_PROMPT = (
    "blurry, watermark, text, logo, low quality, "
    "deformed, duplicate, cropped"
)
# =====================================================
# VIDEO GENERATION
# =====================================================

VIDEO_STEPS = 50

VIDEO_FPS = 8

VIDEO_NUM_FRAMES = 49

VIDEO_GUIDANCE = 6.0

# =====================================================
# RANDOMNESS
# =====================================================

SEED = 42