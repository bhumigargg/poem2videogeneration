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
FLUX_MODEL = "black-forest-labs/FLUX.1-schnell"

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

IMAGE_WIDTH = 1024

IMAGE_HEIGHT = 1024

IMAGE_STEPS = 28

IMAGE_GUIDANCE = 3.5

NEGATIVE_PROMPT = (
    "blurry, watermark, logo, text, "
    "low quality, cropped, deformed, "
    "duplicate, bad anatomy"
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