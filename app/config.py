from pathlib import Path
import torch

# =====================================================
# PROJECT ROOT
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# =====================================================
# LOCAL MODEL PATHS
# =====================================================

GEMMA_MODEL = r"E:\HFModels\gemma-2-2b-it"

SD_MODEL = "stabilityai/stable-diffusion-2-1"

# Used later on cloud only
COGVIDEO_MODEL = "THUDM/CogVideoX-2B"

# =====================================================
# OUTPUT DIRECTORIES
# =====================================================

OUTPUT_DIR = PROJECT_ROOT / "outputs"

IMAGE_DIR = OUTPUT_DIR / "images"

VIDEO_DIR = OUTPUT_DIR / "videos"

PROMPT_DIR = OUTPUT_DIR / "prompts"

for folder in [
    OUTPUT_DIR,
    IMAGE_DIR,
    VIDEO_DIR,
    PROMPT_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

# =====================================================
# JSON FILES
# =====================================================

ANALYSIS_JSON = OUTPUT_DIR / "analysis.json"

STORYBOARD_JSON = OUTPUT_DIR / "storyboard.json"

SCENE_PROMPTS_JSON = OUTPUT_DIR / "scene_prompts.json"

# =====================================================
# GEMMA
# =====================================================

MAX_ANALYSIS_TOKENS = 300

MAX_STORYBOARD_TOKENS = 700

MAX_IMAGE_PROMPT_TOKENS = 180

MAX_VIDEO_PROMPT_TOKENS = 220

# =====================================================
# STABLE DIFFUSION
# =====================================================

IMAGE_WIDTH = 512

IMAGE_HEIGHT = 512

NUM_INFERENCE_STEPS = 30

GUIDANCE_SCALE = 7.5

NEGATIVE_PROMPT = (
    "low quality, blurry, watermark, text, logo, "
    "cropped, duplicate, bad anatomy, extra fingers, "
    "deformed face, oversaturated"
)

# =====================================================
# DEVICE
# =====================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# =====================================================
# VIDEO SETTINGS
# =====================================================

FPS = 8

VIDEO_DURATION = 4

# =====================================================
# RANDOM SEED
# =====================================================

SEED = 42