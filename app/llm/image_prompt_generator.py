"""
image_prompt_generator.py

Uses Gemma to create Stable Diffusion prompts.
"""

import json

from config import (
    SCENE_PROMPTS_JSON,
    MAX_IMAGE_PROMPT_TOKENS
)

from utils.json_parser import JSONParser


class ImagePromptGenerator:

    def __init__(self, engine):

        self.engine = engine

    def build_prompt(self, analysis, scene):

        return f"""
You are an expert AI prompt engineer for Stable Diffusion XL.

Create a highly detailed image generation prompt.

Theme:
{analysis["theme"]}

Mood:
{analysis["mood"]}

Symbols:
{", ".join(analysis["symbols"])}

Visual Concepts:
{", ".join(analysis["visual_concepts"])}

Scene Description:
{scene["description"]}

Camera:
{scene["camera"]}

Lighting:
{scene["lighting"]}

Emotion:
{scene["emotion"]}

Requirements:

- Output ONLY the prompt.
- No explanations.
- Under 120 words.
- Extremely descriptive.
- Photorealistic.
- Cinematic composition.
- Volumetric lighting.
- 35mm lens.
- Shallow depth of field.
- Dramatic atmosphere.
- Film still.
- Ultra detailed.
- Masterpiece.
- Realistic textures.
- No bullet points.
"""

    def generate(self, analysis, scene):

        prompt = self.build_prompt(
            analysis,
            scene
        )

        result = self.engine.generate(
            prompt,
            max_new_tokens=MAX_IMAGE_PROMPT_TOKENS,
            temperature=0.2
        )

        return result.strip()