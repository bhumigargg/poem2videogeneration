"""
video_prompt_generator.py

Uses Gemma to create CogVideoX prompts.
"""

import json

from config import (
    SCENE_PROMPTS_JSON,
    MAX_VIDEO_PROMPT_TOKENS
)

from utils.json_parser import JSONParser


class VideoPromptGenerator:

    def __init__(self, engine):

        self.engine = engine

    def build_prompt(self, analysis, scene):

        return f"""
You are an Oscar-winning cinematographer.

Create a professional CogVideoX prompt.

Theme:
{analysis["theme"]}

Mood:
{analysis["mood"]}

Symbols:
{", ".join(analysis["symbols"])}

Visual Concepts:
{", ".join(analysis["visual_concepts"])}

Scene:
{scene["description"]}

Camera:
{scene["camera"]}

Lighting:
{scene["lighting"]}

Emotion:
{scene["emotion"]}

Requirements:

- Output ONLY the prompt.
- Describe realistic camera movement.
- Mention environmental motion.
- Mention subtle object movement.
- Mention cinematic realism.
- Mention emotional pacing.
- Mention realistic lighting changes.
- Mention smooth motion.
- Mention photorealistic movie quality.
- Under 150 words.
"""

    def generate(self, analysis, scene):

        prompt = self.build_prompt(
            analysis,
            scene
        )

        result = self.engine.generate(
            prompt,
            max_new_tokens=MAX_VIDEO_PROMPT_TOKENS,
            temperature=0.3
        )

        return result.strip()