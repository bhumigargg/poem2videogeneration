"""
prompt_generator.py

Converts storyboard scenes into high-quality Stable Diffusion prompts.
"""

from typing import Dict


class PromptGenerator:

    def __init__(self):

        self.quality_prompt = (
            "masterpiece, best quality, ultra realistic, "
            "cinematic composition, dramatic lighting, "
            "film still, volumetric lighting, "
            "depth of field, 8k, highly detailed, "
            "award winning cinematography"
        )

        self.negative_prompt = (
            "low quality, blurry, watermark, text, logo, "
            "cropped, duplicate, bad anatomy, deformed, "
            "extra limbs, extra fingers, poorly drawn face, "
            "oversaturated, jpeg artifacts"
        )

    def build_prompt(
        self,
        scene: Dict,
        analysis: Dict
    ):

        theme = analysis["theme"]

        mood = analysis["mood"]

        symbols = ", ".join(
            analysis["symbols"]
        )

        visuals = ", ".join(
            analysis["visual_concepts"]
        )

        description = scene["description"]

        prompt = f"""
{description}

Theme:
{theme}

Mood:
{mood}

Important Symbols:
{symbols}

Visual Concepts:
{visuals}


Photorealistic.

Symbolic visual storytelling.

Movie scene.

{self.quality_prompt}
"""

        return " ".join(
            prompt.split()
        )

    def get_negative_prompt(self):

        return self.negative_prompt

    def print_prompt(
        self,
        prompt
    ):

        print("\n========== PROMPT ==========\n")

        print(prompt)

        print("\n============================\n")