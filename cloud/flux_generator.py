"""
flux_generator.py

Reads scene_prompts.json and generates one image
per scene using FLUX.

Author: MetaCinema
"""

import json
import torch

from diffusers import FluxPipeline

from config import (
    FLUX_MODEL,
    SCENE_PROMPTS_JSON,
    IMAGE_DIR,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_STEPS,
    IMAGE_GUIDANCE,
    SEED,
    DTYPE
)


class FluxGenerator:

    def __init__(self):

        print("\nLoading FLUX model...\n")

        self.pipe = FluxPipeline.from_pretrained(
            FLUX_MODEL,
            torch_dtype=DTYPE
        )

        # Memory optimization
        self.pipe.enable_model_cpu_offload()

        print("✓ FLUX loaded successfully.\n")

    def load_prompts(self):

        with open(
            SCENE_PROMPTS_JSON,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    def generate_scene(
        self,
        scene_name,
        image_prompt
    ):

        print(f"\nGenerating {scene_name}...")

        generator = torch.Generator(
            device="cuda"
        ).manual_seed(
            SEED
        )

        image = self.pipe(

            prompt=image_prompt,

            width=IMAGE_WIDTH,

            height=IMAGE_HEIGHT,

            guidance_scale=IMAGE_GUIDANCE,

            num_inference_steps=IMAGE_STEPS,

            max_sequence_length=512,

            generator=generator

        ).images[0]

        output_path = IMAGE_DIR / f"{scene_name}.png"

        image.save(output_path)

        print(f"✓ Saved -> {output_path}")

        return output_path

    def generate_all(self):

        prompts = self.load_prompts()

        generated_images = []

        print("\nGenerating images...\n")

        for scene_name in sorted(prompts.keys()):

            image_prompt = prompts[scene_name]["image_prompt"]

            image_path = self.generate_scene(
                scene_name,
                image_prompt
            )

            generated_images.append(
                image_path
            )

        print("\n===================================")
        print("All images generated successfully.")
        print("===================================\n")

        return generated_images


if __name__ == "__main__":

    generator = FluxGenerator()

    generator.generate_all()