"""
image_generator.py

Stable Diffusion image generator for MetaCinema.
"""

import torch
from diffusers import StableDiffusionPipeline

from config import (
    SD_MODEL,
    IMAGE_DIR,
    DEVICE,
    DTYPE,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    NUM_INFERENCE_STEPS,
    GUIDANCE_SCALE,
    NEGATIVE_PROMPT,
    SEED
)


class ImageGenerator:

    def __init__(self):

        print("\nLoading Stable Diffusion...\n")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL,
            torch_dtype=DTYPE,
            safety_checker=None,
            requires_safety_checker=False
        )

        if DEVICE == "cuda":

            self.pipe.to("cuda")

            self.pipe.enable_attention_slicing()

            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("xFormers enabled.")
            except Exception:
                print("xFormers not available.")

        print("Stable Diffusion loaded successfully.\n")

    def generate_scene(
        self,
        scene_number,
        prompt
    ):

        print(f"\nGenerating Scene {scene_number}...")

        generator = torch.Generator(
            device=DEVICE
        ).manual_seed(
            SEED + scene_number
        )

        image = self.pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator
        ).images[0]

        output_path = IMAGE_DIR / f"scene_{scene_number}.png"

        image.save(output_path)

        print(f"Saved -> {output_path}")

        return output_path

    def generate_all(
        self,
        prompts
    ):

        image_paths = []

        for idx, prompt in enumerate(prompts, start=1):

            path = self.generate_scene(
                idx,
                prompt
            )

            image_paths.append(path)

        return image_paths