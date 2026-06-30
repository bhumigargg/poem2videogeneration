import json
import torch
import gc

from diffusers import AutoPipelineForText2Image

from config import (
    IMAGE_MODEL,
    SCENE_PROMPTS_JSON,
    IMAGE_DIR,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_STEPS,
    IMAGE_GUIDANCE,
    NEGATIVE_PROMPT,
    DTYPE,
    SEED
)


class SDXLGenerator:

    def __init__(self):

        print("\nLoading SDXL Turbo...\n")

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            IMAGE_MODEL,
            torch_dtype=DTYPE,
            variant="fp16"
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe.to(device)
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        print("✓ SDXL Turbo loaded.\n")

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
        prompt
    ):

        print(f"\nGenerating {scene_name}")

        generator = torch.Generator(
            device="cuda"
        ).manual_seed(
            SEED
        )

        image = self.pipe(

            prompt=prompt,

            negative_prompt=NEGATIVE_PROMPT,

            width=IMAGE_WIDTH,

            height=IMAGE_HEIGHT,

            num_inference_steps=IMAGE_STEPS,

            guidance_scale=IMAGE_GUIDANCE,

            generator=generator

        ).images[0]

        output = IMAGE_DIR / f"{scene_name}.png"

        image.save(output)

        print(f"✓ Saved {output}")

        return output

    def generate_all(self):

        prompts = self.load_prompts()

        images = []

        for scene_name in sorted(prompts.keys()):

            path = self.generate_scene(
                scene_name,
                prompts[scene_name]["image_prompt"]
            )

            images.append(path)

        return images

    def unload(self):

        print("\nUnloading SDXL...\n")

        del self.pipe

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("✓ SDXL unloaded.\n")
if __name__ == "__main__":

    SDXLGenerator().generate_all()