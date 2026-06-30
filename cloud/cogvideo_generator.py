"""
CogVideoX Image → Video Generator

Generates one video for every scene image.
"""

import json
import torch
import gc
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import (
    load_image,
    export_to_video
)

from config import (
    COGVIDEO_MODEL,
    SCENE_PROMPTS_JSON,
    IMAGE_DIR,
    VIDEO_DIR,
    VIDEO_STEPS,
    VIDEO_GUIDANCE,
    VIDEO_NUM_FRAMES,
    VIDEO_FPS,
    DTYPE,
    SEED
)


class CogVideoGenerator:

    def __init__(self):

        print("\nLoading CogVideoX...\n")
        self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            COGVIDEO_MODEL,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        # Memory optimizations
        self.pipe.enable_model_cpu_offload()

        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()

        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        print("✓ CogVideoX loaded.\n")

    def load_scene_prompts(self):

        with open(
            SCENE_PROMPTS_JSON,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    def generate_scene(
        self,
        scene_name,
        image_path,
        video_prompt
    ):

        print(f"\nGenerating {scene_name}...")

        image = load_image(str(image_path))
        image = image.resize((512, 512))
        generator=torch.Generator().manual_seed(SEED)

        frames = self.pipe(

            prompt=video_prompt,

            image=image,

            num_videos_per_prompt=1,

            num_inference_steps=VIDEO_STEPS,

            num_frames=VIDEO_NUM_FRAMES,

            guidance_scale=VIDEO_GUIDANCE,

            generator=generator,

            use_dynamic_cfg=True

        ).frames[0]
        video_prompt = " ".join(video_prompt.split()[:80])
        output_path = VIDEO_DIR / f"{scene_name}.mp4"

        export_to_video(
            frames,
            str(output_path),
            fps=VIDEO_FPS
        )
        del frames
        del image

        gc.collect()

        torch.cuda.empty_cache()
        print(f"✓ Saved -> {output_path}")

        return output_path
        
    def generate_all(self):

        prompts = self.load_scene_prompts()

        videos = []

        for scene_name in sorted(prompts.keys()):

            image_path = IMAGE_DIR / f"{scene_name}.png"

            video_prompt = prompts[scene_name]["video_prompt"]

            path = self.generate_scene(

                scene_name,

                image_path,

                video_prompt

            )

            videos.append(path)

        print("\n==============================")
        print("All videos generated.")
        print("==============================\n")

        return videos
    def unload(self):

        print("\nUnloading CogVideoX...\n")

        del self.pipe

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("✓ CogVideoX unloaded.\n")

if __name__ == "__main__":

    generator = CogVideoGenerator()

    generator.generate_all()