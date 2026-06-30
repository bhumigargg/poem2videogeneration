import gc
import torch
import shutil
import os
import glob
from sdxl_generator import SDXLGenerator
from cogvideo_generator import CogVideoGenerator
from video_editor import VideoEditor


def clear_memory():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        torch.cuda.ipc_collect()


def main():
    # -----------------------------------------
    # Find uploaded scene_prompts.json
    # -----------------------------------------

    os.makedirs("inputs", exist_ok=True)

    matches = glob.glob("/kaggle/input/datasets/bhumigarg012/scene-prompts/scene_prompts.json", recursive=True)

    if len(matches) == 0:
        raise FileNotFoundError(
            "scene_prompts.json not found in /kaggle/input"
        )

    source = matches[0]
    destination = "inputs/scene_prompts.json"

    shutil.copy(source, destination)

    print(f"Copied:\n{source}\n->\n{destination}")
    print("\n==============================")
    print("MetaCinema Cloud Pipeline")
    print("==============================")

    ###########################################
    # IMAGE GENERATION
    ###########################################

    sdxl = SDXLGenerator()

    sdxl.generate_all()

    sdxl.unload()

    del sdxl

    clear_memory()

    ###########################################
    # VIDEO GENERATION
    ###########################################

    cog = CogVideoGenerator()

    cog.generate_all()

    cog.unload()

    del cog

    clear_memory()

    ###########################################
    # MERGE
    ###########################################

    editor = VideoEditor()

    editor.merge()

    print("\n===================================")
    print("Pipeline Finished Successfully")
    print("===================================")


if __name__ == "__main__":
    main()