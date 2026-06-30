import gc
import torch

from sdxl_generator import SDXLGenerator
from cogvideo_generator import CogVideoGenerator
from video_editor import VideoEditor


def clear_memory():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        torch.cuda.ipc_collect()


def main():

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