"""
Complete MetaCinema Cloud Pipeline
"""

from flux_generator import FluxGenerator

from cogvideo_generator import CogVideoGenerator

from video_editor import VideoEditor


def main():

    print("\n==============================")
    print("MetaCinema Cloud Pipeline")
    print("==============================\n")

    # ------------------------------------

    flux = FluxGenerator()

    flux.generate_all()

    # ------------------------------------

    cog = CogVideoGenerator()

    cog.generate_all()

    # ------------------------------------

    editor = VideoEditor()

    editor.merge()

    print("\n==============================")
    print("Pipeline Complete")
    print("==============================")

    print("\nOutputs:")

    print("output/images")

    print("output/videos")

    print("output/final/final_movie.mp4")


if __name__ == "__main__":

    main()