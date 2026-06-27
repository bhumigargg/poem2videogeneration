"""
video_editor.py

Concatenates all generated scene videos into one final movie.
"""

import subprocess
from pathlib import Path

from config import (
    VIDEO_DIR,
    FINAL_DIR
)


class VideoEditor:

    def __init__(self):

        FINAL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

    def merge(self):

        print("\nMerging videos...\n")

        videos = sorted(
            VIDEO_DIR.glob("scene_*.mp4")
        )

        if len(videos) == 0:
            raise RuntimeError(
                "No scene videos found."
            )

        list_file = FINAL_DIR / "videos.txt"

        with open(list_file, "w") as f:

            for video in videos:

                f.write(
                    f"file '{video.resolve()}'\n"
                )

        output = FINAL_DIR / "final_movie.mp4"

        command = [

            "ffmpeg",

            "-y",

            "-f",
            "concat",

            "-safe",
            "0",

            "-i",
            str(list_file),

            "-c",
            "copy",

            str(output)

        ]

        subprocess.run(
            command,
            check=True
        )

        print(f"\nFinal movie saved:\n{output}")

        return output


if __name__ == "__main__":

    VideoEditor().merge()