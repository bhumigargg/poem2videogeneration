import json

from llm.gemma_engine import GemmaEngine
from llm.poem_analyzer import PoemAnalyzer
from llm.storyboard_generator import StoryboardGenerator
from llm.image_prompt_generator import ImagePromptGenerator
from llm.video_prompt_generator import VideoPromptGenerator

from image.image_generator import ImageGenerator

from config import SCENE_PROMPTS_JSON


def read_poem():

    print("\nEnter your poem.")
    print("Type END on a new line when finished.\n")

    lines = []

    while True:

        line = input()

        if line.strip().upper() == "END":
            break

        lines.append(line)

    return "\n".join(lines)


def main():

    poem = read_poem()

    # -------------------------------------------------
    # Load Gemma ONLY ONCE
    # -------------------------------------------------

    engine = GemmaEngine()

    analyzer = PoemAnalyzer(engine)

    storyboard_generator = StoryboardGenerator(engine)

    image_prompt_generator = ImagePromptGenerator(engine)

    video_prompt_generator = VideoPromptGenerator(engine)

    #image_generator = ImageGenerator()

    # -------------------------------------------------
    # Literary Analysis
    # -------------------------------------------------

    print("\nAnalyzing poem...\n")

    analysis = analyzer.analyze(poem)

    analyzer.pretty_print(analysis)

    # -------------------------------------------------
    # Storyboard
    # -------------------------------------------------

    print("\nGenerating storyboard...\n")

    storyboard = storyboard_generator.generate(
        analysis
    )

    storyboard_generator.pretty_print(
        storyboard
    )

    # -------------------------------------------------
    # Prompt Generation
    # -------------------------------------------------

    print("\nGenerating prompts...\n")

    image_prompts = []

    video_prompts = []

    prompt_json = {}

    scenes = storyboard["scenes"]

    for scene in scenes:

        image_prompt = image_prompt_generator.generate(
            analysis,
            scene
        )

        video_prompt = video_prompt_generator.generate(
            analysis,
            scene
        )

        image_prompts.append(
            image_prompt
        )

        video_prompts.append(
            video_prompt
        )

        prompt_json[
            f"scene_{scene['scene_number']}"
        ] = {

            "title": scene["title"],

            "description": scene["description"],

            "camera": scene["camera"],

            "lighting": scene["lighting"],

            "emotion": scene["emotion"],

            "image_prompt": image_prompt,

            "video_prompt": video_prompt

        }

        print("\n" + "=" * 60)

        print(
            f"Scene {scene['scene_number']}"
        )

        print("\nIMAGE PROMPT\n")

        print(image_prompt)

        print("\nVIDEO PROMPT\n")

        print(video_prompt)

    # -------------------------------------------------
    # Save prompts
    # -------------------------------------------------

    with open(
        SCENE_PROMPTS_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            prompt_json,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(
        f"\nScene prompts saved to:\n{SCENE_PROMPTS_JSON}"
    )

    # -------------------------------------------------
    # Image Generation
    # -------------------------------------------------

    
    print("\nGenerated files:")

    print("✓ analysis.json")

    print("✓ storyboard.json")

    print("✓ scene_prompts.json")

    
    print("\nReady for CogVideoX cloud generation.")


if __name__ == "__main__":

    main()