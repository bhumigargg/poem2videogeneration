"""
storyboard_generator.py

Generates a cinematic storyboard from the poem analysis.
"""

from config import (
    STORYBOARD_JSON,
    MAX_STORYBOARD_TOKENS
)

from utils.json_parser import JSONParser


class StoryboardGenerator:

    def __init__(self, engine):

        self.engine = engine

    def build_prompt(self, analysis):

        return f"""
You are an Oscar-winning film director and storyboard artist.

Your task is to convert the poem analysis into a cinematic storyboard.

Theme:
{analysis["theme"]}

Mood:
{analysis["mood"]}

Symbols:
{", ".join(analysis["symbols"])}

Visual Concepts:
{", ".join(analysis["visual_concepts"])}

Metaphors:
{analysis["metaphors"]}

Return ONLY valid JSON.

Required format:

{{
    "scenes":[
        {{
            "scene_number":1,
            "title":"",
            "description":"",
            "camera":"",
            "lighting":"",
            "emotion":""
        }}
    ]
}}

Rules:

Requirements:

1. Create EXACTLY 4 scenes.
2. Each description must be ONE sentence only.
3. Maximum 25 words per description.
4. Camera must be short.
5. Lighting must be short.
6. Emotion must be one or two words.
7. Return ONLY valid JSON.
8. Finish the JSON completely.
9. No markdown.
"""

    def generate(self, analysis):

        prompt = self.build_prompt(analysis)

        storyboard = self.engine.generate_json(
            prompt=prompt,
            parser=JSONParser,
            max_new_tokens=MAX_STORYBOARD_TOKENS
        )

        scenes = storyboard.get("scenes", [])

        if len(scenes) < 4:
            raise RuntimeError(
                f"Gemma generated only {len(scenes)} scene(s). Expected 4."
            )

        JSONParser.save_json(
            storyboard,
            STORYBOARD_JSON
        )

        return storyboard

    def pretty_print(self, storyboard):

        print("\n========== STORYBOARD ==========\n")

        for scene in storyboard["scenes"]:

            print("=" * 60)

            print(f"Scene {scene['scene_number']}")

            print(f"Title      : {scene.get('title','')}")

            print(f"Emotion    : {scene.get('emotion','')}")

            print(f"Camera     : {scene.get('camera','')}")

            print(f"Lighting   : {scene.get('lighting','')}")

            print("\nDescription:\n")

            print(scene.get("description",""))

            print()