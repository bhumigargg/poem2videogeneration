"""
poem_analyzer.py

Uses Gemma to perform literary analysis of a poem.
"""

from config import (
    ANALYSIS_JSON,
    MAX_ANALYSIS_TOKENS
)

from utils.json_parser import JSONParser


class PoemAnalyzer:

    def __init__(self, engine):

        self.engine = engine

    def build_prompt(self, poem):

        return f"""
You are an expert literary analyst.

Analyze the poem below.

Return ONLY valid JSON.

Required format:

{{
    "theme":"",
    "mood":"",
    "symbols":[],
    "metaphors":{{}},
    "visual_concepts":[]
}}

Rules:

- Do NOT explain.
- Do NOT use markdown.
- Do NOT use ```json.
- Symbols should be short phrases.
- Metaphors should be key-value pairs.
- Visual concepts should be concrete visual elements.

Poem:

{poem}
"""

    def analyze(self, poem):

        prompt = self.build_prompt(poem)

        analysis = self.engine.generate_json(
            prompt=prompt,
            parser=JSONParser,
            max_new_tokens=MAX_ANALYSIS_TOKENS
        )

        JSONParser.save_json(
            analysis,
            ANALYSIS_JSON
        )

        return analysis

    def pretty_print(self, analysis):

        print("\n========== ANALYSIS ==========\n")

        print("Theme:")
        print(analysis.get("theme", "Unknown"))

        print("\nMood:")
        print(analysis.get("mood", "Unknown"))

        print("\nSymbols:")

        symbols = analysis.get("symbols", [])

        if symbols:
            for symbol in symbols:
                print(f" • {symbol}")
        else:
            print("None")

        print("\nMetaphors:")

        metaphors = analysis.get("metaphors", {})

        if metaphors:
            for key, value in metaphors.items():
                print(f" • {key}")
                print(f"   {value}")
        else:
            print("None")

        print("\nVisual Concepts:")

        visuals = analysis.get("visual_concepts", [])

        if visuals:
            for visual in visuals:
                print(f" • {visual}")
        else:
            print("None")

        print()