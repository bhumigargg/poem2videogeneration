"""
json_parser.py

Utility for extracting valid JSON from LLM responses.
"""

import json
import re


class JSONParser:

    @staticmethod
    def extract_json(text: str):
        """
        Extract the first valid JSON object from an LLM response.

        Returns:
            dict | None
        """

        if not text:
            return None

        # ----------------------------------------
        # Remove markdown code fences
        # ----------------------------------------

        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

        # ----------------------------------------
        # Try direct parsing first
        # ----------------------------------------

        try:
            return json.loads(text)
        except Exception:
            pass

        # ----------------------------------------
        # Find first {...} block
        # ----------------------------------------

        start = text.find("{")

        if start == -1:
            return None

        brace_count = 0
        end = None

        for i in range(start, len(text)):

            if text[i] == "{":
                brace_count += 1

            elif text[i] == "}":
                brace_count -= 1

                if brace_count == 0:
                    end = i
                    break

        if end is None:
            return None

        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)

        except Exception:
            pass

        # ----------------------------------------
        # Regex fallback
        # ----------------------------------------

        matches = re.findall(
            r"\{.*\}",
            text,
            flags=re.DOTALL
        )

        for match in matches:

            try:
                return json.loads(match)

            except Exception:
                continue

        return None

    @staticmethod
    def save_json(data, filepath):
        """
        Save dictionary as formatted JSON.
        """

        with open(filepath, "w", encoding="utf-8") as f:

            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False
            )

    @staticmethod
    def load_json(filepath):
        """
        Load JSON file.
        """

        with open(filepath, "r", encoding="utf-8") as f:

            return json.load(f)

    @staticmethod
    def pretty_print(data):
        """
        Pretty-print JSON.
        """

        print(
            json.dumps(
                data,
                indent=4,
                ensure_ascii=False
            )
        )