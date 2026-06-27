"""
gemma_engine.py

Shared Gemma inference engine.
Loads Gemma only once and provides a simple generate() API.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    GEMMA_MODEL,
    DEVICE,
    DTYPE
)


class GemmaEngine:

    def __init__(self):

        print("\nLoading Gemma model...\n")

        self.tokenizer = AutoTokenizer.from_pretrained(
            GEMMA_MODEL
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            GEMMA_MODEL,
            torch_dtype=DTYPE,
            device_map="auto"
        )

        self.model.eval()

        print("Gemma loaded successfully.\n")

    def generate(
        self,
        prompt,
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9
    ):
        """
        Generate text using Gemma.

        Returns ONLY the assistant response.
        """

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        chat = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            chat,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Remove prompt tokens
        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        return response.strip()

    def generate_json(
        self,
        prompt,
        parser,
        max_new_tokens=512
    ):
        """
        Generate JSON response and parse it.
        """

        response = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0
        )

        print("\n========== RAW GEMMA OUTPUT ==========\n")
        print(response)

        response = response.strip()

        # Remove markdown fences
        response = response.replace("```json", "")
        response = response.replace("```", "")

        data = parser.extract_json(response)

        if data is not None:
            return data

        # Try repairing truncated JSON
        fixed = response

        if '"scenes"' in fixed:
            if not fixed.rstrip().endswith("]}"):
                fixed = fixed.rstrip()

                # Close the last object if needed
                if fixed.endswith(","):
                    fixed = fixed[:-1]

                if not fixed.endswith("}"):
                    fixed += "}"

                if not fixed.endswith("]}"):
                    fixed += "]}"

            data = parser.extract_json(fixed)

            if data is not None:
                return data

        raise RuntimeError("Gemma returned invalid JSON.")

        return data