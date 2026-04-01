"""
Judge B — Direct Gemini judge (no ML model).

Pipeline:
  Image → Gemini Vision → edible/poisonous verdict + reasoning

This serves as the baseline in the A vs. B comparison described in the proposal.
"""

import re
from PIL import Image
from google import genai
from google.genai import types


DIRECT_JUDGE_PROMPT = """
You are Judge B, an expert mycologist making a safety assessment purely
from visual inspection of a mushroom photograph.

Examine the image carefully and:
1. Identify the most prominent visual features (cap, gills, stem, color, texture).
2. Give your verdict: EDIBLE or POISONOUS.
3. State your confidence as a percentage (0–100%).
4. Explain your reasoning in 3-4 sentences.

Format your response EXACTLY as:
VERDICT: <EDIBLE or POISONOUS>
CONFIDENCE: <number>%
REASONING: <your explanation>
"""


class DirectJudge:
    """
    Verdicts based solely on Gemini's visual reasoning — no ML model.

    Parameters
    ----------
    api_key    : Gemini API key.
    model_name : Gemini model to use.
    """

    NAME = "Judge B (Direct Vision)"
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: str, model_name: str = DEFAULT_MODEL):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def judge(self, image: Image.Image) -> dict:
        """
        Returns
        -------
        {
          "verdict":    "edible" | "poisonous",
          "confidence": float (0–1),
          "narrative":  str,
        }
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[DIRECT_JUDGE_PROMPT, image],
            config=types.GenerateContentConfig(temperature=0.2),
        )
        return self._parse(response.text.strip())

    # ── private ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse(text: str) -> dict:
        verdict_match = re.search(r"VERDICT:\s*(EDIBLE|POISONOUS)", text, re.I)
        conf_match = re.search(r"CONFIDENCE:\s*(\d+(?:\.\d+)?)\s*%", text, re.I)
        reasoning_match = re.search(r"REASONING:\s*(.*)", text, re.I | re.DOTALL)

        verdict = "poisonous"  # default to safe side on parse failure
        confidence = 0.5
        narrative = text  # fall back to raw text

        if verdict_match:
            verdict = verdict_match.group(1).lower()
        if conf_match:
            confidence = float(conf_match.group(1)) / 100.0
        if reasoning_match:
            narrative = reasoning_match.group(1).strip()

        return {
            "verdict":    verdict,
            "confidence": confidence,
            "narrative":  narrative,
        }
