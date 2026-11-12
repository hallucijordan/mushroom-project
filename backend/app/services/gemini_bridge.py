"""
GeminiBridge — unified service wrapping both Gemini-based judges.

Judge A (feature extraction):
    Image → Gemini Vision → structured feature dict → ML classifier

Judge B (direct vision):
    Image → Gemini Vision → edible/poisonous verdict + reasoning

Keeping a single shared genai.Client avoids re-initialising the SDK on
every request and makes the API key a single point of configuration.
"""

import json
import re
from PIL import Image
from google import genai
from google.genai import types

# ── prompts ────────────────────────────────────────────────────────────────────

_FEATURE_EXTRACTION_PROMPT = """
You are a mycology expert analyzing a mushroom photograph.
Extract the following 20 features from the image and return them as a
JSON object with exactly these keys.  Use the allowed values listed.
If you cannot determine a feature from the image, use null.

Features and allowed values:
- cap-diameter: numeric in cm (e.g. 5.5)
- cap-shape: b=bell, c=conical, x=convex, f=flat, s=sunken, p=spherical, o=others
- cap-surface: i=fibrous, g=grooves, y=scaly, s=smooth, h=shiny, l=leathery, k=silky, t=sticky, w=wrinkled, e=fleshy
- cap-color: n=brown, b=buff, g=gray, r=green, p=pink, u=purple, e=red, w=white, y=yellow, l=blue, o=orange, k=black
- does-bruise-or-bleed: t=yes, f=no
- gill-attachment: a=adnate, x=adnexed, d=decurrent, e=free, s=sinuate, p=pores, f=none, ?=unknown
- gill-spacing: c=close, d=distant, f=none
- gill-color: (same color codes as cap-color, plus b=buff, o=orange)
- stem-height: numeric in cm
- stem-width: numeric in mm
- stem-root: b=bulbous, s=swollen, c=club, u=cup, e=equal, z=rhizomorphs, r=rooted
- stem-surface: (same as cap-surface)
- stem-color: (same color codes as cap-color)
- veil-type: p=partial, u=universal
- veil-color: (same color codes as cap-color)
- has-ring: t=yes, f=no
- ring-type: c=cobwebby, e=evanescent, r=flaring, g=grooved, l=large, p=pendant, s=sheathing, z=zone, y=scaly, m=movable, f=none, ?=unknown
- spore-print-color: (same color codes as cap-color)
- habitat: d=woods, g=grasses, m=meadows, p=paths, h=heaths, u=urban, w=waste, l=leaves
- season: s=spring, u=summer, a=autumn, w=winter

Return ONLY a valid JSON object, no explanation text.
"""

_DIRECT_JUDGE_PROMPT = """
You are Judge B, an expert mycologist making a safety assessment purely
from visual inspection of a mushroom photograph.

Examine the image carefully and:
1. Identify the most prominent visual features (cap, gills, stem, color, texture).
2. Give your verdict: EDIBLE or POISONOUS.
3. State your confidence as a percentage (0-100%).
4. Explain your reasoning in 3-4 sentences.

Format your response EXACTLY as:
VERDICT: <EDIBLE or POISONOUS>
CONFIDENCE: <number>%
REASONING: <your explanation>
"""

# ── GeminiBridge ───────────────────────────────────────────────────────────────

class GeminiBridge:
    """
    Single entry point for all Gemini Vision calls.

    Parameters
    ----------
    api_key    : Gemini API key (from Settings.gemini_api_key).
    model_name : Gemini model to use for both judges.
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: str, model_name: str = DEFAULT_MODEL):
        # One shared client for all calls in this process
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    # ── Judge A ────────────────────────────────────────────────────────────────

    def extract_features(self, image: Image.Image) -> dict:
        """
        Judge A — ask Gemini to extract the 20 structured mushroom features.

        Parameters
        ----------
        image : PIL.Image.Image  (RGB)

        Returns
        -------
        dict  {feature_name: value, ...}
              Missing / undetectable features are None.
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[_FEATURE_EXTRACTION_PROMPT, image],
            config=types.GenerateContentConfig(temperature=0.1),
        )
        return self._parse_json(response.text)

    # ── Judge B ────────────────────────────────────────────────────────────────

    def direct_judge(self, image: Image.Image) -> dict:
        """
        Judge B — ask Gemini for a direct edible/poisonous verdict.

        Parameters
        ----------
        image : PIL.Image.Image  (RGB)

        Returns
        -------
        {
            "verdict":    "edible" | "poisonous",
            "confidence": float (0.0 – 1.0),
            "narrative":  str,
        }
        Default verdict is "poisonous" when parsing fails (safe side).
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[_DIRECT_JUDGE_PROMPT, image],
            config=types.GenerateContentConfig(temperature=0.2),
        )
        return self._parse_verdict(response.text.strip())

    # ── private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract a JSON object from the model response.

        Handles markdown code fences (```json ... ```) that the model
        sometimes adds around its output.
        """
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fall back: find the first {...} block in the response
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if m:
                return json.loads(m.group())
            return {}

    @staticmethod
    def _parse_verdict(text: str) -> dict:
        """Parse the structured VERDICT / CONFIDENCE / REASONING response."""
        verdict_match = re.search(r"VERDICT:\s*(EDIBLE|POISONOUS)", text, re.I)
        conf_match = re.search(r"CONFIDENCE:\s*(\d+(?:\.\d+)?)\s*%", text, re.I)
        reasoning_match = re.search(r"REASONING:\s*(.*)", text, re.I | re.DOTALL)

        # Default to poisonous on parse failure — safer assumption
        verdict = "poisonous"
        confidence = 0.5
        narrative = text  # keep raw text as fallback

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
