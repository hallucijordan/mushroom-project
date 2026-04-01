"""
Gemini Vision feature extractor.

Given a mushroom image, asks Gemini to output the 20 structured features
used by the classifier.  Returns a dict suitable for MushroomPreprocessor.
"""

import json
import re
from pathlib import Path
from google import genai
from google.genai import types
from PIL import Image

# ── prompt ─────────────────────────────────────────────────────────────────

FEATURE_EXTRACTION_PROMPT = """
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

# ── extractor ──────────────────────────────────────────────────────────────

class GeminiExtractor:
    """
    Calls Gemini Vision to extract structured mushroom features from an image.

    Parameters
    ----------
    api_key : str
        Gemini API key.
    model_name : str
        Gemini model to use (default: gemini-2.0-flash).
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, api_key: str, model_name: str = DEFAULT_MODEL):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def extract_features(self, image: Image.Image) -> dict:
        """
        Extract the 20 mushroom features from an image.

        Parameters
        ----------
        image : PIL.Image.Image

        Returns
        -------
        dict  — {feature_name: value, ...}
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[FEATURE_EXTRACTION_PROMPT, image],
            config=types.GenerateContentConfig(temperature=0.1),
        )
        return self._parse_json(response.text)

    def extract_features_from_path(self, image_path: str | Path) -> dict:
        img = Image.open(str(image_path)).convert("RGB")
        return self.extract_features(img)

    # ── private ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Extract JSON from model response, tolerating markdown code fences."""
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find the first {...} block
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if m:
                return json.loads(m.group())
            return {}
