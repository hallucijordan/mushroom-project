"""
Judge A — Model-backed judge.

Pipeline:
  Image → Gemini feature extraction → MushroomPreprocessor → BaseLearner
         → verdict + confidence + reasoning narrative
"""

from google import genai
from google.genai import types
from PIL import Image

from src.agents.gemini_extractor import GeminiExtractor
from src.data.preprocessor import MushroomPreprocessor
from src.models.base_learner import BaseLearner


class ModelJudge:
    """
    Verdicts grounded in structured ML predictions.

    Parameters
    ----------
    api_key    : Gemini API key (for feature extraction).
    model_name : Gemini model for feature extraction.
    """

    NAME = "Judge A (Model-backed)"

    def __init__(
        self,
        api_key: str,
        model_name: str = GeminiExtractor.DEFAULT_MODEL,
    ):
        self.extractor = GeminiExtractor(api_key=api_key, model_name=model_name)
        self.preprocessor = MushroomPreprocessor.load()
        self.learner = BaseLearner.load()
        self._client = self.extractor.client  # reuse for narration
        self._model_name = model_name

    def judge(self, image: Image.Image) -> dict:
        """
        Analyse a mushroom image and return a structured verdict.

        Returns
        -------
        {
          "verdict":     "edible" | "poisonous",
          "confidence":  float (0–1),
          "p_edible":    float,
          "p_poisonous": float,
          "features":    dict,
          "narrative":   str,   # conversational explanation
        }
        """
        # Step 1 — extract features
        features = self.extractor.extract_features(image)

        # Step 2 — run classifier
        X = self.preprocessor.feature_dict_to_array(features)
        prediction = self.learner.predict_single(X)

        # Step 3 — generate a conversational narrative via Gemini
        narrative = self._narrate(features, prediction)

        return {
            "verdict":     prediction["label"],
            "confidence":  prediction["confidence"],
            "p_edible":    prediction["p_edible"],
            "p_poisonous": prediction["p_poisonous"],
            "features":    features,
            "narrative":   narrative,
        }

    # ── private ────────────────────────────────────────────────────────────

    def _narrate(self, features: dict, prediction: dict) -> str:
        verdict_str = prediction["label"].upper()
        confidence_pct = round(prediction["confidence"] * 100, 1)
        feature_summary = "\n".join(
            f"  • {k}: {v}" for k, v in features.items() if v is not None
        )
        prompt = f"""
You are Judge A, a mycology expert who uses a trained machine-learning classifier
to help assess mushroom safety.

You extracted these features from the mushroom photo:
{feature_summary}

The classifier predicts: {verdict_str} (confidence {confidence_pct}%)

Write 3-4 sentences in a conversational, judge-like tone:
1. Briefly mention 2-3 key features you observed.
2. State your verdict and the model's confidence.
3. Add one real-world safety note (keep it concise).

Do NOT use bullet points — write flowing sentences.
"""
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=200),
        )
        return response.text.strip()
