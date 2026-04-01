"""
Mushroom Safety Judge — Gradio UI

Two judges examine the uploaded mushroom photo and deliver verdicts through
a conversation-style interface:

  Judge A (Model-backed) — Gemini extracts features → ML classifier decides
  Judge B (Direct Vision) — Gemini looks at the photo and decides directly

Run:
    conda activate mushroom-project
    python app.py
"""

import gradio as gr
from PIL import Image

from config.api_keys import GEMINI_API_KEY
from src.judges.model_judge import ModelJudge
from src.judges.direct_judge import DirectJudge
from src.models.base_learner import BaseLearner

# ── initialise judges ──────────────────────────────────────────────────────

def _load_judges():
    if not BaseLearner.is_trained():
        raise RuntimeError(
            "No trained model found. Run `python train.py` first."
        )
    judge_a = ModelJudge(api_key=GEMINI_API_KEY)
    judge_b = DirectJudge(api_key=GEMINI_API_KEY)
    return judge_a, judge_b


# ── verdict helpers ────────────────────────────────────────────────────────

VERDICT_EMOJI = {"edible": "✅", "poisonous": "☠️"}
VERDICT_COLOR = {"edible": "#2ecc71", "poisonous": "#e74c3c"}


def _verdict_html(label: str, confidence: float, judge_name: str) -> str:
    emoji = VERDICT_EMOJI.get(label, "❓")
    color = VERDICT_COLOR.get(label, "#888")
    pct = round(confidence * 100, 1)
    return (
        f'<span style="color:{color};font-weight:bold;font-size:1.1em">'
        f'{emoji} {label.upper()}</span> '
        f'<span style="color:#aaa">({pct}% confidence — {judge_name})</span>'
    )


def _final_verdict_html(a_label, a_conf, b_label, b_conf) -> str:
    """Weighted consensus: model judge gets slightly more weight."""
    p_poison_a = 1 - a_conf if a_label == "edible" else a_conf
    p_poison_b = 1 - b_conf if b_label == "edible" else b_conf
    combined = 0.6 * p_poison_a + 0.4 * p_poison_b
    final = "poisonous" if combined >= 0.5 else "edible"
    emoji = VERDICT_EMOJI[final]
    color = VERDICT_COLOR[final]
    return (
        f'<div style="text-align:center;padding:12px;border-radius:8px;'
        f'background:{color}22;border:2px solid {color}">'
        f'<span style="font-size:2em">{emoji}</span><br/>'
        f'<span style="color:{color};font-size:1.4em;font-weight:bold">'
        f'FINAL VERDICT: {final.upper()}</span>'
        f'</div>'
    )


# ── main inference function ────────────────────────────────────────────────

def analyse(image: Image.Image, judge_a: ModelJudge, judge_b: DirectJudge):
    """Called by Gradio when the user submits an image."""
    if image is None:
        return [], [], "<p>Please upload an image first.</p>"

    chat_a = []
    chat_b = []

    # ── Judge A ────────────────────────────────────────────────────────────
    try:
        result_a = judge_a.judge(image)
        header_a = _verdict_html(result_a["verdict"], result_a["confidence"], ModelJudge.NAME)

        # Build feature summary for display
        feat_lines = [
            f"{k}: {v}" for k, v in result_a["features"].items() if v is not None
        ]
        feature_msg = "**Extracted features:**\n" + "\n".join(
            f"- {line}" for line in feat_lines[:10]
        )
        if len(feat_lines) > 10:
            feature_msg += f"\n- … ({len(feat_lines) - 10} more)"

        chat_a = [
            {"role": "assistant", "content": (
                f"I've analysed the photograph and extracted the mushroom's structural features.\n\n"
                f"{feature_msg}"
            )},
            {"role": "assistant", "content": (
                f"I then ran these features through the trained classifier.\n\n"
                f"**My verdict:** {header_a}\n\n"
                f"{result_a['narrative']}"
            )},
        ]
    except Exception as e:
        chat_a = [{"role": "assistant", "content": f"Error: {e}"}]
        result_a = {"verdict": "poisonous", "confidence": 0.5}

    # ── Judge B ────────────────────────────────────────────────────────────
    try:
        result_b = judge_b.judge(image)
        header_b = _verdict_html(result_b["verdict"], result_b["confidence"], DirectJudge.NAME)

        chat_b = [
            {"role": "assistant", "content": "I'm examining the photograph directly, relying on visual inspection alone."},
            {"role": "assistant", "content": (
                f"**My verdict:** {header_b}\n\n"
                f"{result_b['narrative']}"
            )},
        ]
    except Exception as e:
        chat_b = [{"role": "assistant", "content": f"Error: {e}"}]
        result_b = {"verdict": "poisonous", "confidence": 0.5}

    final_html = _final_verdict_html(
        result_a["verdict"], result_a["confidence"],
        result_b["verdict"], result_b["confidence"],
    )

    return chat_a, chat_b, final_html


# ── Gradio layout ──────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    judge_a, judge_b = _load_judges()

    with gr.Blocks(title="Mushroom Safety Judge") as demo:
        gr.Markdown(
            "# 🍄 Mushroom Safety Judge\n"
            "Upload a mushroom photo. Two independent judges will examine it and "
            "deliver their verdicts through a conversation-style interface.\n\n"
            "> ⚠️ **This is a research demo — never eat wild mushrooms based solely on AI advice.**"
        )

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Upload Mushroom Photo",
                    height=320,
                )
                submit_btn = gr.Button("Analyse", variant="primary", size="lg")

            with gr.Column(scale=2):
                final_verdict = gr.HTML(
                    label="Final Verdict",
                    value='<div style="text-align:center;color:#888;padding:20px">'
                          'Upload an image and click Analyse</div>',
                )

        gr.Markdown("---")

        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    f"### 🤖 {ModelJudge.NAME}\n"
                    "*Uses Gemini to extract 20 structured features, then runs them through a "
                    "trained Random Forest classifier.*"
                )
                chatbox_a = gr.Chatbot(
                    label=ModelJudge.NAME,
                    height=400,
                )

            with gr.Column():
                gr.Markdown(
                    f"### 👁️ {DirectJudge.NAME}\n"
                    "*Asks Gemini to assess edibility purely from the photograph, "
                    "with no ML model involved.*"
                )
                chatbox_b = gr.Chatbot(
                    label=DirectJudge.NAME,
                    height=400,
                )

        submit_btn.click(
            fn=lambda img: analyse(img, judge_a, judge_b),
            inputs=[image_input],
            outputs=[chatbox_a, chatbox_b, final_verdict],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False, theme=gr.themes.Soft())
