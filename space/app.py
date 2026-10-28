"""
Kenya Soil DSS - Soil Advisor (Gradio app for a Hugging Face Space)

A chat app around the smolagents soil advisory agent (agent/soil_agent.py).
Each question runs a fresh agent, so colleagues never share conversation state.

Needs in the Space:
  agent/, config/indicator_thresholds.csv,
  outputs/maps/<prop>_0_30cm.tif, outputs/indicators/<prop>_classes.tif
  README.md with `hf_oauth: true` and the `inference-api` scope

Visitors sign in with their Hugging Face account; each question runs on the
visitor's own inference credits (their OAuth token), not the owner's.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("SOIL_DSS_ROOT", str(ROOT))   # tools read data relative to this
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from agent.soil_agent import build_agent  # noqa: E402

TITLE = "Kenya Soil Advisor (prototype)"
INTRO = """
Ask about **soil pH and soil organic carbon** in **Kakamega and Trans Nzoia**
counties - for a place ("around Kitale", "within 5 km of Malava") or what a class means.
Answers come only from the project's soil maps and indicator table, with confidence and
how well the area is represented by the sampled sites.
"""
NOTICE = """
**Sign in with Hugging Face** (button above) to ask questions. Questions use your own
Hugging Face inference credits (the free monthly allowance is enough for testing).

**Prototype - for testing only.** Maps are modelled from legacy soil data. Not a
substitute for a soil test or local extension advice; no lime or fertilizer rates are
given.
Contact: Japheth Kimeu (FAO Kenya) - Japheth.Kimeu@fao.org
"""
EXAMPLES = [
    "What is the soil pH around Kitale, and does it need liming?",
    "Should a farmer near Kakamega town apply lime? Also comment on organic carbon.",
    "Summarize soil conditions within 5 km of Malava.",
    "What pH values count as strongly acid, and why does it matter for maize?",
]


def answer(message, history, oauth_token: gr.OAuthToken | None = None):
    """Run one question through a fresh agent and return only the final answer.

    Gradio fills oauth_token from the visitor's "Sign in with Hugging Face" login.
    """
    message = (message or "").strip()
    if not message:
        return "Please type a question."
    if oauth_token is None or not getattr(oauth_token, "token", None):
        return ("Please sign in with your Hugging Face account (button at the top of "
                "the page) to ask questions.")
    try:
        agent = build_agent(token=oauth_token.token, verbosity_level=0)
        result = agent.run(message)
    except Exception as e:  # keep the app usable whatever goes wrong
        msg = str(e)
        if "402" in msg or "Payment" in msg:
            return ("Your Hugging Face inference credits for this month are used up. "
                    "Please try again next month or add credits to your account.")
        if "401" in msg or "403" in msg:
            return ("Your Hugging Face sign-in has expired or lacks inference access. "
                    "Please sign out and sign in again.")
        return ("Sorry, the advisor could not answer this question. "
                "Please rephrase it or try again in a moment.")
    text = str(result).strip()
    if not text or text.lower() == "none":
        return "Sorry, no answer was produced. Please rephrase the question."
    return text


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}")
    gr.Markdown(INTRO)
    gr.LoginButton()
    gr.ChatInterface(
        fn=answer,
        examples=EXAMPLES,
        cache_examples=False,
        chatbot=gr.Chatbot(height=480,
                           placeholder="Answers take about 10-30 seconds."),
    )
    gr.Markdown(NOTICE)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch()
