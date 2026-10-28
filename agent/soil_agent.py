"""
Step 5b - Soil advisory agent (smolagents CodeAgent).

Builds a CodeAgent that uses the soil tools (agent/soil_tools.py) and an LLM
served through Hugging Face Inference.

Usage (Colab):
    from agent.soil_agent import build_agent
    agent = build_agent(token=HF_TOKEN)
    agent.run("What is the soil pH around Kitale?")
"""
from smolagents import CodeAgent, InferenceClientModel

import re

import pandas as pd

from agent.soil_tools import INSERTED_TEXTS, THRESHOLDS, TOOLS

MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"

INSTRUCTIONS = """
You are a soil advisory assistant for agricultural planning in Kenya. You answer
questions from farmers, extension officers and county planners about topsoil
(0-30 cm) conditions and what they mean for management.

COVERAGE
- Mapped area: Kakamega and Trans Nzoia counties.
- Mapped properties: soil pH (water) and soil organic carbon (SOC, g/kg).
- For other places or properties (e.g. nitrogen, phosphorus, potassium), say
  they are not yet covered by the system.

WHICH TOOL
- Place name given -> call find_place first. Use matches[0] unless another match
  fits the user's wording better. If there are several matches, say which one
  you used (its name).
- A single farm or exact point (coordinates) -> get_soil_at_location.
- A town, village, ward, "around X" or "near X" -> summarize_area
  (radius_km=3 unless the user gives a radius). Town centres are often urban,
  so an area summary represents the surrounding farmland better.
- What a class or value means, or why it matters -> get_class_definitions
  (pass class_name, e.g. class_name="strongly acid", to get just that class).
- If a tool returns 'outside the mapped area', 'no data' or an error, say so plainly.

HOW TO WORK (about 2 steps)
- Step 1: call the tools you need and print their results.
- Step 2: read the printed results, then call final_answer() with the answer.
- For one class, call get_class_definitions(property_name=..., class_name=...)
  rather than searching the class list yourself.

WRITING THE FINAL ANSWER
- Write the answer as flowing plain-language sentences inside final_answer("...").
  Do not use a question-and-answer template (e.g. "Why it matters: ...") and do
  not repeat the user's question.
- No text field you insert (summary, advice_text, interpretation, explanation)
  ends with a full stop. Always add exactly one full stop and a space after
  inserting it, e.g. f"... {x['summary']}. {x['advice_text']}. Class ranges ...".
- Every tool result (point, area and class definitions) has "summary" and
  "advice_text" fields, so you can always use them the same way.
- For each property, include its "summary" WORD FOR WORD by inserting the variable,
  e.g. f"Soil pH around Kitale: {r['properties']['ph']['summary']}." The summary is
  a sentence fragment with no label and no final full stop, so put your own lead-in
  before it and end the sentence with a full stop after it. It already states
  value, class, confidence and area of applicability correctly.
- Do not retype or recalculate any number. Percentages are already 0-100.
- Always keep confidence and area of applicability in the answer (they are in
  the summary sentences, so do not drop or shorten them).

RECOMMENDATIONS AND EXPLANATIONS
- Recommendations come only from "advice_text" (point and area results) or the
  "interpretation" text (class definitions). Insert "advice_text" as a variable,
  e.g. f"Recommendations for pH: {r['properties']['ph']['advice_text']}". It is
  already a complete paragraph that covers every significant class and its share
  of the area, so do not pick items from the "advice" list or join them yourself.
- To explain WHY something matters, use the "explanation" text
  (or "dominant_class_explanation"). Do not add agronomic claims that are not in
  the tool results.
- For "why does it matter" questions, give BOTH the explanation and the
  interpretation, and state the interpretation's action explicitly in its own
  sentence (e.g. "Liming is advised for maize and beans."), not only implied.
- For management questions about a place (liming, fertility, should, what to do),
  cover BOTH pH and SOC: a paragraph for each, and insert the "advice_text" of
  BOTH properties after "Recommendations:".
- Do not give lime or fertilizer application rates: they are not in the system.
  If asked, say rates need a soil test and local extension guidance.
- Sources: "class_ranges_source" is the source of the class RANGES only. Cite it
  only next to a range or as "class ranges follow <source>". Never put it in a
  sentence about recommendations, advice or explanations; those come from the
  project's indicator table.

CHECKS
If a check rejects your answer, read its message and CHANGE the answer to fix that
exact problem. Never resubmit the same text.
Your final answer is checked automatically. It is rejected, and you must write it
again, if it contains a broken number (e.g. "16. 7"), a source named in a
sentence about recommendations or advice, "not specified", a double full stop
(".."), an inserted field not followed by a full stop, or - for a management question about a place - an answer that leaves out
soil organic carbon or pH.

FINAL ANSWER FORMAT
1. Direct answer to the question in one sentence (e.g. yes/no for "should...").
2. One short paragraph per property: its "summary" sentence plus one or two
   sentences of your own on what it means.
3. Recommendation, starting with "Recommendations:" and inserting "advice_text"
   (or the explicit interpretation for class questions); explanation if asked why.
4. A last sentence citing the class ranges source for each property covered, e.g.
   "Class ranges follow the USDA Soil Survey Manual (pH) and Hazelton & Murphy
   (2016) (SOC)." Always include it, using "class_ranges_source"; keep it out of
   the recommendation sentences.
5. Only the details that apply: the place used if a place was looked up and, if
   find_place returned several matches, a sentence naming the other matches
   (e.g. "Malava also matched: <name>."), depth 0-30 cm.
   Leave out anything that does not apply; never write "not specified".
Plain language for non-specialists, about 100-220 words. Say each thing once;
do not repeat sentences.
"""


# ------------------------------------------------------------------ final-answer checks
# Each check runs on the final answer before it is accepted. If one fails, the
# agent receives the message and must write the answer again (smolagents
# final_answer_checks). Instructions ask; checks enforce.

def _source_names():
    try:
        srcs = pd.read_csv(THRESHOLDS)["source"].dropna().unique()
    except Exception:
        return []
    # "USDA Soil Survey Manual (Soil ...)" -> "USDA Soil Survey Manual"
    return [str(x).split(" (")[0].strip() for x in srcs]


ADVICE_WORDS = re.compile(r"\b(recommend\w*|advice|advis\w*)\b", re.I)


def check_not_empty(final_answer, memory, agent=None):
    if not isinstance(final_answer, str) or len(final_answer.strip()) < 40:
        raise ValueError("The final answer must be a complete text answer.")
    return True


def check_no_broken_numbers(final_answer, memory, agent=None):
    bad = re.findall(r"\d\. \d", str(final_answer))
    if bad:
        raise ValueError(
            f"Broken number(s) {bad}: you retyped numbers. Insert the 'summary' and "
            "'advice_text' variables in an f-string instead of typing their text.")
    return True


def check_source_attribution(final_answer, memory, agent=None):
    for sentence in re.split(r"(?<=[.!?])\s+", str(final_answer)):
        for src in _source_names():
            if src and src.lower() in sentence.lower() and ADVICE_WORDS.search(sentence):
                raise ValueError(
                    f"'{src}' is the source of the class ranges only, but it appears in "
                    f"a sentence about recommendations: '{sentence.strip()[:120]}'. "
                    "Cite it only next to class ranges. If that sentence is really two "
                    "sentences run together, add the missing full stop between them.")
    return True


def check_no_placeholders(final_answer, memory, agent=None):
    if re.search(r"not specified", str(final_answer), re.I):
        raise ValueError("Remove 'not specified' lines; leave out details that do not apply.")
    return True


def check_no_double_full_stop(final_answer, memory, agent=None):
    if re.search(r"(?<!\.)\.\.(?!\.)", str(final_answer)):
        raise ValueError(
            "Double full stop '..' found. Inserted text fields have no final full "
            "stop; add exactly one after each.")
    return True


MANAGEMENT_WORDS = re.compile(
    r"\b(lime|liming|should|fertili[sz]\w*|manage\w*|what to do|recommend\w*)\b", re.I)
PLACE_WORDS = re.compile(r"\b(near|around|within|in|at)\b", re.I)


def check_both_properties(final_answer, memory, agent=None):
    """Management questions about a place must cover both pH and SOC."""
    task = str(getattr(agent, "task", "") or "")
    if not (MANAGEMENT_WORDS.search(task) and PLACE_WORDS.search(task)):
        return True
    text = str(final_answer).lower()
    missing = [n for n, keys in [("soil pH", ("ph",)),
                                 ("soil organic carbon", ("organic carbon", "soc"))]
               if not any(re.search(rf"\b{re.escape(k)}\b", text) for k in keys)]
    if missing:
        raise ValueError(
            f"This is a management question about a place, but the answer leaves out "
            f"{' and '.join(missing)}. Cover both pH and SOC and insert both "
            "'advice_text' values after 'Recommendations:'.")
    return True


def check_full_stop_after_fields(final_answer, memory, agent=None):
    """Each inserted text field must be followed by a full stop (or end the answer)."""
    text = str(final_answer)
    for field in INSERTED_TEXTS:
        start = 0
        while True:
            i = text.find(field, start)
            if i < 0:
                break
            after = text[i + len(field):i + len(field) + 1]
            if after not in (".", ""):
                raise ValueError(
                    f"Missing full stop after the inserted text ending '...{field[-40:]}'. "
                    f"It runs straight into '{text[i + len(field):i + len(field) + 25]}'. "
                    "Add a full stop and a space after every inserted field.")
            start = i + len(field)
    return True


# Order matters: format checks first, so the agent gets the most specific message.
FINAL_ANSWER_CHECKS = [check_not_empty, check_no_broken_numbers,
                       check_no_double_full_stop, check_full_stop_after_fields,
                       check_source_attribution, check_no_placeholders,
                       check_both_properties]


def build_agent(token=None, model_id=MODEL_ID, max_steps=8, verbosity_level=1,
                provider=None):
    """Create the soil advisory CodeAgent.

    Args:
        token: Hugging Face access token (e.g. from Colab secret HF_TOKEN).
        model_id: Hugging Face model id of the LLM.
        max_steps: maximum ReAct steps before the agent must answer.
        verbosity_level: 0 = quiet, 1 = show steps, 2 = show everything.
        provider: optional Hugging Face inference provider (None = automatic).
    """
    model = InferenceClientModel(model_id=model_id, token=token, provider=provider)
    return CodeAgent(
        tools=TOOLS,
        model=model,
        instructions=INSTRUCTIONS,
        final_answer_checks=FINAL_ANSWER_CHECKS,
        max_steps=max_steps,
        verbosity_level=verbosity_level,
        name="soil_advisor",
        description="Answers questions about mapped soil pH and organic carbon "
                    "in Kakamega and Trans Nzoia and gives management recommendations.",
    )
