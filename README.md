# Kenya Soil DSS – Soil Advisor (prototype)

[![Open the Soil Advisor](https://img.shields.io/badge/%F0%9F%A4%97%20Open%20the%20app-Kenya%20Soil%20Advisor-yellow)](https://huggingface.co/spaces/kimeujapheth/kenya-soil-advisor)

A prototype decision-support system that maps soil properties from soil data and
serves them through an **AI agent** that answers questions and gives management
recommendations, as a basis for agricultural planning in Kenya.

**Try it:** [Kenya Soil Advisor on Hugging Face](https://huggingface.co/spaces/kimeujapheth/kenya-soil-advisor)
– sign in with a free Hugging Face account and ask, for example,
*"Should a farmer near Kakamega town apply lime?"* or
*"Summarize soil conditions within 5 km of Malava."*

> **Prototype.** Current maps cover **Kakamega and Trans Nzoia** counties
> (soil pH and soil organic carbon, 0–30 cm) and are modelled from legacy soil data.
> Not a substitute for a soil test or local extension advice; no lime or fertilizer
> rates are given.

## How it works

```
soil data ─▶ covariates (GEE) ─▶ Quantile Regression Forest ─▶ prediction + uncertainty maps
                                                                     │
                                         indicator classes ◀─────────┘
                                                │
             question ─▶ smolagents CodeAgent ─▶ tools (maps + classes) ─▶ checked answer
```

| Step | What | File |
|---|---|---|
| 1 | Point table: topsoil pH and SOC per site | `scripts/01_build_point_table.py` |
| 2 | Covariates: terrain, climate, Sentinel-2 (100 m) | `scripts/02_covariates_gee.js` (GEE Code Editor) |
| 3 | Quantile Regression Forest, spatial CV, prediction (q05–q95), Area of Applicability | `scripts/03_model_predict.py` |
| 4 | Indicator classes and class confidence | `scripts/04_indicators.py`, `config/indicator_thresholds.csv` |
| 5 | Agent tools and smolagents CodeAgent with final-answer checks | `agent/soil_tools.py`, `agent/soil_agent.py` |
| 6 | Gradio app (Hugging Face Space, sign-in with Hugging Face) | `space/app.py`, `space/deploy_space.py` |

Notebooks: `notebooks/Quantile_Regression_Forest.ipynb` (steps 1, 3, 4) and
`notebooks/SoilAgent.ipynb` (agent, tests, deployment), written for Google Colab.

## The agent

- **Framework:** [smolagents](https://github.com/huggingface/smolagents) `CodeAgent`,
  model `Qwen/Qwen2.5-Coder-32B-Instruct` via Hugging Face Inference.
- **Tools:** `find_place` (OpenStreetMap), `get_soil_at_location`, `summarize_area`,
  `get_class_definitions`. Tools return ready-made summaries and advice, so numbers
  and recommendations come from the maps and the indicator table, not from the model.
- **Grounding:** every answer reports value, class, confidence and how well the area is
  represented by the sampled sites (Area of Applicability); recommendations come only
  from `config/indicator_thresholds.csv`.
- **Final-answer checks:** code checks reject answers with retyped numbers, sources
  attributed to recommendations, broken punctuation, or (for management questions)
  a missing property; the agent then rewrites its answer.

## Running it

```bash
pip install -r requirements.txt smolagents
python scripts/01_build_point_table.py      # needs the source workbooks in data/raw/
python scripts/03_model_predict.py          # needs covariates in data/covariates/
python scripts/04_indicators.py
```

Then in Python:

```python
from agent.soil_agent import build_agent
agent = build_agent(token="<your Hugging Face token>")
agent.run("What is the soil pH around Kitale, and does it need liming?")
```

Deploy the app: `python space/deploy_space.py --space <hf-username>/kenya-soil-advisor`
(needs a Hugging Face write token in `HF_WRITE_TOKEN`).

## Data

Source soil data and covariates are not included in this repository. The app uses
only modelled prediction layers (no point observations).

## Status and roadmap

- Scale to national soil testing data and more counties and properties
- Crop suitability and nutrient indicators
- Expose the tools through MCP

## Author

**Japheth Kimeu** · [japhethkimeu.github.io](https://japhethkimeu.github.io)

## Licence

Code released under the [MIT Licence](LICENSE).
