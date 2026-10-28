---
title: Kenya Soil Advisor
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
  - inference-api
short_description: Prototype soil advisory agent for Kakamega and Trans Nzoia
---

# Kenya Soil Advisor (prototype)

Chat app that answers questions about soils and gives soil management recommendations.
Ask about soil pH and soil organic carbon in Kakamega and Trans Nzoia counties.

The app runs a smolagents CodeAgent (Qwen2.5-Coder-32B-Instruct via Hugging Face
Inference) whose tools read modelled soil maps and an indicator table. Answers report
values, classes, confidence and area of applicability, with recommendations taken only
from the indicator table and checked automatically before they are shown.

Sign in with your Hugging Face account to ask questions; each question runs on your
own inference credits.

Prototype on legacy soil data; not a substitute for a soil test or extension advice.
Source code: https://github.com/japhethkimeu/kenya-soil-dss
