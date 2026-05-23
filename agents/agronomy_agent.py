# agents/agronomy_agent.py
import vertexai
from vertexai.generative_models import GenerativeModel
import google.generativeai as genai
import os
import json

class AgronomyAgent:
    def __init__(self, model_name: str = "gemini-2.0-flash-001"):
        vertexai.init(
            project=os.getenv("GCP_PROJECT_ID"),
            location="europe-west1"
        )
        self.model = GenerativeModel(model_name)

    def analyze(self, yolo_metadata: dict, weather_data: dict) -> dict:
        prompt = f"""
You are an expert agronomist analyzing drone survey data.

## Drone Image Detections
{json.dumps(yolo_metadata, indent=2)}

## Weather Conditions at Time of Survey
{json.dumps(weather_data, indent=2)}

Based on this data, provide a structured agronomic assessment covering:

1. **Crop Health Assessment** — What do the detected objects suggest about crop status?
2. **Stress Indicators** — Any signs of water stress, disease, or pest pressure?
3. **Weather Impact** — How do current/recent conditions affect the findings?
4. **Recommended Actions** — Prioritized list of interventions (irrigation, spraying, etc.)
5. **Confidence Score** — Rate your assessment confidence 0-100 with justification.
6. **Follow-up Surveys** — What additional data would improve accuracy?

Respond in valid JSON only, no markdown.
"""
        response = self.model.generate_content(prompt)
        
        try:
            # Strip any accidental markdown fences
            text = response.text.strip().lstrip("```json").rstrip("```").strip()
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw_response": response.text, "parse_error": True}