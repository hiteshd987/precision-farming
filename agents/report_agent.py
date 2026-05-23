# agents/report_agent.py
import vertexai
from vertexai.generative_models import GenerativeModel
import google.generativeai as genai
import os
import json
from datetime import datetime
from pathlib import Path

class ReportAgent:
    def __init__(self, model_name: str = "gemini-2.0-flash-001"):
        vertexai.init(
            project=os.getenv("GCP_PROJECT_ID"),
            location="europe-west1"
        )
        self.model = GenerativeModel(model_name)

    def verify_and_compile(self, yolo_data, weather_data, agronomy_analysis) -> dict:
        """Second Gemini pass — cross-checks and verifies the first analysis."""
        prompt = f"""
You are a senior agronomist verifying a junior analyst's report.

## Original Data
YOLO Detections: {json.dumps(yolo_data)}
Weather: {json.dumps(weather_data)}

## Analysis to Verify
{json.dumps(agronomy_analysis, indent=2)}

Review the analysis for:
- Factual consistency with the raw data
- Agronomic accuracy of recommendations
- Any contradictions or overconfident claims

Output a final verified report as JSON with fields:
- verified (bool)
- corrections (list of any changes made)
- final_assessment (the approved analysis)
- field_health_grade ("A: Excellent", "B: Minor Weeds", "C: Moderate Stress", "F: Severe Intervention Required")
- generated_at (ISO timestamp)

JSON only, no markdown.
"""
        response = self.model.generate_content(prompt)
        
        try:
            text = response.text.strip().lstrip("```json").rstrip("```").strip()
            result = json.loads(text)
            result["generated_at"] = datetime.utcnow().isoformat()
            return result
        except Exception:
            return {"raw": response.text, "generated_at": datetime.utcnow().isoformat()}

    def save(self, report: dict, image_name: str):
        out_path = Path("reports") / f"{Path(image_name).stem}_report.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✅ Report saved: {out_path}")
        return out_path