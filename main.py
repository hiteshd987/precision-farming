# main.py
import os
import sys
import json
from agents.model_selector import pick_model
from dotenv import load_dotenv
from pathlib import Path

from agents.yolo_extractor import DroneImageExtractor
from agents.weather_agent import WeatherAgent
from agents.agronomy_agent import AgronomyAgent
from agents.report_agent import ReportAgent

load_dotenv()

# ── Point directly at your DJI images folder ──────────────────────────────────
DRONE_IMAGES_DIR = Path("/Users/hitesh/Documents/try_precision/datasets/drone_images")
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".dng", ".JPG", ".JPEG", ".PNG"}

def run_pipeline(image_path: Path, model_name: str):
    print(f"\n{'='*60}")
    print(f" STARTING PIPELINE FOR: {image_path.name}")
    print(f"{'='*60}\n")

    # ── STEP 1: Vision ──────────────────────────────────────────────
    print("📍 STEP 1: Running YOLO with aerial weeds & Metadata Extraction...")
    extractor = DroneImageExtractor("models/aerial_weeds.pt")
    yolo_data = extractor.extract(str(image_path))
    
    # Exposing the variables:
    print(f"   ↳ 📸 Detections: {yolo_data['detection_summary']}")
    print(f"   ↳ 🌍 GPS Found: lat {yolo_data['gps']['lat']}, lon {yolo_data['gps']['lon']}")
    print(f"   ↳ ⏰ Timestamp: {yolo_data['timestamp']}\n")

    # ── STEP 2: Weather ─────────────────────────────────────────────
    print("📍 STEP 2: Fetching Historical Weather Data...")
    weather_agent = WeatherAgent()
    weather = weather_agent.get_historical(
        lat=yolo_data["gps"]["lat"],
        lon=yolo_data["gps"]["lon"],
        timestamp=yolo_data["timestamp"]
    )
    
    # Exposing the variables:
    print(f"   ↳  Conditions: {weather.get('conditions', 'unavailable')}")
    print(f"   ↳  Temp: {weather.get('temperature_c', 'N/A')}°C | 🌧️ Rain: {weather.get('precipitation_mm', 'N/A')}mm\n")

    # ── STEP 3: Junior AI Pass ──────────────────────────────────────
    print(f"📍 STEP 3: Gemini Agent 1 ({model_name}) - Agronomy Analysis...")
    agronomy_agent = AgronomyAgent(model_name)
    analysis = agronomy_agent.analyze(yolo_data, weather)
    
    # Exposing the variables:
    status = "⚠️ Failed JSON Parse" if analysis.get("parse_error") else "✅ Success"
    print(f"   ↳  Generation Status: {status}")
    print(f"   ↳ AI Confidence Score: {analysis.get('Confidence Score', 'N/A')}\n")

    # ── STEP 4: Senior AI Pass ──────────────────────────────────────
    print(f"📍 STEP 4: Gemini Agent 2 ({model_name}) - Verification...")
    report_agent = ReportAgent(model_name)
    final_report = report_agent.verify_and_compile(yolo_data, weather, analysis)
    
    # Exposing the variables:
    print(f"   ↳  Verified: {final_report.get('verified', 'N/A')}")
    print(f"   ↳  Final Grade: {final_report.get('report_grade', 'N/A')}")
    
    corrections = final_report.get("corrections", [])
    if corrections:
        print(f"   ↳ ⚠️ Boss Corrections Made: {len(corrections)}")
        for i, correction in enumerate(corrections, 1):
            print(f"        {i}. {correction}")
    print("\n")

    # ── STEP 5: Save ────────────────────────────────────────────────
    print("📍 STEP 5: Saving Final Report...")
    
    # BUNDLE ALL DATA TOGETHER BEFORE SAVING
    final_report["source_image"] = image_path.name 
    final_report["timestamp"] = yolo_data["timestamp"]
    final_report["gps"] = yolo_data["gps"]
    final_report["detection_summary"] = yolo_data["detection_summary"]
    final_report["weather"] = weather

    # If you still have the report_agent.save() line, you can leave it, 
    # but the critical part is returning the fully bundled final_report
    return final_report

if __name__ == "__main__":
    selected_model = pick_model()

    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        # Single image mode
        single_report = run_pipeline(Path(sys.argv[1]), selected_model)
        
        # Open in "append" mode ("a") to safely add to the bottom of the file
        out_path = Path("reports/argo_report.jsonl")
        with open(out_path, "a") as f:
            f.write(json.dumps(single_report) + "\n")
            
    else:
        # Batch mode
        limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
        images = sorted([f for f in DRONE_IMAGES_DIR.iterdir() if f.suffix in SUPPORTED_EXTENSIONS])

        if limit:
            images = images[:limit]
            
        if not images:
            print(f"⚠️  No images found in {DRONE_IMAGES_DIR}")
        else:
            success, failed = 0, []
            
            out_path = Path("reports/argo_report.jsonl")
            
            # Optional: If you want to wipe the old report at the start of a brand new run, uncomment the next line:
            # open(out_path, 'w').close() 

            for img in images:
                try:
                    # 1. Run the pipeline
                    report_data = run_pipeline(img, selected_model)
                    
                    # 2. Instantly append the result to disk and add a newline
                    with open(out_path, "a") as f:
                        f.write(json.dumps(report_data) + "\n")
                        
                    success += 1
                except Exception as e:
                    print(f" Failed: {img.name} — {e}")
                    failed.append(img.name)

            print(f"\n{'─'*50}")
            print(f"📊 Pipeline Summary")
            print(f"   ✅ Processed: {success}/{len(images)}")
            print(f"   📁 MASTER REPORT APPENDED TO: {out_path}")