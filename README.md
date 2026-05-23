# 🚁 AgroAgent — Precision Farming AI & Spatial Analytics

An end-to-end AI pipeline for precision agriculture. AgroAgent processes drone imagery through a custom YOLO weed-detection model, fetches real-time weather data, and passes the combined evidence through a two-pass Gemini agent chain to produce verified, graded agronomic reports all visualised in an interactive Streamlit dashboard with a built-in Farm AI Assistant.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
  - [Phase 1 — ETL Backend](#phase-1--etl-backend)
  - [Phase 2 — Streamlit Dashboard](#phase-2--streamlit-dashboard)
- [Agent Pipeline Deep Dive](#agent-pipeline-deep-dive)
- [Dashboard Features](#dashboard-features)
- [Model Training (Optional)](#model-training-optional)
- [Configuration Reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

AgroAgent is designed as a decoupled two-phase system:

**Phase 1 (ETL)** — `main.py` orchestrates the full inference pipeline: drone images are ingested, run through a fine-tuned YOLO model for weed detection, matched with live weather data via the OpenWeatherMap API, and processed by two sequential Gemini agents (a junior analyst and a senior verifier). Each processed image produces a structured JSON report appended to a lightweight `.jsonl` database.

**Phase 2 (Visualisation)** — `visualize/dashboard.py` reads the `.jsonl` database and renders an interactive Streamlit dashboard featuring spatial anomaly maps, health-grade distributions, detection timelines, and a conversational Farm AI Assistant backed by a tool-calling Gemini agent.

---

## Key Features

**Computer Vision (YOLO)**
- Custom-trained YOLOv8 model (`aerial_weeds.pt`) identifies weed versus crop zones in high-resolution drone imagery
- Automatically parses EXIF/GPS metadata from DJI images to geo-reference every detection
- Supports `.jpg`, `.jpeg`, `.png`, `.tiff`, `.tif`, `.dng` (and uppercase variants)

**Multi-Agent Agronomy AI (Gemini on Vertex AI)**
- *Junior Analyst Agent* (`AgronomyAgent`) — assesses crop health, stress indicators, weather impact, and recommends interventions with a self-reported confidence score
- *Senior Verifier Agent* (`ReportAgent`) — independently cross-checks the junior analysis for factual consistency and agronomic accuracy, assigns a final health grade (A / B / C / F), and records any corrections

**Interactive Spatial Dashboard**
- Plotly Mapbox scatter map showing exact stress locations colour-coded by status
- Toggle-able density heatmap for infestation hotspots
- Bar chart of detected issue frequencies
- Time-series bar chart of daily detection trends
- Donut chart of health-grade distribution (A / B / C / F)
- Raw report table with expandable expander view
- Sidebar filters by field status and health grade

**Farm AI Assistant**
- Conversational Gemini agent embedded in the dashboard
- Parallel tool-calling: queries live OpenWeatherMap weather AND the internal `.jsonl` database simultaneously
- Pre-built FAQ buttons for the most common agronomic queries
- Provides decisive, data-driven herbicide / intervention recommendations

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PHASE 1 — ETL BACKEND                        │
│                                                                      │
│  Drone Images (DJI)                                                  │
│        │                                                             │
│        ▼                                                             │
│  ┌─────────────────┐     GPS + Timestamp                            │
│  │ DroneImageExtractor │──────────────────► WeatherAgent            │
│  │   (YOLOv8 YOLO)  │                      (OpenWeatherMap API)     │
│  └─────────────────┘                              │                 │
│        │ detections + metadata                    │ weather data    │
│        └────────────────┬─────────────────────────┘                 │
│                         ▼                                            │
│                  ┌─────────────┐                                     │
│                  │ AgronomyAgent│  ← Gemini 2.x (Vertex AI)         │
│                  │ (Junior AI) │    "Assess crop health"             │
│                  └─────────────┘                                     │
│                         │ JSON analysis                              │
│                         ▼                                            │
│                  ┌─────────────┐                                     │
│                  │ ReportAgent │  ← Gemini 2.x (Vertex AI)          │
│                  │ (Senior AI) │    "Verify and grade"               │
│                  └─────────────┘                                     │
│                         │ verified report + grade                    │
│                         ▼                                            │
│               reports/argo_report.jsonl                              │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 2 — STREAMLIT DASHBOARD                     │
│                                                                      │
│   dashboard.py ──► Plotly Maps & Charts ──► Farm AI Assistant       │
│                                              (Gemini + Tool Calls)  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
agro-agent/
│
├── main.py                        # Backend orchestrator
│
├── agents/
│   ├── yolo_extractor.py          # YOLO inference + EXIF/GPS parsing
│   ├── weather_agent.py           # OpenWeatherMap API client
│   ├── agronomy_agent.py          # Junior analyst Gemini agent
│   ├── report_agent.py            # Senior verifier Gemini agent
│   └── model_selector.py          # Interactive Gemini model picker (CLI)
│
├── visualize/
│   └── dashboard.py               # Streamlit dashboard + Farm AI Assistant
│
├── datasets/
│   └── weed_crop_aerial/          # YOLO training data and validation sets
│       └── data.yaml
│
├── models/
│   └── aerial_weeds.pt            # Fine-tuned YOLOv8 weights
│
├── reports/
│   └── argo_report.jsonl          # Append-only JSONL report database
│
├── .env                           # API keys (never commit this)
└── requirements.txt               # Python dependencies
```

---

## Prerequisites

- Python 3.9 or higher
- A Google Cloud Platform (GCP) project with **Vertex AI API** enabled
- GCP Application Default Credentials configured (`gcloud auth application-default login`)
- An **OpenWeatherMap** API key (free tier is sufficient)
- DJI drone images with EXIF/GPS metadata embedded (JPEG, PNG, TIFF, or DNG)

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/precision-farming.git
cd precision-farming

#

# 2. (Recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**Core dependencies include:** `ultralytics`, `streamlit`, `plotly`, `pandas`, `google-cloud-aiplatform`, `vertexai`, `google-generativeai`, `requests`, `Pillow`, `python-dotenv`

---

## Environment Setup

Create a `.env` file in the project root:

```env
WEATHER_API_KEY="your_openweathermap_api_key_here"
GCP_PROJECT_ID="your_google_cloud_project_id_here"
```

Then authenticate with GCP:

```bash
gcloud auth application-default login
```

> **Security note:** `.env` contains credentials. Add it to `.gitignore` and never commit it.

---

## Usage

AgroAgent operates in two independent phases. Run Phase 1 first to populate the database, then Phase 2 to explore and query the results.

### Phase 1 — ETL Backend

```bash
python main.py
```

On startup you will be prompted to select a Gemini model:

```
🤖 Available Gemini Models on Vertex AI:
────────────────────────────────────────
  [1] gemini-2.5-flash          ← recommended
  [2] gemini-2.5-flash-lite
  [3] gemini-2.0-flash-001
  ...
────────────────────────────────────────
Pick a model number (default=1):
```

**Batch mode (default)** — processes every supported image in `DRONE_IMAGES_DIR`:

```bash
python main.py
```

**Batch mode with limit** — processes only the first N images (useful for testing):

```bash
python main.py 10
```

**Single image mode** — pass a direct file path as the argument:

```bash
python main.py /path/to/your/image.jpg
```

Each successfully processed image appends one JSON line to `reports/argo_report.jsonl`. The pipeline is resumable — re-running appends new results without overwriting existing ones. To start fresh, either delete the file or uncomment the wipe line in `main.py`:

```python
# open(out_path, 'w').close()
```

**Pipeline console output per image:**

```
============================================================
 STARTING PIPELINE FOR: field_zone_042.jpg
============================================================

📍 STEP 1: Running YOLO with aerial weeds & Metadata Extraction...
   ↳ 📸 Detections: {'weed': 12, 'crop': 87}
   ↳ 🌍 GPS Found: lat 51.3456, lon 9.8765
   ↳ ⏰ Timestamp: 2024:06:12 09:34:21

📍 STEP 2: Fetching Historical Weather Data...
   ↳  Conditions: light rain
   ↳  Temp: 14.2°C | 🌧️ Rain: 0.4mm

📍 STEP 3: Gemini Agent 1 (gemini-2.5-flash) - Agronomy Analysis...
   ↳  Generation Status: ✅ Success
   ↳ AI Confidence Score: 82

📍 STEP 4: Gemini Agent 2 (gemini-2.5-flash) - Verification...
   ↳  Verified: True
   ↳  Final Grade: B

📍 STEP 5: Saving Final Report...
```

### Phase 2 — Streamlit Dashboard

Once `argo_report.jsonl` contains at least one record:

```bash
streamlit run visualize/dashboard.py
```

Your browser will open automatically at `http://localhost:8501`.

---

## Agent Pipeline Deep Dive

### Step 1 — `DroneImageExtractor` (YOLO + EXIF)

Loads the fine-tuned `aerial_weeds.pt` model and runs inference on the drone image. For each detected bounding box it records the class name, confidence score, and pixel coordinates. In parallel, EXIF metadata is parsed to extract GPS coordinates (latitude/longitude) and the image capture timestamp. These are returned as a single structured dictionary passed downstream.

### Step 2 — `WeatherAgent` (OpenWeatherMap)

Uses the GPS coordinates extracted in Step 1 to fetch current weather conditions from the OpenWeatherMap API. Returns temperature, humidity, wind speed, precipitation, and a human-readable condition string. Falls back gracefully when no GPS data is present.

### Step 3 — `AgronomyAgent` — Junior Analyst (Gemini via Vertex AI)

Receives the full YOLO detection dictionary and the weather data. Constructs a structured prompt instructing Gemini to act as an expert agronomist and return a JSON assessment covering: crop health, stress indicators, weather impact, recommended interventions, a confidence score (0–100), and suggested follow-up surveys. The raw Gemini response is stripped of any accidental markdown fences and parsed into a Python dict.

### Step 4 — `ReportAgent` — Senior Verifier (Gemini via Vertex AI)

Receives the original raw data *and* the junior analysis. A second, independent Gemini call is made with a verification prompt. The model checks for factual consistency, agronomic accuracy, and contradictions. It outputs a final report with a boolean `verified` field, a list of any corrections made, the approved `final_assessment`, a letter-grade (`A` through `F`), and a UTC timestamp.

### Step 5 — Save

The final report is bundled with the source image name, GPS, timestamp, detection summary, and weather, then appended as a single JSON line to `reports/argo_report.jsonl`.

---

## Dashboard Features

### Spatial Analysis Map

Displays all surveyed zones as colour-coded markers on an OpenStreetMap base layer (green = Healthy, red = Weed). Hover over any marker to see the source image name, health grade, weather conditions, and timestamp. Toggle the **Show Infestation Heatmap** switch to replace the scatter plot with a density heatmap of weed zones only.

### Categorical & Temporal Charts

- Horizontal bar chart showing detection frequency by status type
- Time-series stacked bar chart of daily detection trends grouped by status
- Donut chart showing the proportion of A / B / C / F graded fields

### Sidebar Filters

Filter all charts and the map simultaneously by **Field Status** (e.g., `weed`, `Healthy`) and **Health Grade** (A / B / C / F).

### Top Metrics

Four headline KPIs updated live with the current filter selection: total surveys, acceptable health count (A/B), action-required count (C/F), and number of AI-verified reports.

### Farm AI Assistant

A stateful chat interface backed by a tool-calling Gemini agent. The agent has access to two tools:

- `get_live_weather(lat, lon)` — fetches current conditions from OpenWeatherMap for any field coordinate
- `get_field_status(status)` — queries the local `.jsonl` database for zone counts and top GPS locations matching a given status

The agent calls both tools in parallel when needed and synthesises the results into a definitive, actionable recommendation. Pre-built FAQ buttons are provided for the three most common queries:

- *"Where are the worst weed zones?"*
- *"Safe to spray today?"*
- *"Conditions for weed growth?"*

---

## Model Training (Optional)

To fine-tune the weed detection model on a new aerial dataset:

1. Format your dataset in standard YOLO format (images + labels with `data.yaml`)
2. Place it under `datasets/weed_crop_aerial/`
3. Run the training script:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(
    data="datasets/weed_crop_aerial/data.yaml",
    epochs=20,
    imgsz=640,       # Balanced for small-object detection in aerial imagery
    device="mps",    # Use "cuda" for NVIDIA GPUs, "cpu" as fallback
    batch=16
)
```

The trained weights will be saved to `runs/detect/train/weights/best.pt`. Copy this file to `models/aerial_weeds.pt` to use it in the pipeline.

> `imgsz=640` is the recommended resolution for aerial weed detection — large enough to resolve small plants without exceeding GPU memory on typical hardware.

---

## Configuration Reference

| Variable | File | Description |
|---|---|---|
| `WEATHER_API_KEY` | `.env` | OpenWeatherMap API key |
| `GCP_PROJECT_ID` | `.env` | GCP project ID with Vertex AI enabled |
| `DRONE_IMAGES_DIR` | `main.py` | Absolute path to your drone image folder |
| `SUPPORTED_EXTENSIONS` | `main.py` | File extensions to include in batch mode |
| `model_path` | `yolo_extractor.py` | Path to the YOLO weights file |
| `location` | `agronomy_agent.py` / `report_agent.py` | Vertex AI region (default: `europe-west1`) |

---

## Troubleshooting

**`⚠️ No valid data found in reports/argo_report.jsonl`**
Run Phase 1 (`python main.py`) at least once before launching the dashboard.

**`No images found in DRONE_IMAGES_DIR`**
Update the `DRONE_IMAGES_DIR` path in `main.py` to point at your actual images folder.

**Gemini returns `parse_error: True`**
Occasionally the model returns markdown-fenced JSON. The agents strip common fences automatically. If it persists, try a different model from the selector — `gemini-2.5-flash` is the most reliable.

**GPS shows `None` for all images**
The EXIF parser did not find embedded GPS data. The dashboard will inject mock GPS coordinates for visualisation. Ensure your drone is set to embed GPS in image metadata before the next survey.

**`DefaultCredentialsError` from Vertex AI**
Run `gcloud auth application-default login` and verify that `GCP_PROJECT_ID` in `.env` matches your active GCP project.

**Weather API returns `401 Unauthorized`**
Check that `WEATHER_API_KEY` in `.env` is correct and that your OpenWeatherMap account is active (new keys can take up to 2 hours to activate).