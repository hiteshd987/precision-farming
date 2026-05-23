# agents/yolo_extractor.py
from ultralytics import YOLO
from PIL.ExifTags import TAGS, GPSTAGS
from PIL import Image
import numpy as np
from pathlib import Path

class DroneImageExtractor:
    def __init__(self, model_path="models/aerial_weeds.pt"):
        # Downloads automatically on first run if not present
        self.model = YOLO(model_path)

    def extract(self, image_path: str) -> dict:
        image_path = Path(image_path)
        img = Image.open(image_path)

        results = self.model(str(image_path))[0]

        detections = []
        for box in results.boxes:
            detections.append({
                "class": results.names[int(box.cls)],
                "confidence": round(float(box.conf), 3),
                "bbox": box.xyxy[0].tolist()
            })

        # Extract EXIF metadata (GPS, timestamp from drone)
        exif_data = {}
        if hasattr(img, '_getexif') and img._getexif():
            for tag_id, value in img._getexif().items():
                tag = TAGS.get(tag_id, tag_id)
                exif_data[tag] = value

        gps = self._parse_gps(exif_data)

        return {
            "image": image_path.name,
            "width": img.width,
            "height": img.height,
            "detections": detections,
            "detection_summary": self._summarize(detections),
            "gps": gps,
            "timestamp": exif_data.get("DateTime", "unknown")
        }

    def _summarize(self, detections):
        summary = {}
        for d in detections:
            summary[d["class"]] = summary.get(d["class"], 0) + 1
        return summary

    def _parse_gps(self, exif):
        try:
            gps_info = exif.get("GPSInfo", {})
            lat = gps_info.get(2, None)
            lon = gps_info.get(4, None)
            if lat and lon:
                def to_decimal(dms):
                    d, m, s = dms
                    return float(d) + float(m)/60 + float(s)/3600
                return {
                    "lat": to_decimal(lat),
                    "lon": to_decimal(lon)
                }
        except Exception:
            pass
        return {"lat": None, "lon": None}