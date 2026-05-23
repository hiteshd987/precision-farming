# agents/weather_agent.py
import requests
import os
from datetime import datetime

class WeatherAgent:
    def __init__(self):
        self.api_key = os.getenv("WEATHER_API_KEY")
        self.base_url = "https://api.openweathermap.org/data/2.5"

    def get_historical(self, lat: float, lon: float, timestamp: str) -> dict:
        """Fetch weather for coordinates. Falls back to current if no GPS."""
        if not lat or not lon:
            return {"error": "No GPS data in image", "data": None}

        try:
            # Current weather (free tier)
            url = f"{self.base_url}/weather"
            params = {
                "lat": lat, "lon": lon,
                "appid": self.api_key,
                "units": "metric"
            }
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()

            return {
                "location": data.get("name", f"{lat},{lon}"),
                "temperature_c": data["main"]["temp"],
                "humidity_pct": data["main"]["humidity"],
                "conditions": data["weather"][0]["description"],
                "wind_speed_ms": data["wind"]["speed"],
                "precipitation_mm": data.get("rain", {}).get("1h", 0),
                "timestamp": timestamp
            }
        except Exception as e:
            return {"error": str(e), "data": None}