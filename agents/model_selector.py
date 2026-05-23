# agents/model_selector.py
import vertexai
from google.cloud import aiplatform
import os

# Known working Gemini models on Vertex AI
KNOWN_MODELS = [
    "gemini-2.5-flash",        
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite-001",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-002",
    "gemini-1.5-pro-001",
]

def pick_model() -> str:
    """Show available models and let user pick one."""
    print("\n🤖 Available Gemini Models on Vertex AI:")
    print("─" * 40)
    for i, model in enumerate(KNOWN_MODELS, 1):
        print(f"  [{i}] {model}")
    print("─" * 40)

    while True:
        try:
            choice = input("Pick a model number (default=1): ").strip()
            if choice == "":
                selected = KNOWN_MODELS[0]
                break
            idx = int(choice) - 1
            if 0 <= idx < len(KNOWN_MODELS):
                selected = KNOWN_MODELS[idx]
                break
            else:
                print(f"  ⚠️  Enter a number between 1 and {len(KNOWN_MODELS)}")
        except ValueError:
            print("  ⚠️  Please enter a valid number")

    print(f"  ✅ Using: {selected}\n")
    return selected