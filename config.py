# config.py
import os
from datetime import timezone, timedelta
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
IST = timezone(timedelta(hours=5, minutes=30))
MODEL_NAME = "google/gemini-2.5-flash-lite"

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Food Vision API",
    }
)