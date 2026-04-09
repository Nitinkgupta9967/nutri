# config.py
import os
import logging
from datetime import timezone, timedelta
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

log = logging.getLogger("food-vision")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
IST = timezone(timedelta(hours=5, minutes=30))
MODEL_NAME = "google/gemini-2.5-flash-lite"

if not OPENROUTER_API_KEY:
    log.critical("❌ OPENROUTER_API_KEY is not set. API will not function.")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": os.getenv("APP_URL", "http://localhost:8000"),
        "X-Title": "Food Vision API",
    },
)