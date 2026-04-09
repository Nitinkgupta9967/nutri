"""
# Food Vision — FastAPI Backend
# Powered by Gemini 2.5 Flash Lite via OpenRouter (Ultra Robust JSON Parsing)
"""

import base64
import io
import json
import logging
import re
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from config import client, MODEL_NAME, IST, OPENROUTER_API_KEY
from diet import diet_router

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("food-vision")

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_IMAGE_PX = 768
MAX_TOKENS = 2048


# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class Nutrition(BaseModel):
    calories: float
    protein: float
    carbohydrates: float
    fat: float


class FoodItem(BaseModel):
    name: str
    quantity: Optional[int] = None
    unit: str = "g"
    estimated_weight_g: float
    confidence: float = Field(..., ge=0.0, le=1.0)
    nutrition: Nutrition


class MacroDistribution(BaseModel):
    protein_percentage: float
    carbs_percentage: float
    fat_percentage: float


class MealAnalysis(BaseModel):
    status: str = "success"
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: str
    items: list[FoodItem]
    totals: Nutrition
    macro_distribution: MacroDistribution


# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert Indian food nutritionist.

Analyze the meal image and output **ONLY** valid JSON in this exact structure.
Nothing else — no explanation, no markdown, no extra text.

{
  "status": "success",
  "confidence": 0.85,
  "timestamp": "",
  "items": [
    {
      "name": "Roti",
      "quantity": 2,
      "unit": "pieces",
      "estimated_weight_g": 60.0,
      "confidence": 0.95,
      "nutrition": {
        "calories": 160.0,
        "protein": 4.0,
        "carbohydrates": 30.0,
        "fat": 3.0
      }
    },
    {
      "name": "Dal Tadka",
      "quantity": null,
      "unit": "g",
      "estimated_weight_g": 150.0,
      "confidence": 0.9,
      "nutrition": {
        "calories": 180.0,
        "protein": 10.0,
        "carbohydrates": 24.0,
        "fat": 5.0
      }
    }
  ],
  "totals": {
    "calories": 340.0,
    "protein": 14.0,
    "carbohydrates": 54.0,
    "fat": 8.0
  },
  "macro_distribution": {
    "protein_percentage": 16.0,
    "carbs_percentage": 64.0,
    "fat_percentage": 20.0
  }
}

QUANTITY RULES — follow strictly:
- COUNTABLE items (roti, chapati, idli, vada, dosa, paratha, puri, egg, samosa,
  piece of chicken/fish/paneer, biscuit, ladoo, gulab jamun, etc.):
  → quantity = INTEGER count visible in the image (e.g. 3 idlis → quantity: 3)
  → unit = "pieces"
- UNCOUNTABLE / POURABLE dishes (dal, curry, sabzi, rice, khichdi, soup, raita,
  chutney, gravy, halwa, sambar, coconut chutney, etc.):
  → quantity = null
  → unit = "g"
- quantity must be an integer (1, 2, 3 …) — NEVER a float like 1.0
- estimated_weight_g is always total grams regardless of unit.
- All nutrition values represent the TOTAL for that item (not per piece).
- All numbers except quantity must be floats.
"""


# ── Image Processing ──────────────────────────────────────────────────────────
def process_image(raw_bytes: bytes) -> str:
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    w, h = img.size
    scale = min(MAX_IMAGE_PX / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        log.info(f"   Resized: {w}×{h} → {img.size[0]}×{img.size[1]}")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


# ── Gemini Vision Call ────────────────────────────────────────────────────────
def call_gemini_vision(b64_image: str) -> str:
    log.info(f"🤖 Calling {MODEL_NAME}...")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
                    {"type": "text", "text": (
                        "Return only the exact JSON structure. No other text.\n\n"
                        "IMPORTANT — quantity field:\n"
                        "- COUNTABLE items (roti, idli, vada, pieces of chicken/paneer/fish, "
                        "samosa, egg, ladoo, etc.): set quantity to the INTEGER COUNT you see "
                        "(e.g. 2 rotis → quantity: 2, unit: 'pieces'). Never use 1.0 — use 1.\n"
                        "- UNCOUNTABLE dishes (dal, curry, rice, sabzi, soup, raita, chutney, "
                        "sambar, etc.): set quantity to null, unit: 'g'.\n"
                        "- estimated_weight_g is always total grams regardless of unit."
                    )}
                ]
            }
        ]
    )

    raw = response.choices[0].message.content.strip()
    finish_reason = response.choices[0].finish_reason
    log.info(f"✅ OpenRouter responded ({len(raw)} chars) | finish_reason={finish_reason}")

    if finish_reason == "length":
        log.warning("⚠️ Response was truncated by token limit — consider raising MAX_TOKENS further")

    return raw


# ── Ultra Robust JSON Cleaner ─────────────────────────────────────────────────
def clean_and_parse_json(raw: str, timestamp: str) -> MealAnalysis:
    text = raw.strip()

    # Remove markdown code blocks
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Extract the largest JSON object
    json_match = re.search(r"(\{[\s\S]*\})", text)
    if json_match:
        text = json_match.group(1)

    # Fix common Gemini JSON errors
    text = re.sub(r",\s*([}\]])", r"\1", text)   # trailing commas
    text = re.sub(r'"\s*,\s*"', '", "', text)    # fix broken strings
    text = re.sub(r'(\}\s*)\n\s*\{', r'\1,\n  {', text)  # missing comma before new object

    # Fix quantity written as float (1.0 → 1)
    text = re.sub(
        r'("quantity":\s*)(\d+)\.0\b',
        lambda m: f'{m.group(1)}{int(float(m.group(2)))}',
        text
    )

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        log.error(f"JSON parse still failed: {e}\nLast 600 chars:\n{text[-600:]}")
        raise HTTPException(
            status_code=500,
            detail={"success": False, "message": "Model returned malformed JSON. Please try a different/clearer image."}
        )

    data["timestamp"] = timestamp
    data.setdefault("status", "success")

    # Coerce quantity to int if the model still sneaked in a float
    for item in data.get("items", []):
        if isinstance(item, dict) and item.get("quantity") is not None:
            try:
                item["quantity"] = int(item["quantity"])
            except (TypeError, ValueError):
                item["quantity"] = None

    return MealAnalysis(**data)


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Food Vision API",
    description="Indian meal nutrition analysis",
    version="2.8.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(diet_router)


@app.get("/")
async def root():
    return {"service": "Food Vision API", "version": "2.8.0", "model": MODEL_NAME, "status": "online"}


@app.get("/health")
async def health():
    return {"status": "ok", "key_set": bool(OPENROUTER_API_KEY)}


@app.post("/analyze-meal")
async def analyze_meal(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    start_time = datetime.now(IST)

    log.info(f"[{request_id}] 📥 Received: {file.filename} ({file.content_type})")

    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=503, detail={"success": False, "message": "API key not configured"})

    try:
        raw_bytes = await file.read()

        if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
            raise HTTPException(status_code=400, detail={"success": False, "message": "Unsupported file type"})

        if len(raw_bytes) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail={"success": False, "message": "File too large (max 5MB)"})

        b64 = process_image(raw_bytes)
        raw_response = call_gemini_vision(b64)
        result = clean_and_parse_json(raw_response, datetime.now(IST).isoformat())

        response_time = (datetime.now(IST) - start_time).total_seconds()
        log.info(f"[{request_id}] ✅ Done | items={len(result.items)} | confidence={result.confidence:.0%}")

        return {
            "success": True,
            "message": "Meal analyzed successfully",
            "request_id": request_id,
            "response_time": response_time,
            "data": result.dict()
        }

    except HTTPException:
        raise
    except Exception as exc:
        log.error(f"[{request_id}] ❌ Error: {exc}")
        raise HTTPException(
            status_code=500,
            detail={"success": False, "message": "Failed to analyze the meal. Try a clearer image."}
        )