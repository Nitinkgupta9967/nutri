"""
Food Vision — FastAPI Backend
Powered by Gemini 2.5 Flash Lite via OpenRouter
"""

import base64
import io
import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
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
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB


# ── Standard API Response Envelope ───────────────────────────────────────────
class APIResponse(BaseModel):
    success: bool
    status_code: int
    message: str
    data: Optional[Any] = None
    error: Optional[dict] = None
    request_id: Optional[str] = None
    response_time: Optional[float] = None


def success_response(
    data: Any,
    message: str = "Request successful",
    request_id: str = "",
    response_time: float = 0.0,
    status_code: int = status.HTTP_200_OK,
) -> dict:
    return {
        "success": True,
        "status_code": status_code,
        "message": message,
        "data": data,
        "error": None,
        "request_id": request_id,
        "response_time": round(response_time, 4),
    }


def error_response(
    message: str,
    error_code: str = "INTERNAL_ERROR",
    details: Optional[str] = None,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
) -> dict:
    return {
        "success": False,
        "status_code": status_code,
        "message": message,
        "data": None,
        "error": {
            "code": error_code,
            "details": details,
        },
    }


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
    """Resize, compress, and base64-encode the uploaded image."""
    try:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error_response(
                message="Uploaded file is not a valid or readable image.",
                error_code="INVALID_IMAGE",
                details="PIL could not open the file. Ensure the file is a valid JPEG, PNG, or WEBP.",
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            ),
        )

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
    """Send the base64 image to the Gemini model and return raw text."""
    log.info(f"🤖 Calling {MODEL_NAME}...")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                        },
                        {
                            "type": "text",
                            "text": (
                                "Return only the exact JSON structure. No other text.\n\n"
                                "IMPORTANT — quantity field:\n"
                                "- COUNTABLE items (roti, idli, vada, pieces of chicken/paneer/fish, "
                                "samosa, egg, ladoo, etc.): set quantity to the INTEGER COUNT you see "
                                "(e.g. 2 rotis → quantity: 2, unit: 'pieces'). Never use 1.0 — use 1.\n"
                                "- UNCOUNTABLE dishes (dal, curry, rice, sabzi, soup, raita, chutney, "
                                "sambar, etc.): set quantity to null, unit: 'g'.\n"
                                "- estimated_weight_g is always total grams regardless of unit."
                            ),
                        },
                    ],
                },
            ],
        )
    except Exception as exc:
        log.error(f"OpenRouter API call failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=error_response(
                message="Failed to reach the AI model. Please try again later.",
                error_code="UPSTREAM_AI_ERROR",
                details=str(exc),
                status_code=status.HTTP_502_BAD_GATEWAY,
            ),
        )

    raw = response.choices[0].message.content.strip()
    finish_reason = response.choices[0].finish_reason
    log.info(f"✅ OpenRouter responded ({len(raw)} chars) | finish_reason={finish_reason}")

    if finish_reason == "length":
        log.warning("⚠️ Response was truncated by token limit — JSON may be incomplete")

    return raw


# ── Robust JSON Cleaner ───────────────────────────────────────────────────────
def clean_and_parse_json(raw: str, timestamp: str) -> MealAnalysis:
    """Parse, sanitise, and validate the model's raw JSON response."""
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Extract largest JSON object
    json_match = re.search(r"(\{[\s\S]*\})", text)
    if json_match:
        text = json_match.group(1)

    # Fix common model JSON errors
    text = re.sub(r",\s*([}\]])", r"\1", text)          # trailing commas
    text = re.sub(r'"\s*,\s*"', '", "', text)           # broken strings
    text = re.sub(r'(\}\s*)\n\s*\{', r'\1,\n  {', text) # missing object separator

    # Coerce float quantities like 1.0 → 1
    text = re.sub(
        r'("quantity":\s*)(\d+)\.0\b',
        lambda m: f'{m.group(1)}{int(float(m.group(2)))}',
        text,
    )

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        log.error(f"JSON parse failed: {exc}\nLast 600 chars:\n{text[-600:]}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error_response(
                message="The AI model returned malformed JSON. Please try a clearer image.",
                error_code="MALFORMED_AI_RESPONSE",
                details=str(exc),
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            ),
        )

    data["timestamp"] = timestamp
    data.setdefault("status", "success")

    # Final quantity type coercion
    for item in data.get("items", []):
        if isinstance(item, dict) and item.get("quantity") is not None:
            try:
                item["quantity"] = int(item["quantity"])
            except (TypeError, ValueError):
                item["quantity"] = None

    try:
        return MealAnalysis(**data)
    except Exception as exc:
        log.error(f"Pydantic validation failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error_response(
                message="The AI response did not match the expected schema.",
                error_code="SCHEMA_VALIDATION_ERROR",
                details=str(exc),
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            ),
        )


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Food Vision API",
    description="Indian meal nutrition analysis powered by Gemini 2.5 Flash Lite",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(diet_router)


# ── Global Exception Handlers ─────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic/FastAPI request validation errors (422)."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response(
            message="Request validation failed. Check your input fields.",
            error_code="REQUEST_VALIDATION_ERROR",
            details=str(exc.errors()),
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        ),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Normalise all HTTPExceptions into the standard envelope."""
    # If detail is already a structured dict, pass it through
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(
            message=str(exc.detail),
            error_code="HTTP_ERROR",
            status_code=exc.status_code,
        ),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all for unexpected server errors (500)."""
    log.exception(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response(
            message="An unexpected error occurred. Please try again later.",
            error_code="INTERNAL_SERVER_ERROR",
            details=type(exc).__name__,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        ),
    )


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", summary="Root", tags=["Meta"])
async def root():
    return success_response(
        data={"service": "Food Vision API", "version": "3.0.0", "model": MODEL_NAME},
        message="Food Vision API is running.",
    )


@app.get("/health", summary="Health Check", tags=["Meta"])
async def health():
    ok = bool(OPENROUTER_API_KEY)
    return JSONResponse(
        status_code=status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=success_response(
            data={"api_key_configured": ok},
            message="Service is healthy." if ok else "API key is not configured.",
            status_code=status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE,
        ),
    )


@app.post(
    "/analyze-meal",
    summary="Analyze a meal image",
    tags=["Meal Analysis"],
    responses={
        200: {"description": "Meal analyzed successfully"},
        400: {"description": "Invalid file type"},
        413: {"description": "File too large"},
        422: {"description": "Unprocessable image or AI response"},
        500: {"description": "Internal server error"},
        502: {"description": "AI upstream error"},
        503: {"description": "API key not configured"},
    },
)
async def analyze_meal(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    start_time = datetime.now(IST)

    log.info(f"[{request_id}] 📥 Received: {file.filename} ({file.content_type})")

    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error_response(
                message="The AI service is temporarily unavailable. API key not configured.",
                error_code="SERVICE_UNAVAILABLE",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            ),
        )

    # ── Content-type check ────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_response(
                message=f"Unsupported file type '{file.content_type}'. Allowed: JPEG, PNG, WEBP.",
                error_code="UNSUPPORTED_FILE_TYPE",
                details=f"Received: {file.content_type}",
                status_code=status.HTTP_400_BAD_REQUEST,
            ),
        )

    raw_bytes = await file.read()

    # ── File-size check ───────────────────────────────────────────────────────
    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=error_response(
                message="File too large. Maximum allowed size is 5 MB.",
                error_code="FILE_TOO_LARGE",
                details=f"Received {len(raw_bytes) / 1024 / 1024:.2f} MB",
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            ),
        )

    b64 = process_image(raw_bytes)
    raw_response = call_gemini_vision(b64)
    result = clean_and_parse_json(raw_response, datetime.now(IST).isoformat())

    response_time = (datetime.now(IST) - start_time).total_seconds()
    log.info(
        f"[{request_id}] ✅ Done | items={len(result.items)} | confidence={result.confidence:.0%} | time={response_time:.2f}s"
    )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=success_response(
            data=result.dict(),
            message="Meal analyzed successfully.",
            request_id=request_id,
            response_time=response_time,
        ),
    )