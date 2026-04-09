# diet.py
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import JSONResponse
from huggingface_hub import InferenceClient
from pydantic import BaseModel, Field

from config import client, MODEL_NAME, IST

log = logging.getLogger("food-vision")

# ── Hugging Face Image Generation Client ─────────────────────────────────────
_hf_token = os.getenv("HF_TOKEN", "")
hf_client = InferenceClient(token=_hf_token) if _hf_token else None

if not _hf_token:
    log.warning("⚠️  HF_TOKEN not found. Meal image generation will be disabled.")

diet_router = APIRouter(prefix="/diet", tags=["Diet Recommendation"])


# ── Standard Response Helpers (local copies to avoid circular import) ─────────
def _success(data, message="Request successful", status_code=200):
    return {
        "success": True,
        "status_code": status_code,
        "message": message,
        "data": data,
        "error": None,
    }


def _error(message, error_code="INTERNAL_ERROR", details=None, status_code=500):
    return {
        "success": False,
        "status_code": status_code,
        "message": message,
        "data": None,
        "error": {"code": error_code, "details": details},
    }


# ── Request Model ─────────────────────────────────────────────────────────────
VALID_GENDERS = {"Male", "Female", "Other"}
VALID_ACTIVITY_LEVELS = {
    "Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Super Active"
}
VALID_MEAL_PREFERENCES = {"Veg", "Non-Veg", "Eggitarian", "Vegan"}


class DietRequest(BaseModel):
    gender: str = Field(..., example="Male")
    age: int = Field(..., gt=12, lt=100, example=28)
    activity_level: str = Field(..., example="Moderately Active")
    meal_preference: str = Field(..., example="Veg")
    cuisine_preference: str = Field(..., example="North Indian")
    primary_goal: str = Field(..., example="Weight Loss")
    chronic_diseases: List[str] = Field(default=[], example=["Diabetes"])
    allergies: List[str] = Field(default=[], example=["Peanuts"])


# ── Response Models ───────────────────────────────────────────────────────────
class Meal(BaseModel):
    time: str
    name: str
    calories: int
    protein: int
    carbs: int
    fat: int
    description: str
    image_prompt: str
    image_url: str = ""


class DailyPlan(BaseModel):
    day: str
    meals: List[Meal]
    total_calories: int
    total_protein: int
    total_carbs: int
    total_fat: int


class SmartSwap(BaseModel):
    instead_of: str
    swap_with: str
    reason: str


class DietResponse(BaseModel):
    success: bool
    status_code: int
    message: str
    data: Optional[dict] = None
    error: Optional[dict] = None


# ── System Prompt ─────────────────────────────────────────────────────────────
DIET_SYSTEM_PROMPT = """You are an expert Indian dietitian.

Create a realistic, tasty 7-day weekly diet plan based on the user's cuisine preference and health goals.

Return ONLY valid JSON in this exact format. No extra text, no markdown, no code blocks.

CRITICAL: Every meal object MUST contain all 8 fields: time, name, calories, protein, carbs, fat, description, image_prompt.
Missing "carbs" or "fat" from ANY meal is a fatal error.

{
  "daily_calories_target": 1800,
  "plan": [
    {
      "day": "Day 1 - Monday",
      "meals": [
        {
          "time": "08:00 AM BREAKFAST",
          "name": "Masala Oats Upma",
          "calories": 350,
          "protein": 12,
          "carbs": 45,
          "fat": 8,
          "description": "Oats with mustard seeds, curry leaves, mixed veggies",
          "image_prompt": "masala oats upma bowl, Indian breakfast"
        },
        {
          "time": "01:00 PM LUNCH",
          "name": "Dal Tadka with Brown Rice",
          "calories": 480,
          "protein": 20,
          "carbs": 72,
          "fat": 10,
          "description": "Yellow dal with cumin and garlic, served with brown rice",
          "image_prompt": "dal tadka brown rice, Indian lunch"
        },
        {
          "time": "04:00 PM SNACK",
          "name": "Roasted Chana",
          "calories": 150,
          "protein": 8,
          "carbs": 22,
          "fat": 3,
          "description": "Crunchy roasted chickpeas with chaat masala",
          "image_prompt": "roasted chana bowl, healthy snack"
        },
        {
          "time": "08:00 PM DINNER",
          "name": "Palak Paneer with 2 Rotis",
          "calories": 420,
          "protein": 22,
          "carbs": 38,
          "fat": 15,
          "description": "Spinach curry with paneer and whole wheat rotis",
          "image_prompt": "palak paneer rotis, Indian dinner"
        }
      ],
      "total_calories": 1400,
      "total_protein": 62,
      "total_carbs": 177,
      "total_fat": 36
    }
  ],
  "smart_swaps": [
    {"instead_of": "White Rice", "swap_with": "Brown Rice", "reason": "Lower glycemic index, more fiber"},
    {"instead_of": "Maida Roti", "swap_with": "Whole Wheat Roti", "reason": "Higher fiber, stabilizes blood sugar"},
    {"instead_of": "Full Fat Milk", "swap_with": "Skimmed Milk", "reason": "Lower saturated fat, same protein"},
    {"instead_of": "Fruit Juice", "swap_with": "Whole Fruit", "reason": "Fiber slows sugar absorption"},
    {"instead_of": "Fried Snacks", "swap_with": "Roasted Alternatives", "reason": "Lower in calories and fat"}
  ],
  "foods_to_avoid": [
    "Fried snacks like samosa, pakora, and vada",
    "Sugary drinks like cola, packaged juices, and energy drinks",
    "White bread and maida-based products like pav and naan",
    "Processed foods like chips, biscuits, and instant noodles",
    "High-sugar sweets like gulab jamun, jalebi, and rasgulla",
    "Alcohol and carbonated beverages"
  ],
  "ai_notes": "Plan tailored for your goal. Stay hydrated with 2.5-3L water daily."
}

STRICT RULES:
- MANDATORY meal fields (all 8, no exceptions): time, name, calories, protein, carbs, fat, description, image_prompt.
- Each day must have exactly 4 meals: Breakfast, Lunch, Evening Snack, Dinner.
- Every day must include total_calories, total_protein, total_carbs, total_fat as integers.
- Meals MUST match cuisine_preference (South Indian: idli/dosa/sambar; North Indian: roti/dal/sabzi; Continental: salads/wraps/soups).
- Strictly respect meal_preference (Veg / Non-Veg / Eggitarian / Vegan).
- smart_swaps: exactly 5 items. foods_to_avoid: exactly 6 items.
- description: max 12 words. image_prompt: max 8 words.
- All numeric values must be integers.
- No trailing commas. No markdown. Raw JSON only.
"""


# ── Input Validation ──────────────────────────────────────────────────────────
def validate_diet_request(req: DietRequest) -> None:
    errors = []

    if req.gender not in VALID_GENDERS:
        errors.append(f"gender must be one of {sorted(VALID_GENDERS)}.")

    if req.activity_level not in VALID_ACTIVITY_LEVELS:
        errors.append(f"activity_level must be one of {sorted(VALID_ACTIVITY_LEVELS)}.")

    if req.meal_preference not in VALID_MEAL_PREFERENCES:
        errors.append(f"meal_preference must be one of {sorted(VALID_MEAL_PREFERENCES)}.")

    if not req.cuisine_preference.strip():
        errors.append("cuisine_preference cannot be blank.")

    if not req.primary_goal.strip():
        errors.append("primary_goal cannot be blank.")

    if errors:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=_error(
                message="Request validation failed.",
                error_code="REQUEST_VALIDATION_ERROR",
                details=errors,
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            ),
        )


# ── Image Prompt Enhancer ─────────────────────────────────────────────────────
def enhance_image_prompt(meal_name: str, base_prompt: str) -> str:
    """Convert a short LLM prompt into a detailed, realistic Indian food prompt."""
    enhancements = {
        "palak paneer": "vibrant green spinach gravy with soft paneer cubes, garnished with fresh cream and coriander",
        "dal makhani": "rich dark creamy black lentil curry, velvety texture, garnished with coriander",
        "dal tadka": "yellow dal with cumin garlic tadka, garnished with coriander",
        "upma": "coarse masala oats upma with mustard seeds, curry leaves, mixed vegetables",
        "roti": "fresh soft whole wheat roti",
        "idli": "steaming hot soft idlis served with coconut chutney and sambar",
        "dosa": "crispy golden dosa with potato masala and chutneys",
        "paneer": "soft paneer cubes in rich gravy",
    }

    extra = next(
        (desc for key, desc in enhancements.items() if key in meal_name.lower()), ""
    )

    return (
        f"Authentic Indian food photography of {base_prompt}, {extra}, "
        "steaming hot, appetizing, professional close-up shot, natural studio lighting, "
        "high detail, sharp focus, vibrant colors, realistic, mouthwatering, restaurant style"
    )


# ── Meal Image Generation ─────────────────────────────────────────────────────
async def generate_meal_image(meal_name: str, base_prompt: str) -> str:
    """Generate a meal image via Hugging Face and return its static URL path."""
    if not base_prompt or not hf_client:
        return ""

    enhanced_prompt = enhance_image_prompt(meal_name, base_prompt)

    try:
        image = hf_client.text_to_image(
            prompt=enhanced_prompt,
            model="black-forest-labs/FLUX.1-schnell",
            width=1024,
            height=768,
            num_inference_steps=25,
        )

        safe_name = re.sub(r"[^a-zA-Z0-9]", "_", meal_name.lower())[:40]
        os.makedirs("meal_images", exist_ok=True)
        filename = f"meal_images/{safe_name}.png"
        image.save(filename)

        return f"/static/meal_images/{os.path.basename(filename)}"

    except Exception as exc:
        log.error(f"Image generation failed for '{meal_name}': {exc}")
        return ""


# ── Robust JSON Cleaner ───────────────────────────────────────────────────────
def clean_and_parse_diet_json(raw: str) -> dict:
    """Strip noise from the model's response and return a parsed dict."""
    text = raw.strip()

    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    json_match = re.search(r"(\{[\s\S]*\})", text)
    if json_match:
        text = json_match.group(1)

    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r'"\s*,\s*"', '", "', text)
    text = re.sub(r'(\}\s*)\n\s*\{', r'\1,\n  {', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        log.error(f"Diet JSON parse failed: {exc}\nLast 300 chars:\n{text[-300:]}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=_error(
                message="The AI model returned malformed JSON for the diet plan.",
                error_code="MALFORMED_AI_RESPONSE",
                details=str(exc),
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            ),
        )


# ── Parallel Image Enrichment ─────────────────────────────────────────────────
async def enrich_plan_with_images(plan: List[dict]) -> None:
    """Attach generated image URLs to every meal in the plan (runs in parallel)."""
    tasks = [
        generate_meal_image(meal.get("name", ""), meal.get("image_prompt", ""))
        for day in plan
        for meal in day.get("meals", [])
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    idx = 0
    for day in plan:
        for meal in day.get("meals", []):
            url = results[idx] if idx < len(results) and not isinstance(results[idx], Exception) else ""
            meal["image_url"] = url
            idx += 1


# ── Endpoint ──────────────────────────────────────────────────────────────────
@diet_router.post(
    "/recommend-diet",
    summary="Generate a 7-day personalised diet plan",
    responses={
        200: {"description": "Diet plan generated successfully"},
        422: {"description": "Validation error or malformed AI response"},
        502: {"description": "AI upstream error"},
        503: {"description": "API key not configured"},
    },
)
async def recommend_diet(request: DietRequest):
    from config import OPENROUTER_API_KEY  # local import to avoid circular

    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_error(
                message="The AI service is temporarily unavailable. API key not configured.",
                error_code="SERVICE_UNAVAILABLE",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            ),
        )

    validate_diet_request(request)

    user_prompt = f"""
Gender: {request.gender}
Age: {request.age} years
Activity Level: {request.activity_level}
Meal Preference: {request.meal_preference}
Cuisine Preference: {request.cuisine_preference}
Primary Goal: {request.primary_goal}
Chronic Diseases: {', '.join(request.chronic_diseases) if request.chronic_diseases else 'None'}
Allergies: {', '.join(request.allergies) if request.allergies else 'None'}

Generate a complete 7-day diet plan matching the cuisine preference above.
REMINDER: Every meal must include all 8 fields — especially "carbs" and "fat". Never skip them.
Include 5 smart swaps and 6 foods to avoid based on the goal and chronic diseases.
Return raw JSON only — no markdown, no code blocks.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=8000,
            temperature=0.7,
            messages=[
                {"role": "system", "content": DIET_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        log.error(f"OpenRouter API call failed (diet): {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=_error(
                message="Failed to reach the AI model. Please try again later.",
                error_code="UPSTREAM_AI_ERROR",
                details=str(exc),
                status_code=status.HTTP_502_BAD_GATEWAY,
            ),
        )

    raw = response.choices[0].message.content.strip()
    finish_reason = response.choices[0].finish_reason

    log.info(f"Diet response: {len(raw)} chars | finish_reason={finish_reason}")
    if finish_reason == "length":
        log.warning("⚠️ Diet response truncated — JSON may be incomplete")

    data = clean_and_parse_diet_json(raw)
    data["generated_at"] = datetime.now(IST).isoformat()

    await enrich_plan_with_images(data["plan"])

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=_success(
            data=data,
            message="7-day diet plan generated successfully.",
            status_code=status.HTTP_200_OK,
        ),
    )