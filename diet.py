# diet.py
import json
import re
import logging
from datetime import datetime
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import client, MODEL_NAME, IST

log = logging.getLogger("food-vision")

diet_router = APIRouter(prefix="/diet", tags=["Diet Recommendation"])


# ── Request Model ─────────────────────────────────────────────────────────────
class DietRequest(BaseModel):
    gender: str = Field(..., example="Male")
    age: int = Field(..., gt=12, lt=100, example=28)
    activity_level: str = Field(..., example="Moderately Active")
    meal_preference: str = Field(..., example="Veg")              # Veg, Non-Veg, Eggitarian, Vegan
    cuisine_preference: str = Field(..., example="North Indian")  # North Indian, South Indian, Continental, etc.
    primary_goal: str = Field(..., example="Weight Loss")
    chronic_diseases: List[str] = Field(default=[], example=["Diabetes"])
    allergies: List[str] = Field(default=[], example=["Peanuts"])


# ── Response Models ───────────────────────────────────────────────────────────
class Meal(BaseModel):
    time: str
    name: str
    calories: int
    protein: int
    description: str
    image_prompt: str


class DailyPlan(BaseModel):
    day: str
    meals: List[Meal]
    total_calories: int
    total_protein: int


class SmartSwap(BaseModel):
    instead_of: str
    swap_with: str
    reason: str


class DietResponse(BaseModel):
    success: bool
    daily_calories_target: int
    plan: List[DailyPlan]
    smart_swaps: List[SmartSwap]
    foods_to_avoid: List[str]
    ai_notes: str
    generated_at: str


# ── System Prompt ─────────────────────────────────────────────────────────────
DIET_SYSTEM_PROMPT = """You are an expert Indian dietitian.

Create a realistic, tasty 7-day weekly diet plan based on the user's cuisine preference and health goals.

Return **ONLY** valid JSON in this exact format. No extra text, no markdown, no code blocks.

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
          "description": "Oats cooked with mustard seeds, curry leaves, mixed veggies and spices",
          "image_prompt": "bowl of masala oats upma with vegetables, Indian breakfast, appetizing food photo"
        },
        {
          "time": "01:00 PM LUNCH",
          "name": "Dal Tadka with Brown Rice",
          "calories": 480,
          "protein": 20,
          "description": "Yellow dal tempered with cumin, garlic and served with brown rice",
          "image_prompt": "dal tadka with brown rice, Indian lunch, food photography"
        },
        {
          "time": "04:00 PM SNACK",
          "name": "Roasted Chana",
          "calories": 150,
          "protein": 8,
          "description": "Crunchy roasted chickpeas with chaat masala",
          "image_prompt": "roasted chana in a bowl, healthy Indian snack, food photo"
        },
        {
          "time": "08:00 PM DINNER",
          "name": "Palak Paneer with 2 Rotis",
          "calories": 420,
          "protein": 22,
          "description": "Spinach curry with paneer cubes served with whole wheat rotis",
          "image_prompt": "palak paneer with rotis, Indian dinner, food photography"
        }
      ],
      "total_calories": 1400,
      "total_protein": 62
    }
  ],
  "smart_swaps": [
    {
      "instead_of": "White Rice",
      "swap_with": "Brown Rice or Cauliflower Rice",
      "reason": "Lower glycemic index, more fiber, better for weight loss and blood sugar control"
    },
    {
      "instead_of": "Maida Roti / Naan",
      "swap_with": "Whole Wheat Roti or Multigrain Roti",
      "reason": "Higher fiber content keeps you fuller longer and stabilizes blood sugar"
    },
    {
      "instead_of": "Full Fat Milk",
      "swap_with": "Skimmed Milk or Unsweetened Soy Milk",
      "reason": "Lower in saturated fat and calories while maintaining protein content"
    },
    {
      "instead_of": "Fruit Juice",
      "swap_with": "Whole Fruit",
      "reason": "Whole fruit has fiber that slows sugar absorption and keeps you full"
    },
    {
      "instead_of": "Fried Snacks",
      "swap_with": "Roasted or Baked Alternatives",
      "reason": "Significantly lower in calories and unhealthy fats"
    }
  ],
  "foods_to_avoid": [
    "Fried snacks like samosa, pakora, and vada",
    "Sugary drinks like cola, packaged juices, and energy drinks",
    "White bread and maida-based products like pav and naan",
    "Processed foods like chips, biscuits, and instant noodles",
    "High-sugar sweets like gulab jamun, jalebi, and rasgulla",
    "Alcohol and carbonated beverages"
  ],
  "ai_notes": "This plan is tailored for your goal with cuisine-specific meals. Stay hydrated with 2.5-3L water daily."
}

STRICT RULES:
- Each day must have exactly 4 meals: Breakfast, Lunch, Evening Snack, Dinner.
- Meals MUST match the user's cuisine_preference (e.g. South Indian → idli, dosa, sambar, rasam; North Indian → roti, sabzi, dal, paratha; Continental → salads, sandwiches, soups, wraps).
- Also strictly respect meal_preference (Veg / Non-Veg / Eggitarian / Vegan).
- smart_swaps must have exactly 5 items relevant to the user's goal and cuisine.
- foods_to_avoid must have exactly 6 items relevant to the user's chronic diseases and goal.
- All calorie and protein values must be integers.
- No trailing commas anywhere in the JSON.
- No markdown, no code fences, no extra explanation — raw JSON only.
"""


# ── Robust JSON Cleaner ───────────────────────────────────────────────────────
def clean_and_parse_diet_json(raw: str) -> dict:
    text = raw.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Extract the largest JSON object
    json_match = re.search(r"(\{[\s\S]*\})", text)
    if json_match:
        text = json_match.group(1)

    # Fix common JSON errors
    text = re.sub(r",\s*([}\]])", r"\1", text)            # trailing commas
    text = re.sub(r'"\s*,\s*"', '", "', text)             # broken strings
    text = re.sub(r'(\}\s*)\n\s*\{', r'\1,\n  {', text)   # missing comma between objects

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error(f"Diet JSON parse failed: {e}\nLast 300 chars:\n{text[-300:]}")
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "message": "Failed to generate diet recommendation",
                "error": str(e)
            }
        )


# ── Endpoint ──────────────────────────────────────────────────────────────────
@diet_router.post("/recommend-diet", response_model=DietResponse)
async def recommend_diet(request: DietRequest):
    user_prompt = f"""
Gender: {request.gender}
Age: {request.age} years
Activity Level: {request.activity_level}
Meal Preference: {request.meal_preference}
Cuisine Preference: {request.cuisine_preference}
Primary Goal: {request.primary_goal}
Chronic Diseases: {', '.join(request.chronic_diseases) if request.chronic_diseases else 'None'}
Allergies: {', '.join(request.allergies) if request.allergies else 'None'}

Generate a complete 7-day diet plan where every meal strictly matches the cuisine preference above.
Include 5 smart swaps and 6 foods to avoid based on the goal and chronic diseases.
Return raw JSON only — no markdown, no code blocks.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=4000,
        temperature=0.7,
        messages=[
            {"role": "system", "content": DIET_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )

    raw = response.choices[0].message.content.strip()
    finish_reason = response.choices[0].finish_reason

    log.info(f"Diet response: {len(raw)} chars | finish_reason={finish_reason}")

    if finish_reason == "length":
        log.warning("⚠️ Diet response truncated — JSON likely incomplete")

    data = clean_and_parse_diet_json(raw)
    data["generated_at"] = datetime.now(IST).isoformat()

    return {
        "success": True,
        **data
    }