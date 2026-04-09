# 🍱 Food Vision API

An AI-powered **Indian meal nutrition analyser** and **personalised diet recommendation system** built with **FastAPI**, powered by **Gemini 2.5 Flash Lite** via OpenRouter.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Environment Variables](#environment-variables)
- [Running the Server](#running-the-server)
- [API Reference](#api-reference)
  - [Standard Response Format](#standard-response-format)
  - [Error Codes Reference](#error-codes-reference)
  - [GET /](#get-)
  - [GET /health](#get-health)
  - [POST /analyze-meal](#post-analyze-meal)
  - [POST /diet/recommend-diet](#post-dietrecommend-diet)
- [How It Works](#how-it-works)
- [Constraints & Limits](#constraints--limits)
- [Security Notes](#security-notes)
- [Future Improvements](#future-improvements)

---

## Features

### 🥗 Meal Analysis (Computer Vision)
- Upload any Indian meal image (JPEG / PNG / WEBP)
- Detects multiple food items in a single photo
- Returns per-item and total nutrition: **Calories, Protein, Carbohydrates, Fat**
- Computes macro-percentage distribution
- Handles countable (roti, idli) and uncountable (dal, curry) items correctly
- Ultra-robust JSON parser that auto-fixes common model output errors

### 🧠 Smart Diet Recommendation
- Generates a **7-day personalised diet plan** (Breakfast, Lunch, Snack, Dinner)
- Customised by age, gender, activity level, cuisine preference, health goal, chronic diseases, and allergies
- Supports **Veg / Non-Veg / Eggitarian / Vegan** meal preferences
- Includes **5 smart food swaps** and **6 foods to avoid**
- Optional AI-generated meal images via Hugging Face FLUX.1-schnell

---

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI 0.111+ |
| AI Model | Gemini 2.5 Flash Lite (via OpenRouter) |
| Image Processing | Pillow (PIL) |
| Schema Validation | Pydantic v2 |
| Image Generation | Hugging Face Inference API (FLUX.1-schnell) |
| Runtime | Python 3.11+ |

---

## Project Structure

```
food-vision/
├── app.py          # Main FastAPI app — meal analysis endpoints
├── diet.py         # Diet recommendation router & logic
├── config.py       # API configuration and OpenRouter client
├── .env            # Secret keys (git-ignored)
├── meal_images/    # Generated meal images (auto-created, git-ignored)
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd food-vision
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Required — OpenRouter API key for Gemini model access
OPENROUTER_API_KEY=your_openrouter_key_here

# Optional — Hugging Face token for meal image generation
HF_TOKEN=your_hf_token_here

# Optional — shown in OpenRouter request headers (default: http://localhost:8000)
APP_URL=https://your-domain.com
```

> `.env` is listed in `.gitignore` and is never committed.

---

## Running the Server

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`.

Interactive docs: `http://localhost:8000/docs`

---

## API Reference

### Standard Response Format

Every endpoint returns a consistent JSON envelope regardless of success or failure.

#### ✅ Success Response

```json
{
  "success": true,
  "status_code": 200,
  "message": "Meal analyzed successfully.",
  "data": { ... },
  "error": null,
  "request_id": "a1b2c3d4-...",
  "response_time": 2.341
}
```

#### ❌ Error Response

```json
{
  "success": false,
  "status_code": 422,
  "message": "The AI model returned malformed JSON.",
  "data": null,
  "error": {
    "code": "MALFORMED_AI_RESPONSE",
    "details": "Expecting ',' delimiter: line 12 column 3"
  }
}
```

---

### Error Codes Reference

| HTTP Status | Error Code | Meaning |
|---|---|---|
| 400 | `UNSUPPORTED_FILE_TYPE` | File is not JPEG, PNG, or WEBP |
| 413 | `FILE_TOO_LARGE` | File exceeds the 5 MB limit |
| 422 | `INVALID_IMAGE` | PIL could not read the uploaded file |
| 422 | `MALFORMED_AI_RESPONSE` | Model returned invalid JSON |
| 422 | `SCHEMA_VALIDATION_ERROR` | Parsed JSON does not match expected schema |
| 422 | `REQUEST_VALIDATION_ERROR` | Invalid request body fields |
| 500 | `INTERNAL_SERVER_ERROR` | Unexpected server-side error |
| 502 | `UPSTREAM_AI_ERROR` | Could not reach the OpenRouter / AI model |
| 503 | `SERVICE_UNAVAILABLE` | `OPENROUTER_API_KEY` is not configured |

---

### GET /

Health and version info.

**Response `200 OK`**
```json
{
  "success": true,
  "status_code": 200,
  "message": "Food Vision API is running.",
  "data": {
    "service": "Food Vision API",
    "version": "3.0.0",
    "model": "google/gemini-2.5-flash-lite"
  }
}
```

---

### GET /health

Checks whether the API key is configured.

**Response `200 OK`** — key is set
```json
{
  "success": true,
  "status_code": 200,
  "message": "Service is healthy.",
  "data": { "api_key_configured": true }
}
```

**Response `503 Service Unavailable`** — key is missing
```json
{
  "success": false,
  "status_code": 503,
  "message": "API key is not configured.",
  "data": { "api_key_configured": false }
}
```

---

### POST /analyze-meal

Analyse an Indian meal image and return detailed nutrition data.

**Request**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `multipart/form-data` | ✅ | Meal image (JPEG / PNG / WEBP, max 5 MB) |

**Response `200 OK`**
```json
{
  "success": true,
  "status_code": 200,
  "message": "Meal analyzed successfully.",
  "request_id": "a1b2c3d4-e5f6-...",
  "response_time": 2.84,
  "data": {
    "status": "success",
    "confidence": 0.88,
    "timestamp": "2025-07-15T14:30:00+05:30",
    "items": [
      {
        "name": "Dal Tadka",
        "quantity": null,
        "unit": "g",
        "estimated_weight_g": 150.0,
        "confidence": 0.92,
        "nutrition": {
          "calories": 180.0,
          "protein": 10.0,
          "carbohydrates": 24.0,
          "fat": 5.0
        }
      },
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
      }
    ],
    "totals": {
      "calories": 340.0,
      "protein": 14.0,
      "carbohydrates": 54.0,
      "fat": 8.0
    },
    "macro_distribution": {
      "protein_percentage": 16.5,
      "carbs_percentage": 63.5,
      "fat_percentage": 20.0
    }
  }
}
```

**Error Responses**

| Status | When |
|---|---|
| `400 Bad Request` | File is not JPEG, PNG, or WEBP |
| `413 Request Entity Too Large` | File is larger than 5 MB |
| `422 Unprocessable Entity` | File is corrupt / AI returned bad JSON |
| `502 Bad Gateway` | Could not reach OpenRouter |
| `503 Service Unavailable` | `OPENROUTER_API_KEY` not set |

---

### POST /diet/recommend-diet

Generate a 7-day personalised Indian diet plan.

**Request Body**

```json
{
  "gender": "Male",
  "age": 28,
  "activity_level": "Moderately Active",
  "meal_preference": "Veg",
  "cuisine_preference": "North Indian",
  "primary_goal": "Weight Loss",
  "chronic_diseases": ["Diabetes"],
  "allergies": ["Peanuts"]
}
```

| Field | Type | Required | Allowed Values |
|---|---|---|---|
| `gender` | string | ✅ | `Male`, `Female`, `Other` |
| `age` | integer | ✅ | 13 – 99 |
| `activity_level` | string | ✅ | `Sedentary`, `Lightly Active`, `Moderately Active`, `Very Active`, `Super Active` |
| `meal_preference` | string | ✅ | `Veg`, `Non-Veg`, `Eggitarian`, `Vegan` |
| `cuisine_preference` | string | ✅ | e.g. `North Indian`, `South Indian`, `Continental` |
| `primary_goal` | string | ✅ | e.g. `Weight Loss`, `Muscle Gain`, `Maintenance` |
| `chronic_diseases` | array of strings | ❌ | e.g. `["Diabetes", "Hypertension"]` |
| `allergies` | array of strings | ❌ | e.g. `["Peanuts", "Gluten"]` |

**Response `200 OK`**
```json
{
  "success": true,
  "status_code": 200,
  "message": "7-day diet plan generated successfully.",
  "data": {
    "daily_calories_target": 1800,
    "generated_at": "2025-07-15T14:30:00+05:30",
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
            "image_prompt": "masala oats upma bowl Indian breakfast",
            "image_url": "/static/meal_images/masala_oats_upma.png"
          }
        ],
        "total_calories": 1380,
        "total_protein": 60,
        "total_carbs": 175,
        "total_fat": 34
      }
    ],
    "smart_swaps": [
      {
        "instead_of": "White Rice",
        "swap_with": "Brown Rice",
        "reason": "Lower glycemic index, more fiber"
      }
    ],
    "foods_to_avoid": [
      "Fried snacks like samosa, pakora, and vada"
    ],
    "ai_notes": "Plan tailored for weight loss. Stay hydrated with 2.5-3L water daily."
  }
}
```

**Error Responses**

| Status | When |
|---|---|
| `422 Unprocessable Entity` | Invalid field values (gender, activity_level, etc.) or bad AI JSON |
| `502 Bad Gateway` | Could not reach OpenRouter |
| `503 Service Unavailable` | `OPENROUTER_API_KEY` not set |

---

## How It Works

### 📸 Image → Nutrition

```
Upload Image → Resize & Compress (max 768px) → Base64 Encode
    → Gemini Vision Prompt → Parse & Sanitise JSON → Pydantic Validation
    → Return structured nutrition data
```

### 🥗 Diet Plan Generation

```
User Profile → Prompt Engineering → Gemini Text Model (8k tokens)
    → Parse & Sanitise JSON → Parallel Meal Image Generation (FLUX.1-schnell)
    → Return enriched 7-day plan
```

---

## Constraints & Limits

| Constraint | Value |
|---|---|
| Max image size | 5 MB |
| Supported image formats | JPEG, PNG, WEBP |
| Max image resolution (internal) | 768 × 768 px |
| Diet plan length | 7 days, 4 meals/day |
| Smart swaps | 5 |
| Foods to avoid | 6 |
| AI model token limit (meal) | 2 048 |
| AI model token limit (diet) | 8 000 |

---

## Security Notes

- `.env` is listed in `.gitignore` — API keys are never committed
- All request inputs are validated via Pydantic before reaching the AI model
- File type and size are checked before image processing
- `CORS` is open (`*`) by default — tighten `allow_origins` in production

---

## Future Improvements

- React / Streamlit frontend dashboard
- User authentication and JWT-based sessions
- Meal history tracking with a database (PostgreSQL / SQLite)
- Portion size estimation from image scale references
- Weekly nutrition summary and trend charts
- Integration with wearable fitness devices

---

## Author

Built by **Shashank Gurav**

---

*Give it a ⭐ on GitHub if you find it useful!*