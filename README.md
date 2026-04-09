# 🍱 Food Vision API

AI-powered **Indian meal nutrition analysis + smart diet recommendation system** built with **FastAPI** and powered by **Gemini 2.5 Flash Lite via OpenRouter**.

---

## 🚀 Features

### 🥗 Meal Analysis (Computer Vision)

* Upload food images
* Detect multiple food items
* Get:

  * Calories
  * Protein, Carbs, Fat
  * Macro distribution
* Handles **Indian meals accurately**
* Ultra-robust JSON parsing (fixes model errors automatically)

---

### 🧠 Smart Diet Recommendation

* Generates **7-day personalized diet plan**
* Based on:

  * Age, gender
  * Activity level
  * Cuisine preference
  * Health goals
* Includes:

  * Daily meals (Breakfast, Lunch, Snack, Dinner)
  * Smart swaps
  * Foods to avoid

---

## 🏗️ Tech Stack

* **Backend:** FastAPI
* **AI Model:** Gemini 2.5 Flash Lite (via OpenRouter)
* **Image Processing:** Pillow (PIL)
* **Validation:** Pydantic
* **Environment:** Python

---

## 📂 Project Structure

```
.
├── config.py          # API config + OpenRouter client
├── main.py            # FastAPI app (meal analysis)
├── diet.py            # Diet recommendation system
├── .env               # API keys (ignored)
├── venv/              # Virtual environment (ignored)
└── README.md
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone <your-repo-url>
cd food-vision
```

---

### 2️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Add environment variables

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_api_key_here
```

---

### 5️⃣ Run the server

```bash
uvicorn main:app --reload
```

---

## 🌐 API Endpoints

### 🔹 Health Check

```
GET /health
```

---

### 🔹 Meal Analysis

```
POST /analyze-meal
```

**Input:** Image (JPEG/PNG/WEBP)
**Output:** Nutrition breakdown + items

---

### 🔹 Diet Recommendation

```
POST /diet/recommend-diet
```

**Sample Request:**

```json
{
  "gender": "Male",
  "age": 28,
  "activity_level": "Moderately Active",
  "meal_preference": "Veg",
  "cuisine_preference": "North Indian",
  "primary_goal": "Weight Loss",
  "chronic_diseases": [],
  "allergies": []
}
```

---

## 🧠 How It Works

### 📸 Image → Nutrition

1. Image resized & compressed
2. Converted to base64
3. Sent to Gemini Vision model
4. AI returns structured JSON
5. Backend cleans & validates response 

---

### 🥗 Diet Plan Generation

1. User profile → prompt engineering
2. AI generates structured weekly plan
3. JSON cleaned & validated 

---

## ⚠️ Constraints

* Max image size: **5MB**
* Supported formats: **JPEG, PNG, WEBP**
* Requires valid OpenRouter API key

---

## 🔒 Security Notes

* `.env` is ignored via `.gitignore`
* API keys are never exposed
* Input validation via Pydantic

---

## 🧪 Example Use Cases

* Fitness apps
* Diet planning tools
* Health startups
* AI nutrition assistants

---

## 📌 Future Improvements

* Frontend dashboard (React / Streamlit)
* User authentication
* Meal history tracking
* AI-based portion size estimation
* Integration with wearable devices

---

## 👨‍💻 Author

Built by **Shashank Gurav**

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share!

---
