"""
Curanova Gemini Conversational Metabolic Coach Service
======================================================
Interfaces with the Gemini API (Gemini 1.5 Flash / Pro) to transform
precise physical meal measurements (Calories, Protein, Carbs, Fats, Mass,
and Glycemic Speed) into high-agency nutritional coaching.

Provides:
- Real-time metabolic impact assessment
- Postprandial glucose management tips (e.g., for Jalebi, Malpua, Siddu)
- Complementary meal pairing for subsequent eating windows
- Adaptive daily target adjustments
"""

import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, Optional

COACH_SYSTEM_PROMPT = """
You are the Curanova Metabolic Coach — an elite, high-agency sports nutritionist and clinical dietary strategist.
You speak with razor-sharp clarity, encouragement, and actionable precision.
Your mission is to guide the user towards their daily macronutrient targets without judgment.

Rules for Coaching:
1. Deliver exact, high-agency advice based on the meal's verified macronutrient figures.
2. For high glycemic meals (sweets like Jalebi, Malpua, refined carbs), advise immediate walking/movement and protein/fiber pairing to flatten glucose spikes.
3. For dense regional meals (Siddu, Kofta with ghee/cream), evaluate the healthy fats and recommend lean protein sources for the next meal.
4. For high-fiber millet meals (Kali Roti, Bajra), commend the sustained energy release and encourage adequate hydration.
5. Keep guidance punchy, actionable, and structured into:
   - Immediate Summary
   - Metabolic Impact & Glucose Strategy
   - Next Meal Roadmap
"""


class GeminiCoachService:
    """
    Conversational AI Coach powered by Gemini API.
    Gracefully falls back to clinical heuristic coaching if API key is not configured.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    def generate_coach_advice(
        self,
        meal_data: Dict[str, Any],
        user_profile: Dict[str, Any],
        meal_slot: str = "Meal"
    ) -> Dict[str, Any]:
        """
        Generates tailored nutritional advice comparing the meal against
        the user's daily goals and current consumption.
        """
        # If API key is available, call Gemini endpoint
        if self.api_key:
            try:
                return self._call_gemini_api(meal_data, user_profile, meal_slot)
            except Exception as e:
                print(f"Gemini API call encountered an issue ({e}). Serving calibrated clinical coaching.")

        return self._generate_heuristic_coach_advice(meal_data, user_profile, meal_slot)

    def _call_gemini_api(
        self,
        meal_data: Dict[str, Any],
        user_profile: Dict[str, Any],
        meal_slot: str
    ) -> Dict[str, Any]:
        url = f"{self.api_url}?key={self.api_key}"

        prompt = f"""
Meal Data:
- Dish: {meal_data.get('matched_dish', meal_data.get('dish_name', 'Meal'))}
- Mass: {meal_data.get('mass_g', 0)}g
- Calories: {meal_data.get('calories', 0)} kcal
- Protein: {meal_data.get('protein_g', 0)}g
- Carbs: {meal_data.get('carbs_g', 0)}g
- Fat: {meal_data.get('fat_g', 0)}g
- Fiber: {meal_data.get('fiber_g', 0)}g
- Glycemic Speed: {meal_data.get('glycemic_speed', 'moderate')}
- Cooking Style: {meal_data.get('cooking_method', 'standard')}
- Signature Ingredients: {meal_data.get('signature_ingredients', [])}

User Targets & Status:
- Daily Goal: {user_profile.get('fitness_goal', 'lean_fitness')}
- Target Calories: {user_profile.get('daily_calorie_target', 2000)} kcal
- Target Protein: {user_profile.get('daily_protein_target_g', 130)}g
- Target Carbs: {user_profile.get('daily_carbs_target_g', 220)}g
- Target Fat: {user_profile.get('daily_fat_target_g', 60)}g
- Already Consumed Today (Prior to this meal): Calories: {user_profile.get('calories_consumed_today', 0)} kcal, Protein: {user_profile.get('protein_consumed_today', 0)}g
- Active Meal Slot: {meal_slot}

Provide structured coaching in JSON format with:
{{
  "headline": "<punchy 1-sentence assessment>",
  "glucose_strategy": "<actionable advice regarding blood sugar response and digestion>",
  "macro_pacing": "<how this meal affects remaining protein, carbs, and fat for the day>",
  "next_meal_recommendation": "<specific food items to focus on during next meal to balance targets>",
  "action_tip": "<1 immediate habit to do right now, e.g., brisk 10-minute walk or water>"
}}
"""

        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": COACH_SYSTEM_PROMPT + "\n\n" + prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "responseMimeType": "application/json"
            }
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=12) as response:
            result = json.loads(response.read().decode("utf-8"))
            candidate_text = result["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(candidate_text)

    def _generate_heuristic_coach_advice(
        self,
        meal_data: Dict[str, Any],
        user_profile: Dict[str, Any],
        meal_slot: str
    ) -> Dict[str, Any]:
        """
        Deterministic, clinical nutritional coaching logic that operates
        instantaneously without external network calls.
        """
        dish = meal_data.get("matched_dish", meal_data.get("dish_name", "Logged Meal"))
        calories = meal_data.get("calories", 0.0)
        protein = meal_data.get("protein_g", 0.0)
        carbs = meal_data.get("carbs_g", 0.0)
        fat = meal_data.get("fat_g", 0.0)
        glycemic = meal_data.get("glycemic_speed", "moderate")

        target_cals = user_profile.get("daily_calorie_target", 2000.0)
        target_p = user_profile.get("daily_protein_target_g", 130.0)
        consumed_cals = user_profile.get("calories_consumed_today", 0.0) + calories
        consumed_p = user_profile.get("protein_consumed_today", 0.0) + protein

        cals_remaining = max(0.0, target_cals - consumed_cals)
        p_remaining = max(0.0, target_p - consumed_p)

        # Context-specific advice based on regional meal dynamics
        if glycemic == "rapid_spike":
            headline = f"High glycemic density detected in {dish}. Manage your glucose curve."
            glucose_strategy = (
                "Rapidly absorbing simple sugars can trigger a sharp insulin surge followed by an energy dip. "
                "Take a 10-15 minute easy walk right now to channel glucose directly into skeletal muscle without an insulin spike."
            )
            next_meal = "Focus strictly on lean protein (tofu, grilled paneer, chicken breast) and cruciferous greens to stabilize metabolic rate."
            action_tip = "Drink 350ml water and take a 10-minute post-meal stroll."
        elif "siddu" in dish.lower() or "kofta" in dish.lower() or fat > 20.0:
            headline = f"Nutrient-dense meal with wholesome fats logged for {dish}."
            glucose_strategy = (
                "The presence of healthy fats (poppy seeds, walnuts, cream, or ghee) provides sustained gastric emptying, "
                "keeping you satiated for 4-5 hours."
            )
            next_meal = f"You have {round(p_remaining, 1)}g protein remaining today. Prioritize an ultra-lean protein source to hit your daily target."
            action_tip = "Allow 3-4 hours before your next snack to let digestion conclude smoothly."
        elif "roti" in dish.lower() or "kali" in dish.lower() or meal_data.get("fiber_g", 0.0) > 4.0:
            headline = f"Superb complex carbohydrates and dietary fiber in {dish}."
            glucose_strategy = (
                "Coarse grains like bajra/ragi provide slow-release glucose into the bloodstream, avoiding sudden cravings."
            )
            next_meal = f"Balance your daily macronutrient distribution with a high-protein dish (dal, paneer, or eggs) during dinner."
            action_tip = "Stay well hydrated to help dietary fiber promote optimal digestive transit."
        else:
            headline = f"{dish} successfully registered into your daily metabolic log."
            glucose_strategy = "Balanced macronutrient distribution supporting stable cellular energy."
            next_meal = f"You have {round(cals_remaining, 1)} kcal and {round(p_remaining, 1)}g protein remaining for today's target."
            action_tip = "Keep up the consistent tracking."

        return {
            "headline": headline,
            "glucose_strategy": glucose_strategy,
            "macro_pacing": f"Day progress: {round(consumed_cals, 1)}/{target_cals} kcal consumed. Remaining protein: {round(p_remaining, 1)}g.",
            "next_meal_recommendation": next_meal,
            "action_tip": action_tip,
            "dish_analyzed": dish,
            "meal_slot": meal_slot
        }


if __name__ == "__main__":
    coach = GeminiCoachService()

    # Test coaching for regional dishes
    user = {
        "fitness_goal": "lean_muscle_gain",
        "daily_calorie_target": 2200,
        "daily_protein_target_g": 140,
        "calories_consumed_today": 650,
        "protein_consumed_today": 45
    }

    test_meals = [
        {"matched_dish": "Kesar Jalebi", "calories": 335.2, "protein_g": 1.9, "carbs_g": 63.9, "fat_g": 8.0, "glycemic_speed": "rapid_spike"},
        {"matched_dish": "Himachali Siddu", "calories": 696.0, "protein_g": 18.9, "carbs_g": 87.6, "fat_g": 30.0, "glycemic_speed": "moderate_sustained"},
        {"matched_dish": "Bajra Kali Roti", "calories": 210.2, "protein_g": 6.2, "carbs_g": 40.5, "fat_g": 2.6, "fiber_g": 7.8, "glycemic_speed": "slow_sustained"}
    ]

    for m in test_meals:
        advice = coach.generate_coach_advice(m, user, meal_slot="Lunch")
        print("\n--- COACH ADVICE ---")
        print(f"Dish: {advice.get('dish_analyzed')}")
        print(f"Headline: {advice['headline']}")
        print(f"Glucose: {advice['glucose_strategy']}")
        print(f"Action: {advice['action_tip']}")
