"""
Curanova Macro Vision & Cultural Pipeline Test Suite
===================================================
Automated verification of:
1. Curanova Food Database integrity & coverage (Regional Indian + Global staples)
2. Thermodynamic Energy Conservation: 4P + 4C + 9F ~= Calories
3. Physical 3D Volume to Mass translation: Mass = Volume * Density
4. Cultural VLM dish and ingredient decomposition
5. Gemini Coach metabolic guidance generation
"""

import sys
import unittest
from PIL import Image

from curanova_food_db import CuranovaFoodDatabase, FOOD_DATABASE
from cultural_vlm_engine import CulturalVLMEngine
from gemini_coach_service import GeminiCoachService


class TestCuranovaPipeline(unittest.TestCase):
    def setUp(self):
        self.db = CuranovaFoodDatabase()
        self.vlm = CulturalVLMEngine()
        self.coach = GeminiCoachService()

    def test_database_regional_dishes_exist(self):
        """Verifies that regional dishes specified by user are accurately registered."""
        required_dishes = [
            "himachali_siddu",
            "kesar_jalebi",
            "bajra_kali_roti",
            "malpua",
            "malai_kofta",
            "lauki_kofta",
            "grilled_chicken_breast",
            "steamed_white_rice"
        ]
        for dish_key in required_dishes:
            entry = self.db.get_entry(dish_key)
            self.assertIsNotNone(entry, f"Dish key '{dish_key}' must exist in database")
            self.assertGreater(entry["density_g_cm3"], 0.0)
            self.assertGreater(entry["calories_per_100g"], 0.0)
            self.assertGreaterEqual(entry["protein_per_100g"], 0.0)
            self.assertGreaterEqual(entry["carbs_per_100g"], 0.0)
            self.assertGreaterEqual(entry["fat_per_100g"], 0.0)

    def test_thermodynamic_conservation(self):
        """
        Verifies that every single dish in the database satisfies
        thermodynamic conservation of macronutrient energy:
        4 * Protein + 4 * Carbs + 9 * Fat ~= Calories
        """
        for key, dish in FOOD_DATABASE.items():
            p = dish["protein_per_100g"]
            c = dish["carbs_per_100g"]
            f = dish["fat_per_100g"]
            stated_cal = dish["calories_per_100g"]
            thermo_cal = 4.0 * p + 4.0 * c + 9.0 * f

            discrepancy = abs(stated_cal - thermo_cal)
            self.assertLessEqual(
                discrepancy, 5.0,
                f"Dish '{key}' violates thermodynamic conservation: Stated {stated_cal} vs Thermo {thermo_cal}"
            )

    def test_volume_to_mass_translation(self):
        """
        Tests that spatial 3D volume (cm3) correctly scales mass and macros
        using physical density constants (rho).
        """
        # Test Siddu (rho = 1.10)
        res_siddu = self.db.calculate_physical_macros("himachali siddu", volume_cm3=200.0)
        expected_mass = 200.0 * 1.10
        self.assertAlmostEqual(res_siddu["mass_g"], expected_mass, places=1)
        self.assertTrue(res_siddu["thermodynamic_conserved"])

        # Test Jalebi (rho = 1.35)
        res_jalebi = self.db.calculate_physical_macros("kesar jalebi", volume_cm3=100.0)
        expected_jalebi_mass = 100.0 * 1.35
        self.assertAlmostEqual(res_jalebi["mass_g"], expected_jalebi_mass, places=1)

        # Test Kali Roti (rho = 0.65)
        res_roti = self.db.calculate_physical_macros("kali roti", volume_cm3=150.0)
        expected_roti_mass = 150.0 * 0.65
        self.assertAlmostEqual(res_roti["mass_g"], expected_roti_mass, places=1)

    def test_fuzzy_cultural_matching(self):
        """Verifies natural language query matching to regional dish keys."""
        test_queries = [
            ("siddhu", "Himachali Siddu"),
            ("jilapi", "Kesar Jalebi"),
            ("bajre ki roti", "Bajra Kali Roti"),
            ("malpurde", "Malpua"),
            ("paneer kofta", "Malai Kofta")
        ]
        for query, expected_canonical in test_queries:
            matched = self.db.match_dish(query)
            self.assertIsNotNone(matched, f"Query '{query}' should resolve to a dish")
            self.assertEqual(matched["canonical_name"], expected_canonical)

    def test_cultural_vlm_analysis(self):
        """Verifies that Cultural VLM engine returns structured culinary payloads."""
        dummy_img = Image.new("RGB", (224, 224), color=(200, 160, 100))
        analysis = self.vlm.analyze_image(dummy_img, text_context="himachali siddu")

        self.assertIn("primary_dish", analysis)
        self.assertIn("density_g_cm3", analysis)
        self.assertIn("macros_per_100g", analysis)
        self.assertEqual(analysis["primary_dish"], "Himachali Siddu")
        self.assertEqual(analysis["density_g_cm3"], 1.10)

    def test_gemini_coach_guidance(self):
        """Verifies metabolic coaching generation for both sweets and balanced meals."""
        user = {
            "fitness_goal": "fat_loss",
            "daily_calorie_target": 2000,
            "daily_protein_target_g": 130,
            "calories_consumed_today": 700,
            "protein_consumed_today": 40
        }

        # Sweet: Kesar Jalebi
        meal_jalebi = {
            "matched_dish": "Kesar Jalebi",
            "calories": 300.0,
            "protein_g": 2.0,
            "carbs_g": 60.0,
            "fat_g": 6.0,
            "glycemic_speed": "rapid_spike"
        }
        advice_jalebi = self.coach.generate_coach_advice(meal_jalebi, user)
        self.assertIn("glucose", advice_jalebi["headline"].lower() + advice_jalebi["glucose_strategy"].lower())
        self.assertIn("walk", advice_jalebi["action_tip"].lower() + advice_jalebi["glucose_strategy"].lower())

        # Dense: Himachali Siddu
        meal_siddu = {
            "matched_dish": "Himachali Siddu",
            "calories": 600.0,
            "protein_g": 16.0,
            "carbs_g": 75.0,
            "fat_g": 26.0,
            "glycemic_speed": "moderate_sustained"
        }
        advice_siddu = self.coach.generate_coach_advice(meal_siddu, user)
        self.assertIn("fat", advice_siddu["headline"].lower() + advice_siddu["glucose_strategy"].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
