"""
Curanova Unified Food & Macronutrient Physical Database
======================================================
Proprietary culinary and nutritional database synthesizing:
1. ICMR-NIN Indian Food Composition Tables (IFCT)
2. Indian Nutrient Databank (INDB) cooked preparations
3. Regional Culinary Archives (Himachali, Awadhi, Rajasthani, Gujarati, South Indian)
4. USDA FoodData Central Foundation foods

Key Invariant: Every food profile maps visual geometry to physical density (rho in g/cm3),
enabling true 3D spatial mass and macronutrient computation:
    Mass (g) = Volume (cm3) * rho (g/cm3)
    Thermodynamic Conservation: 4.0 * P + 4.0 * C + 9.0 * F ~= Calories
"""

import json
from typing import Dict, Any, List, Optional

# Comprehensive regional & international food knowledge matrix
FOOD_DATABASE: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------
    # Traditional & Regional Indian Dishes
    # -------------------------------------------------------------
    "himachali_siddu": {
        "canonical_name": "Himachali Siddu",
        "regional_aliases": ["siddu", "siddhu", "sidu", "himachali steamed bread"],
        "cuisine": "Himachali / North Indian Pahadi",
        "cooking_method": "fermented_steamed_with_ghee_dip",
        "density_g_cm3": 1.10,
        "calories_per_100g": 287.6,
        "protein_per_100g": 7.8,
        "carbs_per_100g": 36.2,
        "fat_per_100g": 12.4,
        "fiber_per_100g": 4.1,
        "glycemic_speed": "moderate_sustained",
        "typical_portion_g": 200.0,
        "signature_ingredients": ["whole_wheat", "poppy_seeds", "walnuts", "desi_ghee", "spices"]
    },
    "kesar_jalebi": {
        "canonical_name": "Kesar Jalebi",
        "regional_aliases": ["jalebi", "jilapi", "jilabi", "imarti", "zulbia"],
        "cuisine": "Pan-Indian Traditional Sweet",
        "cooking_method": "deep_fried_syrup_soaked",
        "density_g_cm3": 1.35,
        "calories_per_100g": 382.0,
        "protein_per_100g": 2.2,
        "carbs_per_100g": 72.8,
        "fat_per_100g": 9.1,
        "fiber_per_100g": 0.4,
        "glycemic_speed": "rapid_spike",
        "typical_portion_g": 80.0,
        "signature_ingredients": ["refined_flour", "sugar_syrup", "ghee", "saffron", "cardamom"]
    },
    "bajra_kali_roti": {
        "canonical_name": "Bajra Kali Roti",
        "regional_aliases": ["kali roti", "bajre ki roti", "bajra roti", "black millet roti", "sajje rotti"],
        "cuisine": "Rajasthani / Maharashtrian / Haryanvi",
        "cooking_method": "tawa_roasted_open_flame",
        "density_g_cm3": 0.65,
        "calories_per_100g": 232.0,
        "protein_per_100g": 6.8,
        "carbs_per_100g": 44.5,
        "fat_per_100g": 2.9,
        "fiber_per_100g": 8.5,
        "glycemic_speed": "slow_sustained",
        "typical_portion_g": 90.0,
        "signature_ingredients": ["pearl_millet_flour", "water", "pinch_of_salt"]
    },
    "malpua": {
        "canonical_name": "Malpua",
        "regional_aliases": ["malpua", "malpoa", "malpurde", "rabdi malpua"],
        "cuisine": "North & Eastern Indian Festive Sweet",
        "cooking_method": "deep_fried_syrup_soaked",
        "density_g_cm3": 1.25,
        "calories_per_100g": 348.0,
        "protein_per_100g": 4.6,
        "carbs_per_100g": 58.2,
        "fat_per_100g": 10.8,
        "fiber_per_100g": 0.8,
        "glycemic_speed": "rapid_spike",
        "typical_portion_g": 100.0,
        "signature_ingredients": ["semolina", "khoya", "fennel", "sugar_syrup", "desi_ghee"]
    },
    "malai_kofta": {
        "canonical_name": "Malai Kofta",
        "regional_aliases": ["kofta", "malai kofta", "paneer kofta", "shahi kofta"],
        "cuisine": "Mughlai / North Indian",
        "cooking_method": "fried_dumplings_in_cashew_cream_gravy",
        "density_g_cm3": 1.15,
        "calories_per_100g": 218.0,
        "protein_per_100g": 5.4,
        "carbs_per_100g": 14.2,
        "fat_per_100g": 15.6,
        "fiber_per_100g": 2.1,
        "glycemic_speed": "moderate",
        "typical_portion_g": 220.0,
        "signature_ingredients": ["paneer", "potato", "cashew_paste", "fresh_cream", "tomatoes"]
    },
    "lauki_kofta": {
        "canonical_name": "Lauki Kofta Curry",
        "regional_aliases": ["lauki kofta", "ghiya kofta", "doodhi kofta"],
        "cuisine": "North Indian Home Style",
        "cooking_method": "fried_gourd_dumplings_in_onion_tomato_gravy",
        "density_g_cm3": 1.04,
        "calories_per_100g": 142.0,
        "protein_per_100g": 3.8,
        "carbs_per_100g": 12.5,
        "fat_per_100g": 8.6,
        "fiber_per_100g": 2.4,
        "glycemic_speed": "moderate",
        "typical_portion_g": 200.0,
        "signature_ingredients": ["bottle_gourd", "besan", "onion", "tomato", "spices"]
    },
    "dal_makhani": {
        "canonical_name": "Dal Makhani",
        "regional_aliases": ["dal makhani", "maa ki dal", "black lentil curry"],
        "cuisine": "Punjabi / North Indian",
        "cooking_method": "slow_simmered_with_butter_and_cream",
        "density_g_cm3": 1.08,
        "calories_per_100g": 138.0,
        "protein_per_100g": 5.2,
        "carbs_per_100g": 15.6,
        "fat_per_100g": 6.3,
        "fiber_per_100g": 4.2,
        "glycemic_speed": "slow_sustained",
        "typical_portion_g": 200.0,
        "signature_ingredients": ["black_urad_dal", "rajma", "white_butter", "cream", "tomatoes"]
    },
    "paneer_tikka": {
        "canonical_name": "Paneer Tikka",
        "regional_aliases": ["paneer tikka", "tandoori paneer", "grilled paneer"],
        "cuisine": "North Indian Tandoori",
        "cooking_method": "tandoor_clay_oven_roasted",
        "density_g_cm3": 0.88,
        "calories_per_100g": 245.0,
        "protein_per_100g": 15.2,
        "carbs_per_100g": 6.4,
        "fat_per_100g": 17.6,
        "fiber_per_100g": 1.8,
        "glycemic_speed": "low",
        "typical_portion_g": 150.0,
        "signature_ingredients": ["fresh_paneer", "hung_curd", "mustard_oil", "capsicum", "onion"]
    },
    "dal_baati_churma": {
        "canonical_name": "Dal Baati Churma",
        "regional_aliases": ["dal baati", "baati", "churma"],
        "cuisine": "Rajasthani",
        "cooking_method": "baked_wheat_balls_soaked_in_ghee",
        "density_g_cm3": 1.18,
        "calories_per_100g": 312.0,
        "protein_per_100g": 7.4,
        "carbs_per_100g": 42.8,
        "fat_per_100g": 12.5,
        "fiber_per_100g": 4.5,
        "glycemic_speed": "moderate",
        "typical_portion_g": 250.0,
        "signature_ingredients": ["coarse_wheat_flour", "panchmel_dal", "pure_desi_ghee", "jaggery"]
    },
    "hyderabadi_biryani_chicken": {
        "canonical_name": "Hyderabadi Chicken Biryani",
        "regional_aliases": ["chicken biryani", "dum biryani", "hyderabadi biryani"],
        "cuisine": "Deccani / Hyderabadi",
        "cooking_method": "dum_sealed_slow_steam_cook",
        "density_g_cm3": 0.85,
        "calories_per_100g": 185.0,
        "protein_per_100g": 9.8,
        "carbs_per_100g": 22.4,
        "fat_per_100g": 6.1,
        "fiber_per_100g": 1.2,
        "glycemic_speed": "moderate",
        "typical_portion_g": 350.0,
        "signature_ingredients": ["aged_basmati_rice", "chicken", "curd", "fried_onions", "ghee", "saffron"]
    },
    "masala_dosa": {
        "canonical_name": "Masala Dosa",
        "regional_aliases": ["dosa", "masala dosa", "crispy dosa"],
        "cuisine": "South Indian",
        "cooking_method": "fermented_batter_pan_roasted_with_filling",
        "density_g_cm3": 0.72,
        "calories_per_100g": 165.0,
        "protein_per_100g": 3.9,
        "carbs_per_100g": 27.8,
        "fat_per_100g": 4.2,
        "fiber_per_100g": 2.1,
        "glycemic_speed": "moderate",
        "typical_portion_g": 180.0,
        "signature_ingredients": ["fermented_rice_urad_batter", "spiced_potato_masala", "sesame_oil"]
    },
    "idli_sambar": {
        "canonical_name": "Idli with Sambar",
        "regional_aliases": ["idli", "idli sambar", "steamed rice cake"],
        "cuisine": "South Indian",
        "cooking_method": "steam_cooked_fermented_cakes",
        "density_g_cm3": 0.82,
        "calories_per_100g": 108.8,
        "protein_per_100g": 3.8,
        "carbs_per_100g": 21.6,
        "fat_per_100g": 0.8,
        "fiber_per_100g": 2.2,
        "glycemic_speed": "moderate_low",
        "typical_portion_g": 200.0,
        "signature_ingredients": ["fermented_rice_dal", "toor_dal_sambar", "tamarind", "vegetables"]
    },
    "chole_bhature": {
        "canonical_name": "Chole Bhature",
        "regional_aliases": ["chole bhature", "chana bhatura", "amritsari chole"],
        "cuisine": "Punjabi / North Indian",
        "cooking_method": "deep_fried_bread_with_spicy_chickpea_gravy",
        "density_g_cm3": 0.95,
        "calories_per_100g": 268.0,
        "protein_per_100g": 6.8,
        "carbs_per_100g": 35.4,
        "fat_per_100g": 11.2,
        "fiber_per_100g": 4.8,
        "glycemic_speed": "moderate_high",
        "typical_portion_g": 280.0,
        "signature_ingredients": ["kabuli_chana", "refined_wheat_flour", "oil", "pomegranate_seed_spices"]
    },
    "rajma_chawal": {
        "canonical_name": "Rajma Chawal",
        "regional_aliases": ["rajma chawal", "rajma rice", "red kidney beans with rice"],
        "cuisine": "North Indian Comfort Classic",
        "cooking_method": "pressure_cooked_legume_gravy_over_steamed_rice",
        "density_g_cm3": 0.92,
        "calories_per_100g": 145.0,
        "protein_per_100g": 4.9,
        "carbs_per_100g": 26.2,
        "fat_per_100g": 2.2,
        "fiber_per_100g": 3.6,
        "glycemic_speed": "slow_sustained",
        "typical_portion_g": 300.0,
        "signature_ingredients": ["red_kidney_beans", "steamed_basmati_rice", "tomato_gravy", "ginger_garlic"]
    },
    "poha": {
        "canonical_name": "Kanda Poha",
        "regional_aliases": ["poha", "kanda poha", "flattened rice"],
        "cuisine": "Maharashtrian / Central Indian",
        "cooking_method": "tempered_steam_softened_flattened_rice",
        "density_g_cm3": 0.68,
        "calories_per_100g": 158.0,
        "protein_per_100g": 3.2,
        "carbs_per_100g": 28.5,
        "fat_per_100g": 3.6,
        "fiber_per_100g": 1.8,
        "glycemic_speed": "moderate",
        "typical_portion_g": 160.0,
        "signature_ingredients": ["flattened_rice", "roasted_peanuts", "onions", "mustard_seeds", "turmeric"]
    },
    "gulab_jamun": {
        "canonical_name": "Gulab Jamun",
        "regional_aliases": ["gulab jamun", "jamun", "pantua"],
        "cuisine": "Pan-Indian Traditional Sweet",
        "cooking_method": "deep_fried_khoya_balls_in_rose_syrup",
        "density_g_cm3": 1.28,
        "calories_per_100g": 325.0,
        "protein_per_100g": 4.8,
        "carbs_per_100g": 52.4,
        "fat_per_100g": 10.9,
        "fiber_per_100g": 0.2,
        "glycemic_speed": "rapid_spike",
        "typical_portion_g": 80.0,
        "signature_ingredients": ["khoya_mawa", "chenna", "sugar_syrup", "rose_water", "cardamom"]
    },

    # -------------------------------------------------------------
    # Global & Whole-Food Nutrition Staples (USDA Aligned)
    # -------------------------------------------------------------
    "grilled_chicken_breast": {
        "canonical_name": "Grilled Chicken Breast",
        "regional_aliases": ["chicken breast", "grilled chicken", "skinless chicken breast"],
        "cuisine": "Global Athletic Staple",
        "cooking_method": "dry_heat_grilling",
        "density_g_cm3": 1.05,
        "calories_per_100g": 156.4,
        "protein_per_100g": 31.0,
        "carbs_per_100g": 0.0,
        "fat_per_100g": 3.6,
        "fiber_per_100g": 0.0,
        "glycemic_speed": "zero_glycemic",
        "typical_portion_g": 180.0,
        "signature_ingredients": ["chicken_breast", "olive_oil_spray", "black_pepper", "herbs"]
    },
    "steamed_white_rice": {
        "canonical_name": "Steamed White Rice",
        "regional_aliases": ["white rice", "boiled rice", "steamed basmati", "chawal"],
        "cuisine": "Global Staple",
        "cooking_method": "boiling_steaming",
        "density_g_cm3": 0.82,
        "calories_per_100g": 126.3,
        "protein_per_100g": 2.7,
        "carbs_per_100g": 28.2,
        "fat_per_100g": 0.3,
        "fiber_per_100g": 0.4,
        "glycemic_speed": "rapid_moderate",
        "typical_portion_g": 150.0,
        "signature_ingredients": ["milled_rice", "water"]
    },
    "cooked_brown_rice": {
        "canonical_name": "Steamed Brown Rice",
        "regional_aliases": ["brown rice", "whole grain rice"],
        "cuisine": "Global Staple",
        "cooking_method": "boiling_steaming",
        "density_g_cm3": 0.84,
        "calories_per_100g": 111.0,
        "protein_per_100g": 2.6,
        "carbs_per_100g": 23.0,
        "fat_per_100g": 0.9,
        "fiber_per_100g": 1.8,
        "glycemic_speed": "slow_sustained",
        "typical_portion_g": 150.0,
        "signature_ingredients": ["whole_grain_brown_rice", "water"]
    },
    "baked_salmon_fillet": {
        "canonical_name": "Baked Atlantic Salmon",
        "regional_aliases": ["salmon", "baked salmon", "grilled salmon"],
        "cuisine": "Global Seafood",
        "cooking_method": "oven_baking",
        "density_g_cm3": 1.02,
        "calories_per_100g": 205.0,
        "protein_per_100g": 22.0,
        "carbs_per_100g": 0.0,
        "fat_per_100g": 13.0,
        "fiber_per_100g": 0.0,
        "glycemic_speed": "zero_glycemic",
        "typical_portion_g": 170.0,
        "signature_ingredients": ["atlantic_salmon", "omega_3_fatty_acids", "lemon", "herbs"]
    },
    "whole_eggs_scrambled": {
        "canonical_name": "Scrambled Whole Eggs",
        "regional_aliases": ["scrambled eggs", "anda bhurji", "omelette"],
        "cuisine": "Global Breakfast Staple",
        "cooking_method": "pan_scrambled_with_light_butter",
        "density_g_cm3": 0.98,
        "calories_per_100g": 149.0,
        "protein_per_100g": 10.5,
        "carbs_per_100g": 1.6,
        "fat_per_100g": 11.0,
        "fiber_per_100g": 0.0,
        "glycemic_speed": "zero_glycemic",
        "typical_portion_g": 120.0,
        "signature_ingredients": ["whole_eggs", "butter_or_oil", "salt"]
    },
    "greek_yogurt_plain": {
        "canonical_name": "Plain Greek Yogurt",
        "regional_aliases": ["greek yogurt", "strained dahi", "hung curd"],
        "cuisine": "Global Dairy",
        "cooking_method": "strained_fermentation",
        "density_g_cm3": 1.06,
        "calories_per_100g": 99.4,
        "protein_per_100g": 10.0,
        "carbs_per_100g": 3.6,
        "fat_per_100g": 5.0,
        "fiber_per_100g": 0.0,
        "glycemic_speed": "low",
        "typical_portion_g": 170.0,
        "signature_ingredients": ["pasteurized_milk", "live_active_cultures"]
    },
    "rolled_oats_cooked": {
        "canonical_name": "Cooked Rolled Oatmeal",
        "regional_aliases": ["oatmeal", "porridge", "oats"],
        "cuisine": "Global Breakfast",
        "cooking_method": "water_or_milk_simmered",
        "density_g_cm3": 0.94,
        "calories_per_100g": 71.0,
        "protein_per_100g": 2.5,
        "carbs_per_100g": 12.0,
        "fat_per_100g": 1.5,
        "fiber_per_100g": 1.7,
        "glycemic_speed": "slow_sustained",
        "typical_portion_g": 220.0,
        "signature_ingredients": ["rolled_oats", "water", "pinch_of_cinnamon"]
    }
}


class CuranovaFoodDatabase:
    """
    Query and matching engine for Curanova's physical food & nutritional database.
    Supports exact key retrieval, fuzzy name matching, and 3D physical volume
    mass & macronutrient calculations.
    """
    def __init__(self, data: Optional[Dict[str, Dict[str, Any]]] = None):
        self.db = data or FOOD_DATABASE

    def get_entry(self, dish_key: str) -> Optional[Dict[str, Any]]:
        return self.db.get(dish_key.lower().strip())

    def match_dish(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Matches a natural language dish string to the closest database entry
        using substring and token overlap.
        """
        clean_q = query.lower().strip()
        
        # 1. Exact canonical or key match
        if clean_q in self.db:
            return self.db[clean_q]

        for key, item in self.db.items():
            if clean_q == item["canonical_name"].lower():
                return item
            for alias in item["regional_aliases"]:
                if clean_q == alias.lower():
                    return item

        # 2. Token overlap search
        q_tokens = set(clean_q.replace("-", " ").replace("_", " ").split())
        best_match = None
        best_score = 0

        for key, item in self.db.items():
            candidate_tokens = set(key.split("_"))
            candidate_tokens.update(item["canonical_name"].lower().split())
            for alias in item["regional_aliases"]:
                candidate_tokens.update(alias.lower().split())

            overlap = len(q_tokens.intersection(candidate_tokens))
            if overlap > best_score:
                best_score = overlap
                best_match = item

        if best_score > 0:
            return best_match

        return None

    def calculate_physical_macros(
        self,
        dish_identifier: str,
        volume_cm3: float
    ) -> Dict[str, Any]:
        """
        Given a dish identifier and a 3D integrated volume (cm3) from DMP-Net,
        computes the physical mass (grams) and exact macronutrients under
        thermodynamic energy conservation.
        """
        entry = self.match_dish(dish_identifier)
        if not entry:
            # Fallback average density and macro distribution
            rho = 0.95
            mass_g = volume_cm3 * rho
            p = mass_g * 0.08
            c = mass_g * 0.22
            f = mass_g * 0.07
            kcal = 4.0 * p + 4.0 * c + 9.0 * f
            return {
                "matched_dish": "Generic Plate Item",
                "volume_cm3": round(volume_cm3, 1),
                "density_g_cm3": rho,
                "mass_g": round(mass_g, 1),
                "calories": round(kcal, 1),
                "protein_g": round(p, 1),
                "carbs_g": round(c, 1),
                "fat_g": round(f, 1),
                "fiber_g": round(mass_g * 0.02, 1),
                "thermodynamic_conserved": True
            }

        rho = entry["density_g_cm3"]
        mass_g = volume_cm3 * rho

        # Compute from standardized 100g baseline
        factor = mass_g / 100.0
        p = round(entry["protein_per_100g"] * factor, 1)
        c = round(entry["carbs_per_100g"] * factor, 1)
        f = round(entry["fat_per_100g"] * factor, 1)
        fiber = round(entry["fiber_per_100g"] * factor, 1)

        # Enforce exact thermodynamic energy balance
        thermo_kcal = round(4.0 * p + 4.0 * c + 9.0 * f, 1)

        return {
            "matched_dish": entry["canonical_name"],
            "cuisine": entry["cuisine"],
            "cooking_method": entry["cooking_method"],
            "glycemic_speed": entry["glycemic_speed"],
            "signature_ingredients": entry["signature_ingredients"],
            "volume_cm3": round(volume_cm3, 1),
            "density_g_cm3": rho,
            "mass_g": round(mass_g, 1),
            "calories": thermo_kcal,
            "protein_g": p,
            "carbs_g": c,
            "fat_g": f,
            "fiber_g": fiber,
            "thermodynamic_conserved": True
        }

    def export_json(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.db, f, indent=2, ensure_ascii=False)
        print(f"Exported {len(self.db)} dish profiles to {filepath}")


if __name__ == "__main__":
    db = CuranovaFoodDatabase()
    # Test specific cultural dishes
    for test_dish, test_vol in [
        ("himachali siddu", 220.0),
        ("kesar jalebi", 65.0),
        ("kali roti", 140.0),
        ("malpua", 85.0),
        ("malai kofta", 200.0),
        ("grilled chicken", 170.0)
    ]:
        res = db.calculate_physical_macros(test_dish, test_vol)
        print(f"[{res['matched_dish']}] Vol: {res['volume_cm3']}cm3 -> Mass: {res['mass_g']}g | Calories: {res['calories']} kcal (P: {res['protein_g']}g, C: {res['carbs_g']}g, F: {res['fat_g']}g)")
