"""
Curanova Semantic Culinary Ontology (Auto-Ontology Pattern)
==========================================================
Inspired by NVIDIA/auto-ontology design principles:
Builds a semantic, graph-based reasoning layer over culinary dishes,
decomposing complex meals into distinct physical subcomponents,
constituent ingredients, and individual density contributions.

Eliminates monolithic guessing by computing composite physical properties:
    rho_composite = sum(weight_fraction_i * rho_i)
    Nutrients_composite = sum(weight_fraction_i * Nutrients_i)
"""

from typing import Dict, Any, List, Optional


class CulinaryComponent:
    """Represents a discrete physical constituent of a prepared dish."""
    def __init__(
        self,
        name: str,
        weight_percentage: float,
        density_g_cm3: float,
        calories_per_100g: float,
        protein_per_100g: float,
        carbs_per_100g: float,
        fat_per_100g: float,
        fiber_per_100g: float,
        ingredients: List[str]
    ):
        self.name = name
        self.weight_fraction = weight_percentage / 100.0
        self.density_g_cm3 = density_g_cm3
        self.calories_per_100g = calories_per_100g
        self.protein_per_100g = protein_per_100g
        self.carbs_per_100g = carbs_per_100g
        self.fat_per_100g = fat_per_100g
        self.fiber_per_100g = fiber_per_100g
        self.ingredients = ingredients


class DishOntologyNode:
    """Hierarchical graph node representing a complex cultural dish."""
    def __init__(
        self,
        dish_id: str,
        canonical_name: str,
        cuisine_family: str,
        glycemic_speed: str,
        components: List[CulinaryComponent]
    ):
        self.dish_id = dish_id
        self.canonical_name = canonical_name
        self.cuisine_family = cuisine_family
        self.glycemic_speed = glycemic_speed
        self.components = components

    @property
    def composite_density(self) -> float:
        """Computes weighted physical density rho across all subcomponents."""
        return sum(c.weight_fraction * c.density_g_cm3 for c in self.components)

    @property
    def composite_macros_per_100g(self) -> Dict[str, float]:
        """Synthesizes composite macronutrients per 100g from subcomponents."""
        cals = sum(c.weight_fraction * c.calories_per_100g for c in self.components)
        p = sum(c.weight_fraction * c.protein_per_100g for c in self.components)
        c = sum(c.weight_fraction * c.carbs_per_100g for c in self.components)
        f = sum(c.weight_fraction * c.fat_per_100g for c in self.components)
        fib = sum(c.weight_fraction * c.fiber_per_100g for c in self.components)

        # Enforce exact thermodynamic energy balance
        thermo_kcal = 4.0 * p + 4.0 * c + 9.0 * f
        return {
            "calories": round(thermo_kcal, 1),
            "protein_g": round(p, 1),
            "carbs_g": round(c, 1),
            "fat_g": round(f, 1),
            "fiber_g": round(fib, 1)
        }

    def explain_decomposition(self, total_mass_g: float) -> Dict[str, Any]:
        """Provides full explainability for each subcomponent on the plate."""
        breakdown = []
        for c in self.components:
            comp_mass = round(total_mass_g * c.weight_fraction, 1)
            breakdown.append({
                "subcomponent": c.name,
                "weight_fraction_pct": round(c.weight_fraction * 100, 1),
                "mass_g": comp_mass,
                "density_g_cm3": c.density_g_cm3,
                "ingredients": c.ingredients
            })
        return {
            "dish": self.canonical_name,
            "total_mass_g": round(total_mass_g, 1),
            "effective_density": round(self.composite_density, 2),
            "subcomponents": breakdown
        }


# Curated Semantic Graph Registry
ONTOLOGY_REGISTRY: Dict[str, DishOntologyNode] = {
    "himachali_siddu": DishOntologyNode(
        dish_id="himachali_siddu",
        canonical_name="Himachali Siddu",
        cuisine_family="Pahadi / Himalayan",
        glycemic_speed="moderate_sustained",
        components=[
            CulinaryComponent(
                name="Outer Fermented Casing",
                weight_percentage=55.0,
                density_g_cm3=0.98,
                calories_per_100g=220.0,
                protein_per_100g=7.2,
                carbs_per_100g=42.0,
                fat_per_100g=1.8,
                fiber_per_100g=3.2,
                ingredients=["whole_wheat_flour", "yeast", "warm_water"]
            ),
            CulinaryComponent(
                name="Poppy-Walnut Stuffing",
                weight_percentage=35.0,
                density_g_cm3=1.22,
                calories_per_100g=430.0,
                protein_per_100g=12.0,
                carbs_per_100g=28.0,
                fat_per_100g=31.0,
                fiber_per_100g=6.5,
                ingredients=["crushed_poppy_seeds", "chopped_walnuts", "coriander", "spices"]
            ),
            CulinaryComponent(
                name="Desi Ghee Dip Finishing",
                weight_percentage=10.0,
                density_g_cm3=0.92,
                calories_per_100g=890.0,
                protein_per_100g=0.0,
                carbs_per_100g=0.0,
                fat_per_100g=99.0,
                fiber_per_100g=0.0,
                ingredients=["melted_desi_ghee"]
            )
        ]
    ),
    "kesar_jalebi": DishOntologyNode(
        dish_id="kesar_jalebi",
        canonical_name="Kesar Jalebi",
        cuisine_family="Traditional Indian Confectionery",
        glycemic_speed="rapid_spike",
        components=[
            CulinaryComponent(
                name="Fried Flour Matrix",
                weight_percentage=45.0,
                density_g_cm3=1.15,
                calories_per_100g=420.0,
                protein_per_100g=4.8,
                carbs_per_100g=52.0,
                fat_per_100g=20.0,
                fiber_per_100g=0.8,
                ingredients=["fermented_maida", "cornstarch", "ghee"]
            ),
            CulinaryComponent(
                name="Saffron Sugar Syrup Core",
                weight_percentage=55.0,
                density_g_cm3=1.45,
                calories_per_100g=350.0,
                protein_per_100g=0.0,
                carbs_per_100g=88.0,
                fat_per_100g=0.0,
                fiber_per_100g=0.0,
                ingredients=["sugar_syrup_chashni", "saffron", "cardamom"]
            )
        ]
    ),
    "malai_kofta": DishOntologyNode(
        dish_id="malai_kofta",
        canonical_name="Malai Kofta",
        cuisine_family="Mughlai / North Indian",
        glycemic_speed="moderate",
        components=[
            CulinaryComponent(
                name="Fried Dumpling Core",
                weight_percentage=40.0,
                density_g_cm3=1.12,
                calories_per_100g=260.0,
                protein_per_100g=9.5,
                carbs_per_100g=16.0,
                fat_per_100g=18.0,
                fiber_per_100g=1.4,
                ingredients=["fresh_paneer", "potato", "raisins", "cashew_nuts"]
            ),
            CulinaryComponent(
                name="Rich Cashew-Cream Gravy",
                weight_percentage=60.0,
                density_g_cm3=1.18,
                calories_per_100g=190.0,
                protein_per_100g=2.8,
                carbs_per_100g=13.0,
                fat_per_100g=14.0,
                fiber_per_100g=2.5,
                ingredients=["cashew_paste", "melon_seed_paste", "fresh_cream", "pureed_tomatoes"]
            )
        ]
    ),
    "bajra_kali_roti": DishOntologyNode(
        dish_id="bajra_kali_roti",
        canonical_name="Bajra Kali Roti",
        cuisine_family="Rajasthani / Western Millet",
        glycemic_speed="slow_sustained",
        components=[
            CulinaryComponent(
                name="Coarse Pearl Millet Flatbread",
                weight_percentage=100.0,
                density_g_cm3=0.65,
                calories_per_100g=232.0,
                protein_per_100g=6.8,
                carbs_per_100g=44.5,
                fat_per_100g=2.9,
                fiber_per_100g=8.5,
                ingredients=["whole_pearl_millet_flour", "water"]
            )
        ]
    ),
    "malpua": DishOntologyNode(
        dish_id="malpua",
        canonical_name="Malpua",
        cuisine_family="Festive Sweet",
        glycemic_speed="rapid_spike",
        components=[
            CulinaryComponent(
                name="Fried Khoya Pancake",
                weight_percentage=50.0,
                density_g_cm3=1.18,
                calories_per_100g=380.0,
                protein_per_100g=8.2,
                carbs_per_100g=36.0,
                fat_per_100g=21.0,
                fiber_per_100g=1.2,
                ingredients=["khoya_mawa", "semolina", "fennel", "ghee"]
            ),
            CulinaryComponent(
                name="Sugar Syrup Infusion",
                weight_percentage=50.0,
                density_g_cm3=1.40,
                calories_per_100g=320.0,
                protein_per_100g=0.0,
                carbs_per_100g=80.0,
                fat_per_100g=0.0,
                fiber_per_100g=0.0,
                ingredients=["sugar_syrup", "cardamom"]
            )
        ]
    )
}


class CuranovaOntologyEngine:
    """
    Query interface for the hierarchical culinary ontology graph.
    """
    def __init__(self, registry: Optional[Dict[str, DishOntologyNode]] = None):
        self.registry = registry or ONTOLOGY_REGISTRY

    def get_node(self, dish_key: str) -> Optional[DishOntologyNode]:
        clean = dish_key.lower().replace(" ", "_").replace("-", "_")
        for key, node in self.registry.items():
            if clean in key or key in clean:
                return node
        return None

    def resolve_physical_properties(self, dish_query: str) -> Dict[str, Any]:
        node = self.get_node(dish_query)
        if not node:
            return {
                "matched": False,
                "composite_density": 0.95,
                "macros_per_100g": {"calories": 180.0, "protein_g": 6.0, "carbs_g": 22.0, "fat_g": 7.0, "fiber_g": 2.0}
            }

        return {
            "matched": True,
            "canonical_name": node.canonical_name,
            "cuisine_family": node.cuisine_family,
            "glycemic_speed": node.glycemic_speed,
            "composite_density": round(node.composite_density, 2),
            "macros_per_100g": node.composite_macros_per_100g
        }


if __name__ == "__main__":
    ontology = CuranovaOntologyEngine()
    print("=== Testing Curanova Semantic Culinary Ontology ===")
    for query in ["siddu", "jalebi", "kofta", "kali roti", "malpua"]:
        res = ontology.resolve_physical_properties(query)
        print(f"[{res.get('canonical_name')}] Density: {res.get('composite_density')} g/cm3 | Macros/100g: {res.get('macros_per_100g')}")
        node = ontology.get_node(query)
        if node:
            decomp = node.explain_decomposition(200.0)
            print(f"  Decomposition of 200g portion: {len(decomp['subcomponents'])} subcomponents.")
