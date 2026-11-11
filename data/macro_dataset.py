"""
Synthetic Nutrition & Macro Benchmark Dataset
==============================================
Simulates realistic meal representations with ground truth macronutrients:
- Protein (g)
- Carbohydrates (g)
- Fats (g)
- Energy Calories (kcal) obeying thermodynamic conservation: 4*P + 4*C + 9*F
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# 20 representative food and dish categories
FOOD_CATEGORIES = [
    "Grilled Chicken Breast", "Salmon Fillet", "Boiled Eggs", "Greek Yogurt",
    "Tofu Stir-fry", "Whey Protein Shake", "Brown Rice Plate", "Sweet Potato Mash",
    "Rolled Oats with Berries", "Whole Wheat Pasta", "Quinoa Salad", "Avocado Toast",
    "Almonds and Walnuts", "Olive Oil Drizzle", "Peanut Butter Spread",
    "Steamed Broccoli & Veggies", "Lean Ground Beef", "Turkey Sandwich",
    "Lentil Soup", "Chia Seed Pudding"
]


class SyntheticMacroDataset(Dataset):
    """
    Generates synthetic multimodal meal visual feature sequences
    paired with true macronutrient profiles.
    """
    def __init__(self, num_samples: int = 2000, seq_len: int = 16, feature_dim: int = 512, seed: int = 42):
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_samples = num_samples
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        self.samples = []
        for _ in range(num_samples):
            # Pick a dominant category
            cat_id = np.random.randint(0, len(FOOD_CATEGORIES))

            # Simulate realistic macronutrient grams based on dish profile
            if cat_id in [0, 1, 2, 3, 5, 16]:  # High Protein
                protein = np.random.uniform(25.0, 55.0)
                carbs = np.random.uniform(2.0, 20.0)
                fat = np.random.uniform(3.0, 18.0)
            elif cat_id in [6, 7, 8, 9, 10]:   # High Carbohydrate
                protein = np.random.uniform(5.0, 15.0)
                carbs = np.random.uniform(40.0, 85.0)
                fat = np.random.uniform(1.0, 10.0)
            elif cat_id in [11, 12, 13, 14]:   # High Healthy Fat
                protein = np.random.uniform(4.0, 18.0)
                carbs = np.random.uniform(10.0, 30.0)
                fat = np.random.uniform(20.0, 45.0)
            else:                              # Balanced / Mixed
                protein = np.random.uniform(15.0, 35.0)
                carbs = np.random.uniform(20.0, 50.0)
                fat = np.random.uniform(8.0, 22.0)

            # Thermodynamic energy conservation law: 4 kcal/g Protein, 4 kcal/g Carbs, 9 kcal/g Fat
            # plus slight biological variance (+-3%)
            calories = 4.0 * protein + 4.0 * carbs + 9.0 * fat + np.random.normal(0, 5.0)
            calories = max(calories, 10.0)

            # Generate synthetic visual feature tokens conditioned on the category
            base_pattern = np.sin(np.linspace(cat_id, cat_id + 5, feature_dim))
            tokens = np.random.randn(seq_len, feature_dim) * 0.5 + base_pattern

            self.samples.append({
                "features": torch.tensor(tokens, dtype=torch.float32),
                "category": torch.tensor(cat_id, dtype=torch.long),
                "macros": torch.tensor([protein, carbs, fat], dtype=torch.float32),
                "calories": torch.tensor([calories], dtype=torch.float32)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def get_macro_dataloaders(batch_size: int = 32, train_samples: int = 2500, val_samples: int = 500):
    train_dataset = SyntheticMacroDataset(num_samples=train_samples, seed=42)
    val_dataset = SyntheticMacroDataset(num_samples=val_samples, seed=999)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
