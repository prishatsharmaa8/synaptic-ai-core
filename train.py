"""
Training Pipeline for MacroDMPNet
=================================
Trains MacroDMPNet on Google Colab GPU / CUDA:
- Multi-task loss: Classification + Macro Regression + Calorie Prediction
- Thermodynamic Conservation Loss: enforces 4*Protein + 4*Carbs + 9*Fat = Calories
- Saves best checkpoint to models/best_macro_dmpnet.pt
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.macro_dmpnet import MacroDMPNet
from data.macro_dataset import get_macro_dataloaders, FOOD_CATEGORIES


def compute_loss(outputs, batch, physics_weight: float = 0.5):
    # 1. Classification loss
    loss_cls = F.cross_entropy(outputs["class_logits"], batch["category"])

    # 2. Macro regression loss (Protein, Carbs, Fat)
    loss_macro = F.smooth_l1_loss(outputs["macros"], batch["macros"])

    # 3. Calorie regression loss
    loss_cal = F.smooth_l1_loss(outputs["calories"], batch["calories"])

    # 4. Thermodynamic Conservation Penalty:
    # 4 kcal/g Protein + 4 kcal/g Carbs + 9 kcal/g Fat must equal total Calories
    pred_p = outputs["macros"][:, 0]
    pred_c = outputs["macros"][:, 1]
    pred_f = outputs["macros"][:, 2]
    derived_energy = 4.0 * pred_p + 4.0 * pred_c + 9.0 * pred_f

    loss_physics = F.mse_loss(outputs["calories"].squeeze(-1), derived_energy)

    total_loss = loss_cls + 2.0 * loss_macro + 0.1 * loss_cal + physics_weight * loss_physics
    return total_loss, {
        "cls": loss_cls.item(),
        "macro": loss_macro.item(),
        "cal": loss_cal.item(),
        "physics": loss_physics.item()
    }


def train_model(epochs: int = 10, batch_size: int = 32, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"  MacroDMPNet Training Engine")
    print(f"  Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"==================================================")

    train_loader, val_loader = get_macro_dataloaders(batch_size=batch_size)

    model = MacroDMPNet(
        in_features=512,
        num_classes=len(FOOD_CATEGORIES),
        d_model=256,
        n_heads=4,
        n_layers=3,
        d_ff=512
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    os.makedirs("models", exist_ok=True)
    checkpoint_path = os.path.join("models", "best_macro_dmpnet.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for batch in train_loader:
            features = batch["features"].to(device)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(features)
            loss, metrics = compute_loss(outputs, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        epoch_time = time.time() - start_time
        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_macro_err = 0.0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                outputs = model(features)
                loss, metrics = compute_loss(outputs, batch)
                val_loss += loss.item()
                val_macro_err += metrics["macro"]

        avg_val_loss = val_loss / len(val_loader)
        avg_macro_err = val_macro_err / len(val_loader)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.2f}s) | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Macro Err (g): {avg_macro_err:.2f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)

    print(f"\nTraining Complete! Best model saved to: {checkpoint_path}")

    # Run Demonstration Inference
    print("\n--- Model Demonstration (Sample Prediction) ---")
    model.eval()
    sample_batch = next(iter(val_loader))
    features = sample_batch["features"][:1].to(device)
    with torch.no_grad():
        pred = model(features)

    cat_idx = pred["class_logits"].argmax(dim=-1).item()
    pred_p, pred_c, pred_f = pred["macros"][0].tolist()
    pred_kcal = pred["calories"][0].item()

    true_cat = sample_batch["category"][0].item()
    true_p, true_c, true_f = sample_batch["macros"][0].tolist()
    true_kcal = sample_batch["calories"][0].item()

    print(f"Target Meal:     {FOOD_CATEGORIES[true_cat]}")
    print(f"Predicted Meal:  {FOOD_CATEGORIES[cat_idx]}")
    print(f"Predicted:       {pred_kcal:.1f} kcal | Protein: {pred_p:.1f}g | Carbs: {pred_c:.1f}g | Fat: {pred_f:.1f}g")
    print(f"Ground Truth:    {true_kcal:.1f} kcal | Protein: {true_p:.1f}g | Carbs: {true_c:.1f}g | Fat: {true_f:.1f}g")


if __name__ == "__main__":
    train_model(epochs=8)
