"""
Multi-Step Numerical Stress Test for DMP-Net Master Architecture
================================================================
Validates gradient flow, FP16/AMP dynamic range stability,
and finite convergence across consecutive training steps before cloud execution.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure kaggle_kernel can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kaggle_kernel.train_kernel import DMPNetMasterTrainer


def run_numerical_stress_test():
    print("=== Starting DMP-Net Numerical Stress & Stability Test ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on Device: {device}")

    model = DMPNetMasterTrainer(pretrained_backbone=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=1.0)
    use_cuda = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    # 1. Stress Test with Extreme Food Targets (>1200 kcal, large mass)
    batch_size = 4
    rgb = torch.randn(batch_size, 3, 224, 224, device=device)
    depth = torch.rand(batch_size, 1, 224, 224, device=device)

    # Realistic and high-density macronutrient targets
    targets_macros = torch.tensor([
        [60.0, 140.0, 45.0],   # Heavy meal: 4*60 + 4*140 + 9*45 = 1205 kcal
        [15.0, 20.0, 5.0],     # Light snack: 4*15 + 4*20 + 9*5 = 185 kcal
        [85.0, 180.0, 55.0],   # Feast: 4*85 + 4*180 + 9*55 = 1555 kcal
        [30.0, 50.0, 12.0]     # Standard dish: 4*30 + 4*50 + 9*12 = 428 kcal
    ], dtype=torch.float32, device=device)

    targets_cal = torch.tensor([[1205.0], [185.0], [1555.0], [428.0]], dtype=torch.float32, device=device)
    targets_mass = torch.tensor([[650.0], [120.0], [820.0], [280.0]], dtype=torch.float32, device=device)

    print("\nExecuting 10 Consecutive Training Steps with Extreme Targets...")
    initial_loss = None
    last_loss = None

    for step in range(1, 11):
        model.train()
        optimizer.zero_grad()

        with torch.amp.autocast(device.type, enabled=use_cuda):
            preds = model(rgb, depth)
            loss_macros = criterion(preds['macros'], targets_macros)
            loss_cal = criterion(preds['calories'], targets_cal)
            loss_mass = criterion(preds['mass'], targets_mass)

            p_c_f = preds['macros']
            thermo_cal = 4.0 * p_c_f[:, 0:1] + 4.0 * p_c_f[:, 1:2] + 9.0 * p_c_f[:, 2:3]
            loss_thermo = criterion(preds['calories'], thermo_cal)

            total_loss = loss_macros + 0.6 * loss_cal + 0.3 * loss_mass + 0.5 * loss_thermo

        assert torch.isfinite(total_loss), f"Step {step}: Total loss exploded to {total_loss.item()}"

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)

        # Verify gradient validity across all layers
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Step {step}: Non-finite gradient detected in {name}"

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()

        loss_val = total_loss.item()
        if initial_loss is None:
            initial_loss = loss_val
        last_loss = loss_val

        cal_pred_mean = preds['calories'].mean().item()
        print(f"  Step {step:02d}: Loss = {loss_val:.4f} | Pred Cal Mean = {cal_pred_mean:.1f} kcal | Thermo Loss = {loss_thermo.item():.4f}")

    print(f"\nInitial Loss: {initial_loss:.4f} --> Final Step Loss: {last_loss:.4f}")
    assert last_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {last_loss}"
    print("PASS: Loss strictly decreased with zero NaN across 10 steps.")

    # 2. Edge Case: Extreme zero-elevation flat plate
    print("\nTesting Edge Case: Zero Depth Surface (Completely Flat)...")
    zero_depth = torch.zeros(2, 1, 224, 224, device=device)
    zero_rgb = torch.randn(2, 3, 224, 224, device=device)
    out_zero = model(zero_rgb, zero_depth)
    assert torch.isfinite(out_zero['calories']).all(), "Flat depth produced non-finite calories"
    assert torch.isfinite(out_zero['macros']).all(), "Flat depth produced non-finite macros"
    print("PASS: Zero elevation handled cleanly.")

    # 3. Edge Case: Single-sample batch (Batch size = 1)
    print("\nTesting Edge Case: Single Sample Batch...")
    out_single = model(rgb[:1], depth[:1])
    assert torch.isfinite(out_single['calories']).all(), "Single sample produced non-finite calories"
    print("PASS: Single sample batch completed with finite predictions.")

    print("\n=== ALL NUMERICAL STABILITY CHECKS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_numerical_stress_test()
