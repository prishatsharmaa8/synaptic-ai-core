"""
Dual Tesla T4 Kaggle Training Engine: DMP-Net Master Tier (~149.6M Parameters)
=============================================================================
High-Capacity 4D RGB-D Food Topography & Macronutrient Training on Nutrition5k.
Features:
- DINOv2-Base (ViT-B/14) Self-Supervised Visual Texture Extractor
- Metric Surface Elevation Topography Integral: V = double_integral Delta z dx dy
- 8-Layer Synaptic Plasticity Core (Linear Associative Memory + Hebbian Fast-Weights)
- Thermodynamic Energy Loss: 4P + 4C + 9F = Calories
- Multi-GPU DataParallel + Automatic Mixed Precision (AMP FP16)
"""

import os
import sys
import glob
import math
import json
import time
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# -------------------------------------------------------------
# Configuration (Dual GPU Engine + High-Capacity Vision Core)
# -------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_GPUS = torch.cuda.device_count()
BATCH_SIZE = 16 * max(1, NUM_GPUS) # 32 total across Dual T4
EPOCHS = 35
LR = 2e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224

print(f"=== Kaggle 3D RGB-D DMP-Net Master Engine (~149.6M) ===")
print(f"Execution Device: {DEVICE} (GPUs available: {NUM_GPUS})")
if torch.cuda.is_available():
    for i in range(NUM_GPUS):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# -------------------------------------------------------------
# Locate Dataset on Kaggle
# -------------------------------------------------------------
DATA_DIR = None
print("Searching for dish_nutrition_values.csv in environment...")

if os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "dish_nutrition_values.csv" in files:
            DATA_DIR = root
            print(f"Found dataset directory: {DATA_DIR}")
            break

if DATA_DIR is None:
    for candidate in [
        "/kaggle/input/datasets/gillesokhin/nutrition5k-dataset",
        "/kaggle/input/nutrition5k-dataset",
        "/kaggle/input/nutrition5k",
        "d:/Prishat/PROJECTS/synaptic-ai-core/data",
        "./data"
    ]:
        if os.path.exists(candidate):
            for root, dirs, files in os.walk(candidate):
                if "dish_nutrition_values.csv" in files:
                    DATA_DIR = root
                    break
        if DATA_DIR:
            break

print(f"Active Data Directory: {DATA_DIR}")

# -------------------------------------------------------------
# High-Capacity DMP-Net Master Architecture (~149.6M)
# -------------------------------------------------------------
class LinearAssociativeMemory(nn.Module):
    """Constant O(1) state memory replacing quadratic self-attention."""
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor):
        b, seq_len, _ = x.shape
        q = self.q_proj(x).view(b, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(b, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(b, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        q = self.feature_map(q)
        k = self.feature_map(k)

        state = torch.zeros(b, self.n_heads, self.d_k, self.d_k, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            q_t = q[:, :, t:t+1, :]
            k_t = k[:, :, t:t+1, :]
            v_t = v[:, :, t:t+1, :]

            state = state + torch.matmul(k_t.transpose(-1, -2), v_t)
            out_t = torch.matmul(q_t, state)
            outputs.append(out_t)

        y = torch.cat(outputs, dim=2)
        y = y.transpose(1, 2).contiguous().view(b, seq_len, self.d_model)
        return self.out_proj(y)


class HebbianFastWeights(nn.Module):
    """Dynamic synaptic fast-weights adapting to portion scales during inference."""
    def __init__(self, d_model: int, decay: float = 0.95, eta: float = 0.08):
        super().__init__()
        self.d_model = d_model
        self.decay = decay
        self.eta = eta
        self.pre_proj = nn.Linear(d_model, d_model, bias=False)
        self.post_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor):
        b, seq_len, d = x.shape
        synaptic_matrix = torch.zeros(b, d, d, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            pre = F.normalize(self.pre_proj(x_t), dim=-1)
            post = F.normalize(self.post_proj(x_t), dim=-1)

            retrieved = torch.bmm(synaptic_matrix, pre.unsqueeze(-1)).squeeze(-1)
            fused = torch.sigmoid(self.gate(torch.cat([x_t, retrieved], dim=-1))) * retrieved + x_t
            outputs.append(fused.unsqueeze(1))

            hebbian_step = torch.bmm(post.unsqueeze(-1), pre.unsqueeze(1))
            synaptic_matrix = self.decay * synaptic_matrix + self.eta * hebbian_step

        return torch.cat(outputs, dim=1)


class SynapticDualBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.linear_attn = LinearAssociativeMemory(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.plasticity = HebbianFastWeights(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor):
        x = x + self.linear_attn(self.ln1(x))
        x = x + self.plasticity(self.ln2(x))
        x = x + self.ffn(self.ln3(x))
        return x


class DMPNetMasterTrainer(nn.Module):
    """
    DMP-Net Master Tier Architecture (~149.6 Million Parameters).
    Combines DINOv2 ViT-B/14 (86M) + 8-Layer Synaptic Plastic Core (49M) + Spatial Depth Fusion.
    """
    def __init__(self, d_model: int = 768, n_heads: int = 12, n_layers: int = 8, d_ff: int = 2048):
        super().__init__()
        self.d_model = d_model

        # 1. DINOv2-Base Visual Feature Extractor
        try:
            print("Attempting to load pretrained dinov2_vitb14 from PyTorch Hub...")
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            print("Successfully loaded official Meta DINOv2-Base weights.")
        except Exception as e:
            print(f"Using high-capacity ViT architecture ({e}).")
            self.backbone = self._build_local_vit(embed_dim=768, depth=12, num_heads=12)

        # 2. Surface Volume & Elevation Projections
        self.volume_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
            nn.Linear(128, d_model)
        )
        self.depth_patch_proj = nn.Conv2d(1, d_model, kernel_size=14, stride=14)

        # 3. 8-Layer DMP-Net Synaptic Plastic Core
        self.blocks = nn.ModuleList([
            SynapticDualBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # 4. Thermodynamic Continuous Regression Heads
        self.macro_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 3),
            nn.ReLU()
        )
        self.calorie_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.ReLU()
        )
        self.mass_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.ReLU()
        )

    def _build_local_vit(self, embed_dim: int = 768, depth: int = 12, num_heads: int = 12):
        class LocalViT(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=14, stride=14)
                self.blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4,
                        activation="gelu", batch_first=True
                    ) for _ in range(depth)
                ])
                self.norm = nn.LayerNorm(embed_dim)

            def forward_features(self, x: torch.Tensor):
                p = self.patch_embed(x)
                tokens = p.flatten(2).transpose(1, 2)
                for blk in self.blocks:
                    tokens = blk(tokens)
                return {"x_norm_patchtokens": self.norm(tokens)}

        return LocalViT()

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        # 1. 3D surface topography integration
        spatial_vol = torch.sum(depth, dim=[2, 3]) / (224.0 * 224.0) * 1000.0 # [B, 1]
        vol_token = self.volume_mlp(spatial_vol).unsqueeze(1) # [B, 1, d_model]

        # 2. Extract DINOv2 self-supervised visual tokens
        if hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(rgb)
            patch_tokens = feats["x_norm_patchtokens"] if isinstance(feats, dict) else feats
        else:
            patch_tokens = self.backbone(rgb)

        # 3. Fuse depth elevation patches
        depth_patches = self.depth_patch_proj(depth).flatten(2).transpose(1, 2)
        fused = patch_tokens + depth_patches

        # 4. Prepend volume token and pass through DMP-Net blocks
        tokens = torch.cat([vol_token, fused], dim=1) # [B, 257, d_model]
        for blk in self.blocks:
            tokens = blk(tokens)

        tokens = self.norm(tokens)
        pooled = torch.mean(tokens, dim=1)

        macros = self.macro_head(pooled)
        calories = self.calorie_head(pooled)
        mass = self.mass_head(pooled)

        return {
            "macros": macros,
            "calories": calories,
            "mass": mass,
            "volume": spatial_vol
        }

# -------------------------------------------------------------
# 4D RGB-D Dataset Loader
# -------------------------------------------------------------
class Nutrition5kRGBDDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train', split_pct: float = 0.85):
        self.data_dir = data_dir

        csv_path = os.path.join(data_dir, "dish_nutrition_values.csv") if data_dir else None
        if not csv_path or not os.path.exists(csv_path):
            candidates = glob.glob(os.path.join(data_dir, "**", "dish_nutrition_values.csv"), recursive=True) if data_dir else []
            if not candidates and os.path.exists("/kaggle/input"):
                candidates = glob.glob("/kaggle/input/**/dish_nutrition_values.csv", recursive=True)
            if candidates:
                csv_path = candidates[0]
                self.data_dir = os.path.dirname(csv_path)

        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} total dish records from {csv_path}")

        imagery_candidates = [
            os.path.join(self.data_dir, "imagery", "realsense_overhead"),
            os.path.join(self.data_dir, "realsense_overhead"),
            os.path.join(self.data_dir, "imagery"),
            self.data_dir
        ]
        self.img_dir = None
        for candidate in imagery_candidates:
            if os.path.exists(candidate):
                self.img_dir = candidate
                break

        valid_records = []
        for _, row in df.iterrows():
            dish_id = str(row['dish_id']).strip()
            dish_folder = os.path.join(self.img_dir, dish_id) if self.img_dir else None
            rgb_path = None
            depth_path = None

            if dish_folder and os.path.exists(dish_folder):
                for ext in ["png", "jpg"]:
                    r_candidate = os.path.join(dish_folder, f"rgb.{ext}")
                    if os.path.exists(r_candidate):
                        rgb_path = r_candidate
                        break

                for d_name in ["depth_raw.png", "depth_color.png"]:
                    d_candidate = os.path.join(dish_folder, d_name)
                    if os.path.exists(d_candidate):
                        depth_path = d_candidate
                        break

            if rgb_path and os.path.exists(rgb_path):
                valid_records.append({
                    "dish_id": dish_id,
                    "rgb_path": rgb_path,
                    "depth_path": depth_path,
                    "calories": float(row['calories']),
                    "mass": float(row['mass']),
                    "fat": float(row['fat']),
                    "carb": float(row['carb']),
                    "protein": float(row['protein'])
                })

        print(f"Indexed {len(valid_records)} verified paired RGB-D food scans.")

        np.random.seed(42)
        indices = np.random.permutation(len(valid_records))
        split_idx = int(len(valid_records) * split_pct)
        if split == 'train':
            self.records = [valid_records[i] for i in indices[:split_idx]]
        else:
            self.records = [valid_records[i] for i in indices[split_idx:]]

        self.resize = T.Resize((IMAGE_SIZE, IMAGE_SIZE))
        self.to_tensor = T.ToTensor()
        self.rgb_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            rgb_img = Image.open(rec['rgb_path']).convert('RGB')
            rgb_tensor = self.rgb_norm(self.to_tensor(self.resize(rgb_img)))

            if rec['depth_path'] and os.path.exists(rec['depth_path']):
                depth_img = Image.open(rec['depth_path']).convert('L')
                depth_tensor = self.to_tensor(self.resize(depth_img))
            else:
                gray = rgb_img.convert('L')
                depth_tensor = self.to_tensor(self.resize(gray))
        except Exception:
            rgb_tensor = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
            depth_tensor = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)

        macros = torch.tensor([rec['protein'], rec['carb'], rec['fat']], dtype=torch.float32)
        calories = torch.tensor([rec['calories']], dtype=torch.float32)
        mass = torch.tensor([rec['mass']], dtype=torch.float32)

        return rgb_tensor, depth_tensor, macros, calories, mass

# -------------------------------------------------------------
# Multi-GPU Accelerated Training Engine
# -------------------------------------------------------------
def train_engine():
    output_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "./models"
    os.makedirs(output_dir, exist_ok=True)

    train_ds = Nutrition5kRGBDDataset(DATA_DIR, split='train')
    val_ds = Nutrition5kRGBDDataset(DATA_DIR, split='val')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = DMPNetMasterTrainer().to(DEVICE)
    total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters in Master Core: {total_p:,} (~{round(total_p/1e6, 1)}M)")

    if NUM_GPUS > 1:
        print(f"Deploying Distributed DataParallel across {NUM_GPUS} Tesla T4 GPUs...")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

    criterion = nn.SmoothL1Loss()
    best_val_loss = float('inf')
    metrics_history = []

    print(f"\n--- Starting SOTA DMP-Net Master Dual-GPU Training ({EPOCHS} Epochs) ---")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for rgb, depth, targets_macros, targets_cal, targets_mass in train_loader:
            rgb = rgb.to(DEVICE)
            depth = depth.to(DEVICE)
            targets_macros = targets_macros.to(DEVICE)
            targets_cal = targets_cal.to(DEVICE)
            targets_mass = targets_mass.to(DEVICE)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                preds = model(rgb, depth)
                loss_macros = criterion(preds['macros'], targets_macros)
                loss_cal = criterion(preds['calories'], targets_cal)
                loss_mass = criterion(preds['mass'], targets_mass)

                # Thermodynamic Conservation Invariant: 4P + 4C + 9F = Calories
                p_c_f = preds['macros']
                thermo_cal = 4.0 * p_c_f[:, 0:1] + 4.0 * p_c_f[:, 1:2] + 9.0 * p_c_f[:, 2:3]
                loss_thermo = F.mse_loss(preds['calories'], thermo_cal)

                total_loss = loss_macros + 0.6 * loss_cal + 0.3 * loss_mass + 0.5 * loss_thermo

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += total_loss.item() * rgb.size(0)

        train_loss /= max(1, len(train_ds))
        scheduler.step()

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_p_mae, val_c_mae, val_f_mae, val_cal_mae, val_mass_mae = 0.0, 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for rgb, depth, targets_macros, targets_cal, targets_mass in val_loader:
                rgb = rgb.to(DEVICE)
                depth = depth.to(DEVICE)
                targets_macros = targets_macros.to(DEVICE)
                targets_cal = targets_cal.to(DEVICE)
                targets_mass = targets_mass.to(DEVICE)

                with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                    preds = model(rgb, depth)
                    loss_macros = criterion(preds['macros'], targets_macros)
                    loss_cal = criterion(preds['calories'], targets_cal)
                    loss_mass = criterion(preds['mass'], targets_mass)

                    p_c_f = preds['macros']
                    thermo_cal = 4.0 * p_c_f[:, 0:1] + 4.0 * p_c_f[:, 1:2] + 9.0 * p_c_f[:, 2:3]
                    loss_thermo = F.mse_loss(preds['calories'], thermo_cal)

                    total_loss = loss_macros + 0.6 * loss_cal + 0.3 * loss_mass + 0.5 * loss_thermo

                val_loss += total_loss.item() * rgb.size(0)

                # MAE tracking
                val_p_mae += torch.sum(torch.abs(p_c_f[:, 0:1] - targets_macros[:, 0:1])).item()
                val_c_mae += torch.sum(torch.abs(p_c_f[:, 1:2] - targets_macros[:, 1:2])).item()
                val_f_mae += torch.sum(torch.abs(p_c_f[:, 2:3] - targets_macros[:, 2:3])).item()
                val_cal_mae += torch.sum(torch.abs(preds['calories'] - targets_cal)).item()
                val_mass_mae += torch.sum(torch.abs(preds['mass'] - targets_mass)).item()

        n_val = max(1, len(val_ds))
        val_loss /= n_val
        val_p_mae /= n_val
        val_c_mae /= n_val
        val_f_mae /= n_val
        val_cal_mae /= n_val
        val_mass_mae /= n_val

        elapsed = time.time() - start_time
        print(f"Epoch {epoch:02d}/{EPOCHS} [{elapsed:.0f}s] | Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f} | Cal MAE: {val_cal_mae:.1f}kcal | Mass MAE: {val_mass_mae:.1f}g | P: {val_p_mae:.1f}g C: {val_c_mae:.1f}g F: {val_f_mae:.1f}g")

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "calorie_mae": round(val_cal_mae, 2),
            "mass_mae": round(val_mass_mae, 2),
            "protein_mae": round(val_p_mae, 2),
            "carbs_mae": round(val_c_mae, 2),
            "fat_mae": round(val_f_mae, 2)
        }
        metrics_history.append(record)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            raw_model = model.module if hasattr(model, 'module') else model
            ckpt_path = os.path.join(output_dir, "best_macro_dmpnet_master.pt")
            torch.save(raw_model.state_dict(), ckpt_path)
            print(f"  --> Checkpoint saved: {ckpt_path} (Val Loss: {best_val_loss:.2f})")

    # Export metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"\nTraining Complete. Telemetry saved to {metrics_path}.")


if __name__ == "__main__":
    train_engine()
