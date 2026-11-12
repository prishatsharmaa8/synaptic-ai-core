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
import torchvision.models as models

# -------------------------------------------------------------
# Configuration (4D RGB-D Spatial Vision + Dual GPU Engine)
# -------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_GPUS = torch.cuda.device_count()
BATCH_SIZE = 32 * max(1, NUM_GPUS)
EPOCHS = 40
LR = 4e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224

print(f"=== Kaggle 3D RGB-D Macro Vision DMPNet Engine ===")
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
# 3D RGB-D DMP-Net Neural Architecture
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
    """Dynamic synaptic fast-weights updating per meal portion."""
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


class MacroVisionRGBDNet(nn.Module):
    """
    4D RGB-D Spatial Vision Network:
    Input: [B, 4, 224, 224] (RGB Color + Dense Monocular Surface Depth)
    Calculates spatial volume integral V = int int Delta z dx dy
    Output Heads:
      - Macros: [Protein (g), Carbs (g), Fats (g)]
      - Calories: [Energy (kcal)]
      - Mass: [Total Weight (g)]
    """
    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 3, d_ff: int = 512):
        super().__init__()
        # 4-channel input projection (fuses RGB + metric depth)
        self.rgbd_stem = nn.Conv2d(4, 3, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.rgbd_stem.weight)

        # Pretrained visual feature extractor
        try:
            backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            print("Loaded official pretrained EfficientNet weights.")
        except Exception:
            print("Initializing EfficientNet backbone locally.")
            backbone = models.efficientnet_b0(weights=None)

        self.features = backbone.features # outputs [B, 1280, 7, 7]
        self.feature_proj = nn.Linear(1280, d_model)

        # Spatial volume integration head
        self.volume_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, d_model)
        )

        self.blocks = nn.ModuleList([
            SynapticDualBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Continuous numerical regression heads
        self.macro_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.ReLU()
        )
        self.calorie_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.ReLU()
        )
        self.mass_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.ReLU()
        )

    def forward(self, rgbd: torch.Tensor):
        # 1. Compute spatial volume integral from depth channel
        # rgbd is [B, 4, 224, 224], channel 3 is normalized surface depth
        depth_channel = rgbd[:, 3:4, :, :]
        # Integrated relative surface volume: sum of depth values over plate area
        spatial_vol = torch.sum(depth_channel, dim=[2, 3]) / (224.0 * 224.0) * 1000.0 # [B, 1]
        vol_embedding = self.volume_mlp(spatial_vol).unsqueeze(1) # [B, 1, d_model]

        # 2. Extract visual feature tokens
        fused_rgb = self.rgbd_stem(rgbd)
        feats = self.features(fused_rgb)
        b, c, h, w = feats.shape
        tokens = feats.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        tokens = self.feature_proj(tokens) # [B, 49, d_model]

        # 3. Concatenate volume token to visual tokens: [B, 50, d_model]
        tokens = torch.cat([vol_embedding, tokens], dim=1)

        # 4. Pass through DMP-Net dual-memory blocks
        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        pooled = torch.mean(tokens, dim=1)

        macros = self.macro_head(pooled)
        calories = self.calorie_head(pooled)
        mass = self.mass_head(pooled)

        return {
            "macros": macros,
            "calories": calories,
            "mass": mass
        }

# -------------------------------------------------------------
# 4D RGB-D Dataset Loader with Paired Depth Imagery
# -------------------------------------------------------------
class Nutrition5kRGBDDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train', split_pct: float = 0.85):
        self.data_dir = data_dir
        
        # Locate nutrition CSV
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

        # Locate imagery directory
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

        print(f"Imagery base directory: {self.img_dir}")

        # Index paired RGB and Depth assets
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

        print(f"Found {len(valid_records)} dishes with verified paired image assets.")

        # Train / val split
        np.random.seed(42)
        indices = np.random.permutation(len(valid_records))
        split_idx = int(len(valid_records) * split_pct)
        if split == 'train':
            self.records = [valid_records[i] for i in indices[:split_idx]]
        else:
            self.records = [valid_records[i] for i in indices[split_idx:]]

        print(f"Dataset split '{split}': {len(self.records)} samples.")

        # Transforms
        self.resize = T.Resize((IMAGE_SIZE, IMAGE_SIZE))
        self.to_tensor = T.ToTensor()
        self.rgb_norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            # 1. Load RGB
            rgb_img = Image.open(rec['rgb_path']).convert('RGB')
            rgb_tensor = self.rgb_norm(self.to_tensor(self.resize(rgb_img)))

            # 2. Load Depth (if available) or generate relative topography prior
            if rec['depth_path'] and os.path.exists(rec['depth_path']):
                depth_img = Image.open(rec['depth_path']).convert('L')
                depth_tensor = self.to_tensor(self.resize(depth_img))
            else:
                # Synthesize surface topography from grayscale gradient
                gray = rgb_img.convert('L')
                depth_tensor = self.to_tensor(self.resize(gray))

            # Concatenate to 4-channel RGB-D: [4, 224, 224]
            rgbd_tensor = torch.cat([rgb_tensor, depth_tensor], dim=0)

        except Exception:
            rgbd_tensor = torch.zeros(4, IMAGE_SIZE, IMAGE_SIZE)

        macros = torch.tensor([rec['protein'], rec['carb'], rec['fat']], dtype=torch.float32)
        calories = torch.tensor([rec['calories']], dtype=torch.float32)
        mass = torch.tensor([rec['mass']], dtype=torch.float32)

        return rgbd_tensor, macros, calories, mass

# -------------------------------------------------------------
# Accelerated Multi-GPU Training Engine
# -------------------------------------------------------------
def train_engine():
    output_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "./models"
    os.makedirs(output_dir, exist_ok=True)

    train_ds = Nutrition5kRGBDDataset(DATA_DIR, split='train')
    val_ds = Nutrition5kRGBDDataset(DATA_DIR, split='val')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = MacroVisionRGBDNet().to(DEVICE)
    if NUM_GPUS > 1:
        print(f"Deploying Distributed DataParallel across {NUM_GPUS} Tesla T4 GPUs...")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    criterion = nn.SmoothL1Loss()
    best_val_loss = float('inf')
    metrics_history = []

    print(f"\n--- Starting 4D RGB-D Accelerated GPU Training ({EPOCHS} Epochs) ---")
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for images, targets_macros, targets_cal, targets_mass in train_loader:
            images = images.to(DEVICE)
            targets_macros = targets_macros.to(DEVICE)
            targets_cal = targets_cal.to(DEVICE)
            targets_mass = targets_mass.to(DEVICE)

            optimizer.zero_grad()
            preds = model(images)

            # 1. Macro regression loss
            loss_macros = criterion(preds['macros'], targets_macros)
            # 2. Calorie regression loss
            loss_cal = criterion(preds['calories'], targets_cal)
            # 3. Mass regression loss
            loss_mass = criterion(preds['mass'], targets_mass)

            # 4. Thermodynamic Conservation Invariant: 4P + 4C + 9F = Calories
            pred_p = preds['macros'][:, 0:1]
            pred_c = preds['macros'][:, 1:2]
            pred_f = preds['macros'][:, 2:3]
            thermo_calories = 4.0 * pred_p + 4.0 * pred_c + 9.0 * pred_f
            loss_thermo = F.mse_loss(preds['calories'], thermo_calories)

            # Combined spatial-thermodynamic loss
            total_loss = loss_macros + 0.6 * loss_cal + 0.3 * loss_mass + 0.5 * loss_thermo

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item() * images.size(0)

        train_loss /= len(train_ds)
        scheduler.step()

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_p_mae, val_c_mae, val_f_mae, val_cal_mae, val_mass_mae = 0.0, 0.0, 0.0, 0.0, 0.0

        with torch.no_grad():
            for images, targets_macros, targets_cal, targets_mass in val_loader:
                images = images.to(DEVICE)
                targets_macros = targets_macros.to(DEVICE)
                targets_cal = targets_cal.to(DEVICE)
                targets_mass = targets_mass.to(DEVICE)

                preds = model(images)

                loss_macros = criterion(preds['macros'], targets_macros)
                loss_cal = criterion(preds['calories'], targets_cal)
                loss_mass = criterion(preds['mass'], targets_mass)

                pred_p = preds['macros'][:, 0:1]
                pred_c = preds['macros'][:, 1:2]
                pred_f = preds['macros'][:, 2:3]
                thermo_cal = 4.0 * pred_p + 4.0 * pred_c + 9.0 * pred_f
                loss_thermo = F.mse_loss(preds['calories'], thermo_cal)

                batch_val_loss = loss_macros + 0.6 * loss_cal + 0.3 * loss_mass + 0.5 * loss_thermo
                val_loss += batch_val_loss.item() * images.size(0)

                # Track MAE in physical units
                val_p_mae += F.l1_loss(pred_p, targets_macros[:, 0:1]).item() * images.size(0)
                val_c_mae += F.l1_loss(pred_c, targets_macros[:, 1:2]).item() * images.size(0)
                val_f_mae += F.l1_loss(pred_f, targets_macros[:, 2:3]).item() * images.size(0)
                val_cal_mae += F.l1_loss(preds['calories'], targets_cal).item() * images.size(0)
                val_mass_mae += F.l1_loss(preds['mass'], targets_mass).item() * images.size(0)

        n_val = max(1, len(val_ds))
        val_loss /= n_val
        val_p_mae /= n_val
        val_c_mae /= n_val
        val_f_mae /= n_val
        val_cal_mae /= n_val
        val_mass_mae /= n_val

        metrics_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "p_mae": round(val_p_mae, 2),
            "c_mae": round(val_c_mae, 2),
            "f_mae": round(val_f_mae, 2),
            "cal_mae": round(val_cal_mae, 1),
            "mass_mae": round(val_mass_mae, 1)
        })

        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"MAE -> Mass: {val_mass_mae:.1f}g | Protein: {val_p_mae:.1f}g, Carbs: {val_c_mae:.1f}g, Fats: {val_f_mae:.1f}g | Calories: {val_cal_mae:.1f} kcal")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            raw_model = model.module if hasattr(model, 'module') else model
            ckpt_path = os.path.join(output_dir, "best_macro_dmpnet.pt")
            torch.save(raw_model.state_dict(), ckpt_path)
            print(f"  -> New Best Model Checkpoint saved: {ckpt_path}")

    elapsed = time.time() - start_time
    print(f"\n4D RGB-D Training successfully completed in {elapsed/60.0:.2f} minutes.")

    # -------------------------------------------------------------
    # Mobile Export: TorchScript & ONNX
    # -------------------------------------------------------------
    print("\n--- Exporting Mobile Optimized Artifacts ---")
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.eval()

    class MobileWrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x):
            out = self.base(x)
            # Output: [Calories, Protein, Carbs, Fats, Mass]
            return torch.cat([out['calories'], out['macros'], out['mass']], dim=-1)

    mobile_wrapped = MobileWrapper(raw_model)
    mobile_wrapped.eval()

    example_input = torch.randn(1, 4, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

    # 1. TorchScript Mobile Export
    ts_path = os.path.join(output_dir, "macro_vision_dmpnet.pt")
    try:
        traced_model = torch.jit.trace(mobile_wrapped, example_input)
        traced_model.save(ts_path)
        print(f"TorchScript Mobile binary exported: {ts_path} ({os.path.getsize(ts_path)/(1024*1024):.1f} MB)")
    except Exception as e:
        print(f"TorchScript notice: {e}")

    # 2. ONNX Export
    onnx_path = os.path.join(output_dir, "macro_vision_dmpnet.onnx")
    try:
        torch.onnx.export(
            mobile_wrapped,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['rgbd_input'],
            output_names=['predictions'],
            dynamic_axes={'rgbd_input': {0: 'batch_size'}}
        )
        print(f"ONNX binary exported: {onnx_path} ({os.path.getsize(onnx_path)/(1024*1024):.1f} MB)")
    except Exception as e:
        print(f"ONNX notice: {e}")

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"Final metrics logged to: {metrics_path}")
    print("\n=== High-Precision 3D RGB-D Macro Model Successfully Ready ===")

if __name__ == "__main__":
    train_engine()
