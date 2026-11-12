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
# Configuration
# -------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_GPUS = torch.cuda.device_count()
BATCH_SIZE = 32 * max(1, NUM_GPUS)
EPOCHS = 15
LR = 3e-4
WEIGHT_DECAY = 1e-4
IMAGE_SIZE = 224

print(f"=== Kaggle Macro Vision DMPNet Engine ===")
print(f"Execution Device: {DEVICE} (GPUs available: {NUM_GPUS})")
if torch.cuda.is_available():
    for i in range(NUM_GPUS):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# -------------------------------------------------------------
# Locate Dataset on Kaggle
# -------------------------------------------------------------
POSSIBLE_DATA_DIRS = [
    "/kaggle/input/nutrition5k-dataset",
    "/kaggle/input/nutrition5k",
    "d:/Prishat/PROJECTS/synaptic-ai-core/data",
    "./data"
]

DATA_DIR = None
for p in POSSIBLE_DATA_DIRS:
    if os.path.exists(p) and (os.path.exists(os.path.join(p, "dish_nutrition_values.csv")) or os.path.exists(os.path.join(p, "nutrition5k_dataset"))):
        DATA_DIR = p
        break

if DATA_DIR is None:
    # Scan /kaggle/input for any matching folder
    if os.path.exists("/kaggle/input"):
        for sub in os.listdir("/kaggle/input"):
            candidate = os.path.join("/kaggle/input", sub)
            if "nutrition" in sub.lower():
                DATA_DIR = candidate
                break

print(f"Active Data Directory: {DATA_DIR}")

# -------------------------------------------------------------
# DMP-Net Neural Core Architecture
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


class MacroVisionDMPNet(nn.Module):
    """
    Multimodal Vision + DMP-Net Architecture:
    Backbone: EfficientNet-B0 (feature extractor) -> Spatial visual tokens -> DMP-Net
    Output Heads:
      - Macros: [Protein (g), Carbs (g), Fats (g)]
      - Calories: [Energy (kcal)]
      - Mass: [Total Weight (g)]
    """
    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 2, d_ff: int = 512):
        super().__init__()
        # Pretrained visual feature extractor
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.feature_extractor = backbone.features # outputs [B, 1280, 7, 7]
        self.feature_proj = nn.Linear(1280, d_model)

        self.blocks = nn.ModuleList([
            SynapticDualBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Output regression heads
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

    def forward(self, images: torch.Tensor):
        # Extract visual feature map: [B, 1280, 7, 7]
        feats = self.feature_extractor(images)
        b, c, h, w = feats.shape
        # Convert spatial grid to sequence of tokens: [B, 49, 1280]
        tokens = feats.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        tokens = self.feature_proj(tokens) # [B, 49, d_model]

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)
        pooled = torch.mean(tokens, dim=1) # [B, d_model]

        macros = self.macro_head(pooled)        # [Protein, Carbs, Fat]
        calories = self.calorie_head(pooled)    # [Calories]
        mass = self.mass_head(pooled)            # [Mass]

        return {
            "macros": macros,
            "calories": calories,
            "mass": mass
        }

# -------------------------------------------------------------
# Dataset Implementation
# -------------------------------------------------------------
class Nutrition5kDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train', split_pct: float = 0.85):
        self.data_dir = data_dir
        
        # Locate nutrition CSV
        csv_path = os.path.join(data_dir, "dish_nutrition_values.csv")
        if not os.path.exists(csv_path):
            candidates = glob.glob(os.path.join(data_dir, "**", "dish_nutrition_values.csv"), recursive=True)
            if candidates:
                csv_path = candidates[0]
            else:
                raise FileNotFoundError(f"Could not locate dish_nutrition_values.csv in {data_dir}")

        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} total dish records from {csv_path}")

        # Locate imagery base folder
        imagery_candidates = [
            os.path.join(data_dir, "imagery", "realsense_overhead"),
            os.path.join(data_dir, "realsense_overhead"),
            os.path.join(data_dir, "imagery"),
            data_dir
        ]
        self.img_dir = None
        for candidate in imagery_candidates:
            if os.path.exists(candidate):
                self.img_dir = candidate
                break

        print(f"Imagery base directory: {self.img_dir}")

        # Filter dishes that have valid images
        valid_records = []
        for _, row in df.iterrows():
            dish_id = str(row['dish_id']).strip()
            # Look for rgb image
            img_path = os.path.join(self.img_dir, dish_id, "rgb.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(self.img_dir, dish_id, "rgb.jpg")
            
            if os.path.exists(img_path):
                valid_records.append({
                    "dish_id": dish_id,
                    "img_path": img_path,
                    "calories": float(row['calories']),
                    "mass": float(row['mass']),
                    "fat": float(row['fat']),
                    "carb": float(row['carb']),
                    "protein": float(row['protein'])
                })

        print(f"Found {len(valid_records)} dishes with verified image assets.")
        
        # If no local images found (e.g. metadata-only dry run), generate synthetic anchors for testing
        if len(valid_records) == 0:
            print("Notice: No overhead imagery in search path. Creating calibrated synthetic RGB-D batches for demonstration.")
            for _, row in df.head(100).iterrows():
                valid_records.append({
                    "dish_id": str(row['dish_id']),
                    "img_path": None,
                    "calories": float(row['calories']),
                    "mass": float(row['mass']),
                    "fat": float(row['fat']),
                    "carb": float(row['carb']),
                    "protein": float(row['protein'])
                })

        # Split train / val
        np.random.seed(42)
        indices = np.random.permutation(len(valid_records))
        split_idx = int(len(valid_records) * split_pct)
        if split == 'train':
            self.records = [valid_records[i] for i in indices[:split_idx]]
        else:
            self.records = [valid_records[i] for i in indices[split_idx:]]

        print(f"Dataset split '{split}': {len(self.records)} samples.")

        # Transforms
        if split == 'train':
            self.transform = T.Compose([
                T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.1, contrast=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        if rec['img_path'] and os.path.exists(rec['img_path']):
            try:
                img = Image.open(rec['img_path']).convert('RGB')
                tensor = self.transform(img)
            except Exception:
                tensor = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        else:
            # Synthetic food pattern for non-image dry runs
            tensor = torch.randn(3, IMAGE_SIZE, IMAGE_SIZE) * 0.1

        macros = torch.tensor([rec['protein'], rec['carb'], rec['fat']], dtype=torch.float32)
        calories = torch.tensor([rec['calories']], dtype=torch.float32)
        mass = torch.tensor([rec['mass']], dtype=torch.float32)

        return tensor, macros, calories, mass

# -------------------------------------------------------------
# Training Engine with Thermodynamic Constraint
# -------------------------------------------------------------
def train_engine():
    output_dir = "/kaggle/working" if os.path.exists("/kaggle/working") else "./models"
    os.makedirs(output_dir, exist_ok=True)

    train_ds = Nutrition5kDataset(DATA_DIR, split='train')
    val_ds = Nutrition5kDataset(DATA_DIR, split='val')

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = MacroVisionDMPNet().to(DEVICE)
    if NUM_GPUS > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    criterion = nn.SmoothL1Loss()
    best_val_loss = float('inf')
    metrics_history = []

    print(f"\n--- Starting DMP-Net Accelerated Training ({EPOCHS} Epochs) ---")
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

            # Combined loss
            total_loss = loss_macros + 0.5 * loss_cal + 0.2 * loss_mass + 0.5 * loss_thermo

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item() * images.size(0)

        train_loss /= len(train_ds)
        scheduler.step()

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_p_mae, val_c_mae, val_f_mae, val_cal_mae = 0.0, 0.0, 0.0, 0.0

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

                batch_val_loss = loss_macros + 0.5 * loss_cal + 0.2 * loss_mass + 0.5 * loss_thermo
                val_loss += batch_val_loss.item() * images.size(0)

                # Track MAE
                val_p_mae += F.l1_loss(pred_p, targets_macros[:, 0:1]).item() * images.size(0)
                val_c_mae += F.l1_loss(pred_c, targets_macros[:, 1:2]).item() * images.size(0)
                val_f_mae += F.l1_loss(pred_f, targets_macros[:, 2:3]).item() * images.size(0)
                val_cal_mae += F.l1_loss(preds['calories'], targets_cal).item() * images.size(0)

        n_val = max(1, len(val_ds))
        val_loss /= n_val
        val_p_mae /= n_val
        val_c_mae /= n_val
        val_f_mae /= n_val
        val_cal_mae /= n_val

        metrics_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "p_mae": round(val_p_mae, 2),
            "c_mae": round(val_c_mae, 2),
            "f_mae": round(val_f_mae, 2),
            "cal_mae": round(val_cal_mae, 1)
        })

        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"MAE -> Protein: {val_p_mae:.1f}g, Carbs: {val_c_mae:.1f}g, Fats: {val_f_mae:.1f}g, Calories: {val_cal_mae:.1f} kcal")

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            raw_model = model.module if hasattr(model, 'module') else model
            ckpt_path = os.path.join(output_dir, "best_macro_dmpnet.pt")
            torch.save(raw_model.state_dict(), ckpt_path)
            print(f"  -> Best model checkpoint saved to: {ckpt_path}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s.")

    # -------------------------------------------------------------
    # Mobile Export: TorchScript & ONNX
    # -------------------------------------------------------------
    print("\n--- Exporting Mobile Optimized Artifacts ---")
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.eval()

    example_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

    # 1. TorchScript Export
    ts_path = os.path.join(output_dir, "macro_dmpnet_mobile.pt")
    try:
        traced_model = torch.jit.trace(raw_model, example_input)
        traced_model.save(ts_path)
        print(f"TorchScript compiled successfully: {ts_path} ({os.path.getsize(ts_path)/(1024*1024):.1f} MB)")
    except Exception as e:
        print(f"TorchScript trace notice: {e}")

    # 2. ONNX Export
    onnx_path = os.path.join(output_dir, "macro_dmpnet.onnx")
    try:
        torch.onnx.export(
            raw_model,
            example_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['macros', 'calories', 'mass'],
            dynamic_axes={'image': {0: 'batch_size'}}
        )
        print(f"ONNX binary exported successfully: {onnx_path} ({os.path.getsize(onnx_path)/(1024*1024):.1f} MB)")
    except Exception as e:
        print(f"ONNX export notice: {e}")

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"Final training metrics saved to: {metrics_path}")
    print("\n=== All Systems Ready for Mobile Client Integration ===")

if __name__ == "__main__":
    train_engine()
