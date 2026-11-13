"""
MacroVisionRGBDNet: 4D RGB-D Spatial Vision Network
==================================================
Dual-T4 Trained Architecture matching best_macro_dmpnet.pt
Inputs:
  - rgbd: [B, 4, 224, 224] where channel 0..2 are RGB and channel 3 is monocular depth.
Outputs:
  - macros: [B, 3] (Protein (g), Carbs (g), Fats (g))
  - calories: [B, 1] (Total energy in kcal)
  - mass: [B, 1] (Total physical weight in grams)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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
    """Hebbian Fast-Weights layer for dynamic portion scaling."""
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
        except Exception:
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
        # rgbd is [B, 4, 224, 224], channel 3 is normalized surface depth
        depth_channel = rgbd[:, 3:4, :, :]
        # Integrated relative surface volume: sum of depth values over plate area
        spatial_vol = torch.sum(depth_channel, dim=[2, 3]) / (224.0 * 224.0) * 1000.0 # [B, 1]
        vol_embedding = self.volume_mlp(spatial_vol).unsqueeze(1) # [B, 1, d_model]

        # Extract visual feature tokens
        fused_rgb = self.rgbd_stem(rgbd)
        feats = self.features(fused_rgb)
        b, c, h, w = feats.shape
        tokens = feats.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        tokens = self.feature_proj(tokens) # [B, 49, d_model]

        # Concatenate volume token to visual tokens: [B, 50, d_model]
        tokens = torch.cat([vol_embedding, tokens], dim=1)

        # Pass through DMP-Net dual-memory blocks
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
