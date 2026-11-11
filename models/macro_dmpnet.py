"""
MacroDMPNet: Dual-Memory Plastic Network for Food & Nutritional Macros
======================================================================
Architecture tailored for real-time food recognition and macro estimation:
- Predicts Calories (kcal), Protein (g), Carbohydrates (g), and Fats (g)
- Linear Associative Memory: Constant O(1) state per step for visual patches
- Hebbian Fast Weights: Real-time continuous personal adaptation to portion sizes
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAssociativeMemory(nn.Module):
    """
    Constant O(1) memory state per step. Replaces quadratic attention.
    """
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

    def forward(self, x: torch.Tensor, prev_state: torch.Tensor = None):
        b, seq_len, _ = x.shape
        q = self.q_proj(x).view(b, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(b, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(b, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        q = self.feature_map(q)
        k = self.feature_map(k)

        if prev_state is None:
            state = torch.zeros(b, self.n_heads, self.d_k, self.d_k, device=x.device, dtype=x.dtype)
        else:
            state = prev_state

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
        return self.out_proj(y), state


class HebbianFastWeights(nn.Module):
    """
    Dynamic plastic weights that update at inference time for portion scaling.
    """
    def __init__(self, d_model: int, decay: float = 0.95, eta: float = 0.08):
        super().__init__()
        self.d_model = d_model
        self.decay = decay
        self.eta = eta
        self.pre_proj = nn.Linear(d_model, d_model, bias=False)
        self.post_proj = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor, synaptic_matrix: torch.Tensor = None):
        b, seq_len, d = x.shape
        if synaptic_matrix is None:
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

        out = torch.cat(outputs, dim=1)
        return out, synaptic_matrix


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

    def forward(self, x: torch.Tensor, attn_state=None, plastic_state=None):
        attn_out, next_attn = self.linear_attn(self.ln1(x), prev_state=attn_state)
        x = x + attn_out
        plastic_out, next_plastic = self.plasticity(self.ln2(x), synaptic_matrix=plastic_state)
        x = x + plastic_out
        x = x + self.ffn(self.ln3(x))
        return x, next_attn, next_plastic


class MacroDMPNet(nn.Module):
    """
    Unified Multimodal Architecture for Food & Macro Estimation:
    Outputs:
    - class_logits: [batch, num_classes] (Identification of meal/food item)
    - calories: [batch, 1] (Total energy in kcal)
    - macros: [batch, 3] (Protein (g), Carbs (g), Fats (g))
    """
    def __init__(
        self,
        in_features: int = 512,
        num_classes: int = 50,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 512
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_features, d_model)
        self.blocks = nn.ModuleList([
            SynapticDualBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Output Heads
        self.class_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classes)
        )

        # Macro regression head: outputs [Protein (g), Carbs (g), Fats (g)]
        self.macro_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 3),
            nn.ReLU() # Grams cannot be negative
        )

        # Calorie head: outputs total energy (kcal)
        self.calorie_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.ReLU() # Calories cannot be negative
        )

    def forward(self, x: torch.Tensor, states=None):
        """
        x: [batch, seq_len, in_features] (e.g. sequence of image visual tokens)
        """
        x = self.input_proj(x)
        if states is None:
            states = [(None, None) for _ in range(len(self.blocks))]

        next_states = []
        for i, block in enumerate(self.blocks):
            prev_attn, prev_plastic = states[i]
            x, next_attn, next_plastic = block(x, attn_state=prev_attn, plastic_state=prev_plastic)
            next_states.append((next_attn, next_plastic))

        x = self.norm(x)
        
        # Global pooling across sequence tokens
        pooled = torch.mean(x, dim=1)

        class_logits = self.class_head(pooled)
        macros = self.macro_head(pooled)        # [protein, carbs, fat]
        calories = self.calorie_head(pooled)    # [calories]

        return {
            "class_logits": class_logits,
            "macros": macros,
            "calories": calories,
            "states": next_states
        }
