"""
Model Export Pipeline
=====================
Exports trained MacroDMPNet weights into:
1. TorchScript (.pt) - for lightweight mobile C++ and Python execution
2. ONNX (.onnx) - for cross-platform edge inference in Flutter apps
"""

import os
import torch
from models.macro_dmpnet import MacroDMPNet


class ExportWrapper(torch.nn.Module):
    """
    Simplifies the model output dictionary into a single unified tensor
    for clean ONNX / mobile parsing:
    Returns: [batch, num_classes + 4]
    Where indices 0..num_classes-1 are category logits,
    followed by [calories, protein, carbs, fat].
    """
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor):
        out = self.base_model(x)
        logits = out["class_logits"]
        cal = out["calories"]
        macros = out["macros"]
        return torch.cat([logits, cal, macros], dim=-1)


def export_models():
    os.makedirs("models", exist_ok=True)
    device = torch.device("cpu")

    model = MacroDMPNet(
        in_features=512,
        num_classes=20,
        d_model=256,
        n_heads=4,
        n_layers=3,
        d_ff=512
    ).to(device)

    checkpoint_path = os.path.join("models", "best_macro_dmpnet.pt")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded trained weights from: {checkpoint_path}")
    else:
        print("Using initialized weights for export demonstration.")

    model.eval()
    wrapper = ExportWrapper(model)

    dummy_input = torch.randn(1, 16, 512, device=device)

    # 1. Export TorchScript
    ts_path = os.path.join("models", "macro_dmpnet_mobile.pt")
    traced_model = torch.jit.trace(wrapper, dummy_input)
    traced_model.save(ts_path)
    print(f"TorchScript Mobile Model exported to: {ts_path}")

    # 2. Export ONNX (if onnx package installed)
    try:
        import onnx
        onnx_path = os.path.join("models", "macro_dmpnet.onnx")
        torch.onnx.export(
            wrapper,
            dummy_input,
            onnx_path,
            input_names=["visual_tokens"],
            output_names=["predictions"],
            dynamic_axes={"visual_tokens": {0: "batch_size", 1: "sequence_length"}}
        )
        print(f"ONNX Model exported to: {onnx_path}")
    except ImportError:
        print("Note: Install 'onnx' package to generate the .onnx binary file.")


if __name__ == "__main__":
    export_models()
