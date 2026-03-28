import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

sys.path.append(str(Path.cwd()))
from edge_mamba.torch_model import Mamba


class MambaStepONNXWrapper(nn.Module):
    """
    Wrapper for exporting ONLY the inference step function.
    All complex numbers are now real-valued pairs (float32).
    """

    def __init__(self, model: Mamba) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        x: Tensor,
        conv_state: Tensor,
        ssm_re: Tensor,
        ssm_im: Tensor,
        p_Bx_re: Tensor,
        p_Bx_im: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Exportable step function.
        """
        return self.model.step(x, conv_state, ssm_re, ssm_im, p_Bx_re, p_Bx_im)


def export_step_onnx() -> None:
    # Model configuration
    d_model = 128
    d_state = 16
    d_conv = 3
    n_heads = 8
    expand = 2
    d_inner = d_model * expand
    head_dim = d_inner // n_heads
    batch_size = 1

    # Initialize the real-valued Mamba model
    model = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, n_heads=n_heads, expand=expand)
    model.eval()

    wrapper = MambaStepONNXWrapper(model)

    # Dummy inputs for tracing (All float32)
    x = torch.randn(batch_size, 1, d_model)
    conv_state = torch.zeros(batch_size, d_inner, d_conv)
    ssm_re = torch.zeros(batch_size, n_heads, head_dim, d_state)
    ssm_im = torch.zeros(batch_size, n_heads, head_dim, d_state)
    p_Bx_re = torch.zeros(batch_size, n_heads, head_dim, d_state)
    p_Bx_im = torch.zeros(batch_size, n_heads, head_dim, d_state)

    onnx_file = "mamba_step_inference.onnx"

    try:
        torch.onnx.export(
            wrapper,
            (x, conv_state, ssm_re, ssm_im, p_Bx_re, p_Bx_im),
            onnx_file,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["x", "conv_state", "ssm_re", "ssm_im", "p_Bx_re", "p_Bx_im"],
            output_names=[
                "output",
                "next_conv_state",
                "next_ssm_re",
                "next_ssm_im",
                "next_Bx_re",
                "next_Bx_im",
            ],
            dynamic_axes={"x": {0: "batch_size"}},
        )
        onnx_path = Path(onnx_file)
        if onnx_path.exists():
            print(f"Successfully exported ONNX to {onnx_file}")
            print(f"Model size: {onnx_path.stat().st_size / 1024:.2f} KB")
        else:
            print("Export failed: File not found.")
    except Exception as e:
        print(f"Export failed with error: {e}")


if __name__ == "__main__":
    export_step_onnx()
