import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

sys.path.append(str(Path.cwd()))
from edge_mamba.numpy_model import MambaConfig, MambaNumpy
from edge_mamba.torch_model import Mamba


class MambaStepONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, conv_state, ssm_re, ssm_im, p_Bx_re, p_Bx_im):
        return self.model.step(x, conv_state, ssm_re, ssm_im, p_Bx_re, p_Bx_im)


def test_step_consistency():
    d_model, d_state, d_conv, n_heads, expand = 128, 16, 3, 8, 2
    d_inner = d_model * expand
    head_dim = d_inner // n_heads
    batch_size = 1

    # 1. Initialize models
    torch_model = Mamba(
        d_model=d_model, d_state=d_state, d_conv=d_conv, n_heads=n_heads, expand=expand
    )
    torch_model.eval()

    np_config = MambaConfig(
        d_model=d_model, d_state=d_state, d_conv=d_conv, n_heads=n_heads, expand=expand
    )
    np_model = MambaNumpy(np_config)

    # Sync weights: Torch -> NumPy
    with torch.no_grad():
        m = torch_model.mamba
        p = np_model.params
        p["in_proj.weight"] = m.in_proj.weight.numpy()
        if m.in_proj.bias is not None:
            p["in_proj.bias"] = m.in_proj.bias.numpy()
        p["conv1d.weight"] = m.conv1d.weight.squeeze(1).numpy()
        if m.conv1d.bias is not None:
            p["conv1d.bias"] = m.conv1d.bias.numpy()
        p["x_proj.weight"] = m.x_proj.weight.numpy()
        if m.x_proj.bias is not None:
            p["x_proj.bias"] = m.x_proj.bias.numpy()
        p["dt_proj.weight"] = m.dt_proj.weight.numpy()
        p["dt_proj.bias"] = m.dt_proj.bias.numpy()
        p["A_log"] = m.A_log.numpy()
        p["A_imag"] = m.A_imag.numpy()
        p["D"] = m.D.numpy()
        p["out_proj.weight"] = m.out_proj.weight.numpy()
        if m.out_proj.bias is not None:
            p["out_proj.bias"] = m.out_proj.bias.numpy()

    # 2. Export to ONNX
    temp_onnx = "tests/temp_consistency.onnx"
    Path("tests").mkdir(parents=True, exist_ok=True)
    from scripts.export_onnx_step import MambaStepONNXWrapper

    wrapper = MambaStepONNXWrapper(torch_model)
    wrapper.eval()

    input_data = (
        torch.randn(batch_size, 1, d_model),
        torch.randn(batch_size, d_inner, d_conv),
        torch.randn(batch_size, n_heads, head_dim, d_state),
        torch.randn(batch_size, n_heads, head_dim, d_state),
        torch.randn(batch_size, n_heads, head_dim, d_state),
        torch.randn(batch_size, n_heads, head_dim, d_state),
    )

    input_names = ["x", "conv_state", "ssm_re", "ssm_im", "p_Bx_re", "p_Bx_im"]
    output_names = [
        "output",
        "next_conv_state",
        "next_ssm_re",
        "next_ssm_im",
        "next_Bx_re",
        "next_Bx_im",
    ]

    torch.onnx.export(
        wrapper,
        input_data,
        temp_onnx,
        opset_version=18,
        input_names=input_names,
        output_names=output_names,
    )
    ort_session = ort.InferenceSession(temp_onnx)

    # 3. Prepare Test Case
    x_np = np.random.randn(batch_size, 1, d_model).astype(np.float32)
    c_np = np.random.randn(batch_size, d_inner, d_conv).astype(np.float32)
    s_re_np = np.random.randn(batch_size, n_heads, head_dim, d_state).astype(np.float32)
    s_im_np = np.random.randn(batch_size, n_heads, head_dim, d_state).astype(np.float32)
    b_re_np = np.random.randn(batch_size, n_heads, head_dim, d_state).astype(np.float32)
    b_im_np = np.random.randn(batch_size, n_heads, head_dim, d_state).astype(np.float32)

    # 4. Run PyTorch
    with torch.no_grad():
        res_torch = torch_model.step(
            torch.from_numpy(x_np),
            torch.from_numpy(c_np),
            torch.from_numpy(s_re_np),
            torch.from_numpy(s_im_np),
            torch.from_numpy(b_re_np),
            torch.from_numpy(b_im_np),
        )

    # 5. Run NumPy
    res_np = np_model.step(x_np, c_np, s_re_np, s_im_np, b_re_np, b_im_np)

    # 6. Run ONNX
    ort_inputs = dict(
        zip(input_names, [x_np, c_np, s_re_np, s_im_np, b_re_np, b_im_np], strict=True)
    )
    res_ort = ort_session.run(None, ort_inputs)

    # 7. Compare Results
    def check(a, b, label):
        if isinstance(a, torch.Tensor):
            a = a.numpy()
        diff = np.max(np.abs(a - b))
        if diff < 1e-4:
            print(f"✅ {label:20} Match (diff: {diff:.2e})")
            return True
        else:
            print(f"❌ {label:20} FAIL  (diff: {diff:.2e})")
            return False

    success = True
    print("\n--- Torch vs NumPy ---")
    for i, name in enumerate(output_names):
        success &= check(res_torch[i], res_np[i], name)

    print("\n--- Torch vs ONNX ---")
    for i, name in enumerate(output_names):
        success &= check(res_torch[i], res_ort[i], name)

    if success:
        print("\n🏆 VERIFICATION SUCCESSFUL!")
    else:
        print("\n💀 VERIFICATION FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    test_step_consistency()
