import unittest

import numpy as np
import torch

from edge_mamba.numpy_model import MambaConfig, MambaNumpy
from edge_mamba.torch_model import Mamba as MambaTorch


class TestMambaImplementation(unittest.TestCase):
    def test_output_consistency(self) -> None:
        # Config
        d_model = 32
        d_state = 8
        d_conv = 3
        expand = 2
        batch_size = 2
        seq_len = 10

        # 1. Initialize PyTorch Model
        torch_model = MambaTorch(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bias=True,
            conv_bias=True,
        )
        torch_model.eval()  # Set to eval mode (affects dropout etc if present)

        # 2. Extract weights
        # Sync weights: Torch -> NumPy
        np_config = MambaConfig(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bias=True,
            conv_bias=True,
        )
        numpy_model = MambaNumpy(np_config)

        with torch.no_grad():
            m = torch_model.mamba
            p = numpy_model.params
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

        # 5. Create dummy input
        x_numpy = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        x_torch = torch.from_numpy(x_numpy)

        # 6. Forward pass
        with torch.no_grad():
            y_torch = torch_model(x_torch).numpy()

        y_numpy = numpy_model.forward(x_numpy)

        # 7. Compare
        # Tolerances need to be slightly loose due to float32 precision differences
        # between PScan (parallel) and sequential loop, and potential accumulation
        # errors.
        print(f"Max difference: {np.abs(y_torch - y_numpy).max()}")
        np.testing.assert_allclose(y_torch, y_numpy, rtol=2e-3, atol=2e-3)
        print("Forward consistency test passed.")

    def test_step_consistency(self) -> None:
        # Config
        d_model = 32
        d_state = 8
        d_conv = 3
        expand = 2
        batch_size = 2

        # 1. Initialize PyTorch Model
        torch_model = MambaTorch(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bias=True,
            conv_bias=True,
        )
        torch_model.eval()

        # 2. Sync weights
        np_config = MambaConfig(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bias=True,
            conv_bias=True,
        )
        numpy_model = MambaNumpy(np_config)
        with torch.no_grad():
            m = torch_model.mamba
            p = numpy_model.params
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

        # 4. Dummy input for step: (B, 1, D_model)
        x_np = np.random.randn(batch_size, 1, d_model).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        # Initial states
        c_t = None
        s_re_t = None
        s_im_t = None
        p_Bx_re_t = None
        p_Bx_im_t = None

        c_np = None
        s_re_np = None
        s_im_np = None
        p_Bx_re_np = None
        p_Bx_im_np = None

        # 5. Run a few steps
        for _ in range(3):
            with torch.no_grad():
                res_t = torch_model.step(x_torch, c_t, s_re_t, s_im_t, p_Bx_re_t, p_Bx_im_t)
                out_t, c_t, s_re_t, s_im_t, p_Bx_re_t, p_Bx_im_t = res_t

            res_np = numpy_model.step(x_np, c_np, s_re_np, s_im_np, p_Bx_re_np, p_Bx_im_np)
            out_np, c_np, s_re_np, s_im_np, p_Bx_re_np, p_Bx_im_np = res_np

            # 6. Compare
            np.testing.assert_allclose(out_t.numpy(), out_np, rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(c_t.numpy(), c_np, rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(s_re_t.numpy(), s_re_np, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(s_im_t.numpy(), s_im_np, rtol=1e-3, atol=1e-3)
            np.testing.assert_allclose(p_Bx_re_t.numpy(), p_Bx_re_np, rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(p_Bx_im_t.numpy(), p_Bx_im_np, rtol=1e-4, atol=1e-4)

        print("Step consistency test passed!")


if __name__ == "__main__":
    unittest.main()
