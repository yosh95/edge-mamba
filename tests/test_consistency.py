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
        state_dict = torch_model.state_dict()

        # 3. Initialize NumPy Model with same config
        config = MambaConfig(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bias=True,
            conv_bias=True,
        )
        numpy_model = MambaNumpy(config)

        # 4. Load weights into NumPy model
        numpy_model.load_state_dict(state_dict)

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
        np.testing.assert_allclose(y_torch, y_numpy, rtol=1e-4, atol=1e-4)
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

        # 2. Extract weights
        state_dict = torch_model.state_dict()

        # 3. Initialize NumPy Model
        config = MambaConfig(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bias=True,
            conv_bias=True,
        )
        numpy_model = MambaNumpy(config)
        numpy_model.load_state_dict(state_dict)

        # 4. Dummy input for step: (B, 1, D_model)
        x_np = np.random.randn(batch_size, 1, d_model).astype(np.float32)
        x_torch = torch.from_numpy(x_np)

        # Initial states
        conv_state_torch = None
        ssm_state_torch = None
        prev_Bx_torch = None

        conv_state_np = None
        ssm_state_np = None
        prev_Bx_np = None

        # 5. Run a few steps
        for _ in range(3):
            with torch.no_grad():
                out_torch, conv_state_torch, ssm_state_torch, prev_Bx_torch = (
                    torch_model.step(
                        x_torch, conv_state_torch, ssm_state_torch, prev_Bx_torch
                    )
                )

            out_np, conv_state_np, ssm_state_np, prev_Bx_np = numpy_model.step(
                x_np, conv_state_np, ssm_state_np, prev_Bx_np
            )

            # 6. Compare
            np.testing.assert_allclose(out_torch.numpy(), out_np, rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(
                conv_state_torch.numpy(), conv_state_np, rtol=1e-4, atol=1e-4
            )
            np.testing.assert_allclose(
                ssm_state_torch.numpy(), ssm_state_np, rtol=1e-3, atol=1e-3
            )
            np.testing.assert_allclose(
                prev_Bx_torch.numpy(), prev_Bx_np, rtol=1e-4, atol=1e-4
            )

        print("Step consistency test passed!")


if __name__ == "__main__":
    unittest.main()
