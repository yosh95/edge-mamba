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
        print("Test passed: PyTorch and NumPy implementations match.")


if __name__ == "__main__":
    unittest.main()
