import unittest

import numpy as np
import torch

from edge_mamba.numpy_model import MambaConfig, MambaNumpy
from edge_mamba.torch_model import Mamba as MambaTorch


class TestBackwardConsistency(unittest.TestCase):
    def test_backward_gradients(self) -> None:
        # 1. Config
        # Using smaller dimensions for consistency checks
        d_model, d_state, d_conv, expand = 16, 8, 3, 2
        batch_size, seq_len = 2, 8

        # 2. Initialize Models
        # Setting bias=False for simplicity in gradient mapping
        torch_model = MambaTorch(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bias=False,
            conv_bias=False,
        )

        config = MambaConfig(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bias=False,
            conv_bias=False,
        )
        numpy_model = MambaNumpy(config)

        # 3. Synchronize Weights (Torch -> NumPy)
        # Directly map weights from Torch to NumPy
        sd = torch_model.state_dict()
        numpy_model.params["in_proj.weight"] = sd["mamba.in_proj.weight"].numpy()
        numpy_model.params["out_proj.weight"] = sd["mamba.out_proj.weight"].numpy()
        numpy_model.params["conv1d.weight"] = (
            sd["mamba.conv1d.weight"].squeeze(1).numpy()
        )
        numpy_model.params["x_proj.weight"] = sd["mamba.x_proj.weight"].numpy()
        numpy_model.params["dt_proj.weight"] = sd["mamba.dt_proj.weight"].numpy()
        numpy_model.params["dt_proj.bias"] = sd["mamba.dt_proj.bias"].numpy()
        numpy_model.params["A_log"] = sd["mamba.A_log"].numpy()
        numpy_model.params["A_imag"] = sd["mamba.A_imag"].numpy()
        numpy_model.params["D"] = sd["mamba.D"].numpy()

        # 4. Dummy Input
        x_np = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        x_torch = torch.from_numpy(x_np).requires_grad_(True)

        # 5. Forward Pass
        y_torch = torch_model(x_torch)
        y_numpy = numpy_model.forward(x_np, training=True)

        # Verify Forward consistency
        np.testing.assert_allclose(
            y_torch.detach().numpy(), y_numpy, rtol=1e-4, atol=1e-4
        )
        print("Forward pass matched.")

        # 6. Backward Pass
        # Loss function: Mean of squared outputs (MSE-like)
        loss_torch = (y_torch**2).mean()
        loss_torch.backward()

        # In NumPy: dL/dy = 2 * y / total_elements (matching Torch's mean loss)
        grad_output = 2 * y_numpy / y_numpy.size
        d_x_numpy, numpy_grads = numpy_model.backward(grad_output)

        # Verify input gradient consistency
        print("\nInput Gradient Consistency Check:")
        assert x_torch.grad is not None
        d_x_torch = x_torch.grad.numpy()
        max_diff_x = np.abs(d_x_torch - d_x_numpy).max()
        print(f" - {'x_input':16s} | Max Diff: {max_diff_x:.6e}")
        np.testing.assert_allclose(d_x_torch, d_x_numpy, rtol=1e-3, atol=1e-3)

        # 7. Compare Gradients for major layers
        # Mapping names from NumPy model to PyTorch parameters
        check_list = [
            ("in_proj.weight", torch_model.mamba.in_proj.weight),
            ("out_proj.weight", torch_model.mamba.out_proj.weight),
            ("conv1d.weight", torch_model.mamba.conv1d.weight),
            ("x_proj.weight", torch_model.mamba.x_proj),
            ("dt_proj.weight", torch_model.mamba.dt_proj.weight),
            ("dt_proj.bias", torch_model.mamba.dt_proj.bias),
            ("A_log", torch_model.mamba.A_log),
            ("A_imag", torch_model.mamba.A_imag),
            ("D", torch_model.mamba.D),
        ]

        print("\nGradient Consistency Check:")
        for name, torch_obj in check_list:
            if hasattr(torch_obj, "weight"):
                grad = torch_obj.weight.grad
            else:
                grad = getattr(torch_obj, "grad", None)

            assert grad is not None
            t_grad = grad.numpy()
            n_grad = numpy_grads[name]

            # Align shapes
            if name == "conv1d.weight":
                t_grad = t_grad.squeeze(1)

            max_diff = np.abs(t_grad - n_grad).max()
            print(f" - {name:16s} | Max Diff: {max_diff:.6e}")

            # Assert they are close (with tolerance for float32 accumulation)
            np.testing.assert_allclose(t_grad, n_grad, rtol=1e-3, atol=1e-3)

        print("\nBackward consistency test passed!")


if __name__ == "__main__":
    unittest.main()
