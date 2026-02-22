import numpy as np

from edge_mamba.numpy_model import MambaConfig, MambaNumpy


def test_inference_only() -> None:
    config = MambaConfig(d_model=16, d_state=4, d_conv=3, expand=2)
    model = MambaNumpy(config)

    x = np.random.randn(1, 10, 16)  # Batch=1, Seq=10, Dim=16
    y = model.forward(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")

    assert y.shape == (1, 10, 16)
    print("Inference test passed!")


def test_step() -> None:
    config = MambaConfig(d_model=16, d_state=4, d_conv=3, expand=2)
    model = MambaNumpy(config)

    x = np.random.randn(1, 1, 16)  # Batch=1, Seq=1, Dim=16
    out, conv_state, ssm_state, prev_Bx = model.step(x, None, None, None)

    print(f"Step Input shape: {x.shape}")
    print(f"Step Output shape: {out.shape}")

    assert out.shape == (1, 1, 16)
    print("Step test passed!")


if __name__ == "__main__":
    test_inference_only()
    test_step()
