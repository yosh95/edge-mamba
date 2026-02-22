from pathlib import Path

import numpy as np

from edge_mamba.numpy_model import MambaConfig, MambaNumpy


def run_inference() -> None:
    weights_path = "mamba_weights.npz"

    if not Path(weights_path).exists():
        print(f"Error: {weights_path} not found. Please run export_to_numpy.py first.")
        return

    print("Loading NumPy weights...")
    # Load .npz file
    with np.load(weights_path) as data:
        state_dict = {k: data[k] for k in data.files}

    # Initialize config (should match training config)
    config = MambaConfig(d_model=64, d_state=16, d_conv=4, expand=2)

    # Initialize model
    model = MambaNumpy(config)

    # Load weights
    model.load_state_dict(state_dict)

    # Create dummy input for inference
    x = np.random.randn(1, 32, 64)  # (Batch, Seq, Dim)

    # Run inference
    print("Running inference (forward)...")
    output = model.forward(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Run token-by-token inference (step)
    print("\nRunning token-by-token inference (step)...")
    conv_state = None
    ssm_state = None
    prev_Bx = None

    step_outputs = []
    for i in range(x.shape[1]):
        token_input = x[:, i : i + 1, :]  # (Batch, 1, Dim)
        out, conv_state, ssm_state, prev_Bx = model.step(
            token_input, conv_state, ssm_state, prev_Bx
        )
        step_outputs.append(out)

    step_output_total = np.concatenate(step_outputs, axis=1)
    print(f"Step output shape: {step_output_total.shape}")

    # Check if forward and step match
    np.testing.assert_allclose(output, step_output_total, rtol=1e-4, atol=1e-4)
    print("Forward and Step results match!")

    print("Inference successful!")


if __name__ == "__main__":
    run_inference()
