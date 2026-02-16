import os

import numpy as np

from edge_mamba.numpy_model import MambaConfig, MambaNumpy


def run_inference():
    weights_path = "mamba_weights.npz"

    if not os.path.exists(weights_path):
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
    print("Running inference...")
    output = model.forward(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Inference successful!")


if __name__ == "__main__":
    run_inference()
