import os

import numpy as np
import torch


def export_weights(
    pth_path: str = "mamba_model.pth", npz_path: str = "mamba_weights.npz"
) -> None:
    if not os.path.exists(pth_path):
        print(f"Error: {pth_path} not found. Please run train.py first.")
        return

    print(f"Loading weights from {pth_path}...")
    state_dict = torch.load(pth_path, map_location="cpu")

    numpy_dict = {}
    for k, v in state_dict.items():
        # Clean up keys if necessary and convert to numpy
        clean_k = k.replace("mamba.", "")
        numpy_dict[clean_k] = v.detach().cpu().numpy()

    print(f"Saving weights to {npz_path}...")
    np.savez(npz_path, **numpy_dict)
    print("Done!")


if __name__ == "__main__":
    export_weights()
