import time

import numpy as np

from edge_mamba.numpy_model import AdamOptimizer, MambaConfig, MambaNumpy


def train_numpy() -> None:
    # 1. Configuration
    # Using a small configuration for demonstration
    config = MambaConfig(d_model=32, d_state=8, d_conv=3, expand=2)

    # 2. Initialize Model and Optimizer
    model = MambaNumpy(config)
    optimizer = AdamOptimizer(model.params, lr=1e-3)

    # 3. Generate Dummy Dataset (Batch, Length, D_model)
    B, L, D = 4, 16, 32
    X = np.random.randn(B, L, D).astype(np.float32)

    # Target: simple regression task (predict shifted input)
    Y_target = np.roll(X, shift=1, axis=1)

    print("Starting NumPy-only training...")
    print(
        f"Config: d_model={config.d_model}, d_state={config.d_state}, "
        f"batch_size={B}, seq_len={L}"
    )
    print("-" * 30)

    start_time = time.time()
    for epoch in range(21):
        # --- Forward Pass ---
        # Set training=True to cache intermediate values for backward
        preds = model.forward(X, training=True)

        # --- Compute Loss (Mean Squared Error) ---
        loss = np.mean((preds - Y_target) ** 2)

        # --- Backward Pass ---
        # Gradient of MSE loss: 2 * (preds - target) / total_elements
        grad_output = 2 * (preds - Y_target) / (B * L * D)
        _, grads = model.backward(grad_output)

        # --- Parameter Update ---
        optimizer.step(grads)

        if epoch % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:3d} | Loss: {loss:.6f} | Time: {elapsed:.2f}s")

    print("-" * 30)
    print("Training finished!")


if __name__ == "__main__":
    train_numpy()
