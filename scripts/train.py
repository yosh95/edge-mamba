import torch
import torch.nn as nn
import torch.optim as optim

from edge_mamba.torch_model import Mamba


def train() -> None:
    # Configuration
    d_model = 64
    d_state = 16
    d_conv = 4
    batch_size = 8
    seq_len = 32
    learning_rate = 1e-3
    num_epochs = 5

    # Initialize Model
    model = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=2)

    # Optimizer and Loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print("Starting training with dummy data...")

    for epoch in range(num_epochs):
        # Dummy inputs and targets
        inputs = torch.randn(batch_size, seq_len, d_model)
        targets = torch.randn(
            batch_size, seq_len, d_model
        )  # Assuming regression task for simplicity

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Save the model
    save_path = "mamba_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train()
