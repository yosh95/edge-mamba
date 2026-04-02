import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))

import torch
import torch.nn as nn
from create_world_model import MambaWorldModel
from inference_world_model import MambaWorldModelStepper
from story_data_gen import generate_warehouse_trajectories


def main() -> None:
    # 1. Data Preparation (500 movement routes, 40 steps each)
    obs_dim, action_dim = 3, 1
    obs_data, action_data, reward_data = generate_warehouse_trajectories(500, 40)

    # 2. Model Initialization
    model = MambaWorldModel(obs_dim, action_dim, latent_dim=32, mamba_d_model=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # 3. Training (Short 15 epochs)
    print("Mamba-kun is learning the warehouse environment...")
    for epoch in range(15):
        optimizer.zero_grad()
        # Input up to 0:39, target up to 1:40
        pred_obs, pred_reward = model(obs_data[:, :-1, :], action_data)

        loss_obs = criterion(pred_obs, obs_data[:, 1:, :])
        loss_rew = criterion(pred_reward, reward_data)
        loss = loss_obs + loss_rew

        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch + 1}/15, Training Loss: {loss.item():.6f}")

    # 4. Story Simulation (Imagination Phase)
    print("\n--- Simulation: Mamba-kun's Imagination ---")
    print(
        "Story: Mamba-kun is at position 0.0 and decides to accelerate towards the target (10.0)."
    )

    stepper = MambaWorldModelStepper(model)
    current_obs = torch.tensor([[0.0, 0.0, 10.0]], dtype=torch.float32)  # [pos, vel, dist]

    for i in range(10):
        # Keep choosing the action of accelerating by 0.2 each step
        action = torch.tensor([[0.2]], dtype=torch.float32)

        # Let the model predict the "next state"
        next_obs, reward = stepper.step(current_obs, action)

        # Analyze the predicted values
        pred_pos = next_obs[0, 0].item()
        pred_vel = next_obs[0, 1].item()
        pred_dist = next_obs[0, 2].item()

        print(
            f"Step {i + 1:2d}: [Imagined] Pos: {pred_pos:5.2f}, Vel: {pred_vel:5.2f}, "
            f"Distance: {pred_dist:5.2f}, Reward: {reward.item():.4f}"
        )

        # Continue the "dream" by using the prediction as the next input
        current_obs = next_obs


if __name__ == "__main__":
    main()
