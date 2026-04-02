import numpy as np
import torch


def generate_warehouse_trajectories(
    num_sequences: int = 100, seq_len: int = 40
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate data according to the physical laws of the warehouse
    """
    target_pos = 10.0

    all_obs = []
    all_actions = []
    all_rewards = []

    for _ in range(num_sequences):
        traj_obs = []
        traj_actions = []
        traj_rewards = []

        pos = 0.0
        vel = 0.0

        for _ in range(seq_len + 1):
            dist = target_pos - pos
            traj_obs.append([pos, vel, dist])

            if len(traj_obs) > seq_len:
                break

            # Random actions (sometimes mixed with smart ones)
            action = np.random.uniform(-0.5, 0.5)
            if dist > 0:
                action += 0.1  # Bias to move closer to the target

            # Physical laws: velocity = velocity + acceleration, position = position + velocity
            vel = np.clip(vel + action, -1.0, 1.0)
            pos = pos + vel

            # Reward design
            reward = (
                -abs(target_pos - pos) * 0.1
            )  # Higher reward the closer to the target (less negative)
            if abs(target_pos - pos) < 0.5:
                reward += 1.0  # Goal proximity bonus
            if pos < 0 or pos > 15:
                reward -= 5.0  # Wall collision penalty

            traj_actions.append([action])
            traj_rewards.append([reward])

        all_obs.append(traj_obs)
        all_actions.append(traj_actions)
        all_rewards.append(traj_rewards)

    return (
        torch.tensor(all_obs, dtype=torch.float32),
        torch.tensor(all_actions, dtype=torch.float32),
        torch.tensor(all_rewards, dtype=torch.float32),
    )
