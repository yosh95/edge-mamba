import torch
import torch.nn as nn
from edge_mamba.torch_model import Mamba

class MambaWorldModel(nn.Module):
    def __init__(
        self, 
        obs_dim, 
        action_dim, 
        latent_dim=64, 
        mamba_d_model=128, 
        mamba_d_state=16
    ):
        super().__init__()
        
        # 1. Encoder: Obs -> Latent
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # 2. Dynamics: (Latent + Action) -> Mamba Feature
        # We concatenate latent and action to feed into Mamba
        self.dynamics_input_proj = nn.Linear(latent_dim + action_dim, mamba_d_model)
        
        self.mamba = Mamba(
            d_model=mamba_d_model,
            d_state=mamba_d_state,
            expand=2
        )
        
        # 3. Predictors
        self.obs_predictor = nn.Sequential(
            nn.Linear(mamba_d_model, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, obs_dim)
        )
        
        self.reward_predictor = nn.Sequential(
            nn.Linear(mamba_d_model, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1)
        )

    def forward(self, obs_seq, action_seq):
        """
        obs_seq: (batch, seq_len, obs_dim)
        action_seq: (batch, seq_len, action_dim)
        """
        # Encode observations
        latents = self.encoder(obs_seq) # (B, L, latent_dim)
        
        # Prepare Mamba input (concatenate latents and actions)
        mamba_input = torch.cat([latents, action_seq], dim=-1)
        mamba_input = self.dynamics_input_proj(mamba_input)
        
        # Mamba forward pass
        mamba_output = self.mamba(mamba_input)
        
        # Predict next observations and rewards
        pred_obs = self.obs_predictor(mamba_output)
        pred_rewards = self.reward_predictor(mamba_output)
        
        return pred_obs, pred_rewards
