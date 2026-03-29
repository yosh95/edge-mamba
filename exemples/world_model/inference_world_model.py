import torch
import torch.nn as nn
from create_world_model import MambaWorldModel

class MambaWorldModelStepper:
    def __init__(self, model: MambaWorldModel):
        self.model = model
        self.reset()
        
    def reset(self):
        self.conv_state = None
        self.ssm_state_re = None
        self.ssm_state_im = None
        self.prev_Bx_re = None
        self.prev_Bx_im = None

    @torch.no_grad()
    def step(self, obs, action):
        """
        Perform a single step inference.
        obs: (1, obs_dim)
        action: (1, action_dim)
        """
        # 1. Encode observation
        latent = self.model.encoder(obs)
        
        # 2. Prepare Mamba input
        mamba_input = torch.cat([latent, action], dim=-1)
        mamba_input = self.model.dynamics_input_proj(mamba_input).unsqueeze(1) # (1, 1, d_model)
        
        # 3. Mamba Step
        (
            mamba_out, 
            self.conv_state, 
            self.ssm_state_re, 
            self.ssm_state_im, 
            self.prev_Bx_re, 
            self.prev_Bx_im
        ) = self.model.mamba.step(
            mamba_input, 
            self.conv_state, 
            self.ssm_state_re, 
            self.ssm_state_im, 
            self.prev_Bx_re, 
            self.prev_Bx_im
        )
        
        # 4. Predict next state and reward
        pred_obs = self.model.obs_predictor(mamba_out.squeeze(1))
        pred_reward = self.model.reward_predictor(mamba_out.squeeze(1))
        
        return pred_obs, pred_reward
