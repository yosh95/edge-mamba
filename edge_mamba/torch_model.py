import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Mamba(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        n_heads: int = 8,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
    ) -> None:
        super().__init__()
        self.mamba = CustomMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_heads=n_heads,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            bias=bias,
            conv_bias=conv_bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.mamba(x))

    def step(
        self,
        x: Tensor,
        conv_state: Tensor | None,
        ssm_state_re: Tensor | None,
        ssm_state_im: Tensor | None,
        prev_Bx_re: Tensor | None,
        prev_Bx_im: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Mamba-3 inference step with Real-valued computation.
        Returns: (output, conv_state, next_ssm_re, next_ssm_im,
                  current_Bx_re, current_Bx_im)
        """
        return self.mamba.step(x, conv_state, ssm_state_re, ssm_state_im, prev_Bx_re, prev_Bx_im)


class CustomMamba(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        n_heads: int = 8,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        bias: bool = False,
        conv_bias: bool = True,
    ) -> None:
        super().__init__()

        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.n_heads = n_heads
        self.head_dim = self.d_inner // n_heads
        dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=conv_bias,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # Mamba-3 x_proj outputs delta, B (re/im), C (re/im), lambda_gate, evo_gate
        out_dim = dt_rank + (4 * n_heads * d_state) + (2 * n_heads)
        self.x_proj = nn.Linear(self.d_inner, out_dim, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # Initialize dt_proj
        dt_init_std = dt_rank**-0.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # Complex A parameters: A_real (log space) and A_imag
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(n_heads, 1))
        )
        self.A_imag = nn.Parameter(torch.pi * torch.rand(n_heads, d_state))

        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        x_proj_trans = x_proj.transpose(1, 2)
        x_proj_conv = self.conv1d(x_proj_trans)[:, :, :seq_len]
        x_proj_conv = x_proj_conv.transpose(1, 2)
        x_proj_conv = F.silu(x_proj_conv)

        # A parameters (Real and Imaginary parts)
        A_re = -torch.exp(self.A_log.float())
        A_im = self.A_imag.float()

        proj_out = self.x_proj(x_proj_conv)

        # Split projections
        delta_raw, B_re, B_im, C_re, C_im, lambda_gate, evo_gate = torch.split(
            proj_out,
            [
                self.dt_rank,
                self.n_heads * self.d_state,
                self.n_heads * self.d_state,
                self.n_heads * self.d_state,
                self.n_heads * self.d_state,
                self.n_heads,
                self.n_heads,
            ],
            dim=-1,
        )

        delta = F.linear(delta_raw, self.dt_proj.weight, self.dt_proj.bias)
        delta = F.softplus(delta)

        # Reshape for MIMO processing
        delta = delta.view(batch, seq_len, self.n_heads, self.head_dim)
        B_re = B_re.view(batch, seq_len, self.n_heads, self.d_state)
        B_im = B_im.view(batch, seq_len, self.n_heads, self.d_state)
        C_re = C_re.view(batch, seq_len, self.n_heads, self.d_state)
        C_im = C_im.view(batch, seq_len, self.n_heads, self.d_state)
        lambda_gate = torch.sigmoid(lambda_gate).view(batch, seq_len, self.n_heads, 1)
        evo_gate = torch.sigmoid(evo_gate).view(batch, seq_len, self.n_heads, 1)
        x_mimo = x_proj_conv.view(batch, seq_len, self.n_heads, self.head_dim)

        y = self.selective_scan_v3(
            x_mimo, delta, A_re, A_im, B_re, B_im, C_re, C_im, lambda_gate, evo_gate
        )

        z = F.silu(z)
        output = y * z
        output = self.out_proj(output)
        return cast(Tensor, output)

    def selective_scan_v3(
        self,
        x: Tensor,
        delta: Tensor,
        A_re: Tensor,
        A_im: Tensor,
        B_re: Tensor,
        B_im: Tensor,
        C_re: Tensor,
        C_im: Tensor,
        lambda_gate: Tensor,
        evo_gate: Tensor,
    ) -> Tensor:
        B_size, L, H, head_dim = x.shape
        N = A_re.shape[-1]

        # 1. Discretization (Real Math)
        # dtA = delta * A
        dtA_re = delta.unsqueeze(-1) * A_re.view(1, 1, H, 1, N)
        dtA_im = delta.unsqueeze(-1) * A_im.view(1, 1, H, 1, N)

        # alpha = exp(dtA) = exp(dtA_re) * (cos(dtA_im) + i*sin(dtA_im))
        alpha_mag = torch.exp(torch.clamp(dtA_re, min=-20.0, max=20.0))
        alpha_re = alpha_mag * torch.cos(dtA_im)
        alpha_im = alpha_mag * torch.sin(dtA_im)

        # 2. SASM Input Bx = x * B
        # x is real, B is complex
        Bx_re = x.unsqueeze(-1) * B_re.unsqueeze(3)
        Bx_im = x.unsqueeze(-1) * B_im.unsqueeze(3)

        # 3. Trapezoidal Terms
        lg = lambda_gate.unsqueeze(-1)
        eg = evo_gate.unsqueeze(-1)

        # beta = (1 - lg) * delta * alpha
        beta_re = (1 - lg) * delta.unsqueeze(-1) * alpha_re
        beta_im = (1 - lg) * delta.unsqueeze(-1) * alpha_im

        # gamma = lg * delta * eg
        gamma = lg * delta.unsqueeze(-1) * eg

        Bx_re_prev = F.pad(Bx_re[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))
        Bx_im_prev = F.pad(Bx_im[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))

        # u = beta * Bx_prev + gamma * Bx
        # (beta_re + i*beta_im)*(Bx_re_prev + i*Bx_im_prev) =
        # (beta_re*Bx_re_prev - beta_im*Bx_im_prev) + i*(...)
        u_re = (beta_re * Bx_re_prev - beta_im * Bx_im_prev) + gamma * Bx_re
        u_im = (beta_re * Bx_im_prev + beta_im * Bx_re_prev) + gamma * Bx_im

        # 4. Scan using real-valued pairs
        alpha_re_flat = alpha_re.view(B_size, L, -1, N)
        alpha_im_flat = alpha_im.view(B_size, L, -1, N)
        u_re_flat = u_re.view(B_size, L, -1, N)
        u_im_flat = u_im.view(B_size, L, -1, N)

        hs_re, hs_im = associative_scan_real(alpha_re_flat, alpha_im_flat, u_re_flat, u_im_flat)

        # 5. Output: y = real(hs * conj(C))
        # (hs_re + i*hs_im) * (C_re - i*C_im) = (hs_re*C_re + hs_im*C_im) + i(...)
        hs_re = hs_re.view(B_size, L, H, head_dim, N)
        hs_im = hs_im.view(B_size, L, H, head_dim, N)
        y = (hs_re * C_re.unsqueeze(3) + hs_im * C_im.unsqueeze(3)).sum(dim=-1)

        x_flat = x.view(B_size, L, -1)
        y = y.view(B_size, L, -1) + self.D * x_flat
        return cast(Tensor, y)

    def step(
        self,
        x: Tensor,
        conv_state: Tensor | None,
        ssm_state_re: Tensor | None,
        ssm_state_im: Tensor | None,
        prev_Bx_re: Tensor | None,
        prev_Bx_im: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch = x.shape[0]
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        x_in = x_in.squeeze(1)
        z = z.squeeze(1)

        # Conv
        if conv_state is None:
            conv_state = torch.zeros(batch, self.d_inner, self.d_conv, device=x.device)
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state[:, :, -1] = x_in

        weight = self.conv1d.weight.squeeze(1)
        conv_bias = self.conv1d.bias if self.conv1d.bias is not None else torch.tensor(0.0)
        x_conv = torch.sum(conv_state * weight, dim=-1) + conv_bias
        x_conv = F.silu(x_conv)

        # SSM
        proj_out = self.x_proj(x_conv)
        delta_raw, B_re, B_im, C_re, C_im, lambda_gate, evo_gate = torch.split(
            proj_out,
            [
                self.dt_rank,
                self.n_heads * self.d_state,
                self.n_heads * self.d_state,
                self.n_heads * self.d_state,
                self.n_heads * self.d_state,
                self.n_heads,
                self.n_heads,
            ],
            dim=-1,
        )

        delta = F.linear(delta_raw, self.dt_proj.weight, self.dt_proj.bias)
        delta = F.softplus(delta).view(batch, self.n_heads, self.head_dim)
        lambda_gate = torch.sigmoid(lambda_gate).view(batch, self.n_heads, 1)
        evo_gate = torch.sigmoid(evo_gate).view(batch, self.n_heads, 1)

        A_re = -torch.exp(self.A_log.float())
        A_im = self.A_imag.float()

        B_re = B_re.view(batch, self.n_heads, self.d_state)
        B_im = B_im.view(batch, self.n_heads, self.d_state)
        C_re = C_re.view(batch, self.n_heads, self.d_state)
        C_im = C_im.view(batch, self.n_heads, self.d_state)

        # Discretization
        dtA_re = delta.unsqueeze(-1) * A_re.view(1, self.n_heads, 1, self.d_state)
        dtA_im = delta.unsqueeze(-1) * A_im.view(1, self.n_heads, 1, self.d_state)
        alpha_mag = torch.exp(torch.clamp(dtA_re, min=-20.0, max=20.0))
        alpha_re = alpha_mag * torch.cos(dtA_im)
        alpha_im = alpha_mag * torch.sin(dtA_im)

        # SASM Input Bx = x * B
        x_mimo = x_conv.view(batch, self.n_heads, self.head_dim)
        current_Bx_re = x_mimo.unsqueeze(-1) * B_re.unsqueeze(2)
        current_Bx_im = x_mimo.unsqueeze(-1) * B_im.unsqueeze(2)

        # Local copies for MyPy
        s_re = ssm_state_re
        s_im = ssm_state_im
        p_Bx_re = prev_Bx_re
        p_Bx_im = prev_Bx_im

        if s_re is None:
            s_re = torch.zeros(batch, self.n_heads, self.head_dim, self.d_state, device=x.device)
            s_im = torch.zeros(batch, self.n_heads, self.head_dim, self.d_state, device=x.device)
        else:
            assert s_im is not None

        if p_Bx_re is None:
            p_Bx_re = torch.zeros_like(current_Bx_re)
            p_Bx_im = torch.zeros_like(current_Bx_im)
        else:
            assert p_Bx_im is not None

        lg = lambda_gate.unsqueeze(-1)
        eg = evo_gate.unsqueeze(-1)

        beta_re = (1 - lg) * delta.unsqueeze(-1) * alpha_re
        beta_im = (1 - lg) * delta.unsqueeze(-1) * alpha_im
        gamma = lg * delta.unsqueeze(-1) * eg

        # Update SSM state: h = alpha * h + beta * prev_Bx + gamma * current_Bx
        # alpha * h
        h_re = alpha_re * s_re - alpha_im * s_im
        h_im = alpha_re * s_im + alpha_im * s_re
        # + beta * prev_Bx
        h_re = h_re + (beta_re * p_Bx_re - beta_im * p_Bx_im)
        h_im = h_im + (beta_re * p_Bx_im + beta_im * p_Bx_re)
        # + gamma * current_Bx
        h_re = h_re + gamma * current_Bx_re
        h_im = h_im + gamma * current_Bx_im

        # Output y = real(h * conj(C))
        y = (h_re * C_re.unsqueeze(2) + h_im * C_im.unsqueeze(2)).sum(dim=-1)
        y = y.view(batch, self.d_inner) + self.D * x_conv

        z = F.silu(z)
        output = y * z
        output = self.out_proj(output)

        return output.unsqueeze(1), conv_state, h_re, h_im, current_Bx_re, current_Bx_im


def associative_scan_real(
    A_re: Tensor, A_im: Tensor, X_re: Tensor, X_im: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Parallel associative scan using real-valued pairs to simulate complex recurrence.
    h_t = A_t * h_{t-1} + X_t
    """
    B, L, D, N = A_re.shape
    if L == 1:
        return X_re, X_im

    is_odd = L % 2 == 1
    if is_odd:
        A_re = torch.cat([A_re, torch.ones(B, 1, D, N, device=A_re.device)], dim=1)
        A_im = torch.cat([A_im, torch.zeros(B, 1, D, N, device=A_im.device)], dim=1)
        X_re = torch.cat([X_re, torch.zeros(B, 1, D, N, device=X_re.device)], dim=1)
        X_im = torch.cat([X_im, torch.zeros(B, 1, D, N, device=X_im.device)], dim=1)

    # Pairs
    A_re_even, A_im_even = A_re[:, 0::2], A_im[:, 0::2]
    X_re_even, X_im_even = X_re[:, 0::2], X_im[:, 0::2]
    A_re_odd, A_im_odd = A_re[:, 1::2], A_im[:, 1::2]
    X_re_odd, X_im_odd = X_re[:, 1::2], X_im[:, 1::2]

    # A_prime = A_odd * A_even
    A_prime_re = A_re_odd * A_re_even - A_im_odd * A_im_even
    A_prime_im = A_re_odd * A_im_even + A_im_odd * A_re_even

    # X_prime = A_odd * X_even + X_odd
    X_prime_re = (A_re_odd * X_re_even - A_im_odd * X_im_even) + X_re_odd
    X_prime_im = (A_re_odd * X_im_even + A_im_odd * X_re_even) + X_im_odd

    Y_re_prime, Y_im_prime = associative_scan_real(A_prime_re, A_prime_im, X_prime_re, X_prime_im)

    Y_re = torch.empty_like(A_re)
    Y_im = torch.empty_like(A_im)
    Y_re[:, 1::2] = Y_re_prime
    Y_im[:, 1::2] = Y_im_prime

    Y_re_prev = torch.cat([torch.zeros(B, 1, D, N, device=A_re.device), Y_re_prime[:, :-1]], dim=1)
    Y_im_prev = torch.cat([torch.zeros(B, 1, D, N, device=A_im.device), Y_im_prime[:, :-1]], dim=1)

    # Y_even = A_even * Y_prev + X_even
    Y_re[:, 0::2] = (A_re_even * Y_re_prev - A_im_even * Y_im_prev) + X_re_even
    Y_im[:, 0::2] = (A_re_even * Y_im_prev + A_im_even * Y_re_prev) + X_re_even

    if is_odd:
        return Y_re[:, :L], Y_im[:, :L]
    return Y_re, Y_im
