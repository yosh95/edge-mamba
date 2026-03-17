import math

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
        return self.mamba(x)  # type: ignore

    def step(
        self,
        x: Tensor,
        conv_state: Tensor | None,
        ssm_state: Tensor | None,
        prev_Bx: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Mamba-3 inference step with Trapezoidal Discretization.
        Returns: (output, conv_state, ssm_state, current_Bx)
        """
        return self.mamba.step(x, conv_state, ssm_state, prev_Bx)


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

        # Mamba-3 x_proj: delta, complex B (re/im), complex C (re/im),
        # lambda_gate, evo_gate for each head
        out_dim = dt_rank + (4 * n_heads * d_state) + (2 * n_heads)
        self.x_proj = nn.Linear(self.d_inner, out_dim, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # Initialize dt_proj
        dt_init_std = dt_rank**-0.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # Mamba-3 Complex A: A_real (log space) and A_imag
        # One per head for MIMO
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float32).repeat(n_heads, 1)
            )
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

        # SSM Params
        A_real = -torch.exp(self.A_log.float())
        A_imag = self.A_imag.float()
        A = torch.complex(A_real, A_imag)  # (n_heads, d_state)

        proj_out = self.x_proj(x_proj_conv)

        # Split proj_out for MIMO
        # delta_raw: (B, L, dt_rank)
        # B, C: (B, L, n_heads * d_state)
        # gates: (B, L, n_heads)
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
        delta = F.softplus(delta)  # (B, L, d_inner)

        # Reshape for MIMO
        delta = delta.view(batch, seq_len, self.n_heads, self.head_dim)
        B = torch.complex(B_re, B_im).view(batch, seq_len, self.n_heads, self.d_state)
        C = torch.complex(C_re, C_im).view(batch, seq_len, self.n_heads, self.d_state)
        lambda_gate = torch.sigmoid(lambda_gate).view(batch, seq_len, self.n_heads, 1)
        evo_gate = torch.sigmoid(evo_gate).view(batch, seq_len, self.n_heads, 1)
        x_mimo = x_proj_conv.view(batch, seq_len, self.n_heads, self.head_dim)

        y = self.selective_scan_v3(x_mimo, delta, A, B, C, lambda_gate, evo_gate)

        z = F.silu(z)
        output = y * z
        output = self.out_proj(output)
        return output  # type: ignore

    def selective_scan_v3(
        self,
        x: Tensor,  # (B, L, H, head_dim)
        delta: Tensor,  # (B, L, H, head_dim)
        A: Tensor,  # (H, N)
        B: Tensor,  # (B, L, H, N)
        C: Tensor,  # (B, L, H, N)
        lambda_gate: Tensor,  # (B, L, H, 1)
        evo_gate: Tensor,  # (B, L, H, 1)
    ) -> Tensor:
        B_size, L, H, head_dim = x.shape
        N = A.shape[-1]

        # 1. Discretization
        # delta: (B, L, H, head_dim) -> (B, L, H, head_dim, 1)
        # A: (H, N) -> (1, 1, H, 1, N)
        dtA = delta.unsqueeze(-1) * A.view(1, 1, H, 1, N)
        # Clip only the real part to prevent numerical explosion
        dtA_real = torch.clamp(dtA.real, min=-20.0, max=20.0)
        alpha = torch.exp(torch.complex(dtA_real, dtA.imag))  # (B, L, H, head_dim, N)

        # 2. SASM Input Bx_t = x_t * B_t
        # x: (B, L, H, head_dim) -> (B, L, H, head_dim, 1)
        # B: (B, L, H, N) -> (B, L, H, 1, N)
        Bx = x.unsqueeze(-1) * B.unsqueeze(3)  # (B, L, H, head_dim, N)

        # 3. Trapezoidal Terms with Evolution Gate
        # lambda_gate: (B, L, H, 1) -> (B, L, H, 1, 1)
        lg = lambda_gate.unsqueeze(-1)
        eg = evo_gate.unsqueeze(-1)

        beta = (1 - lg) * delta.unsqueeze(-1) * alpha
        gamma = lg * delta.unsqueeze(-1) * eg

        Bx_prev = F.pad(Bx[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))
        u = beta * Bx_prev + gamma * Bx  # (B, L, H, head_dim, N)

        # 4. Scan
        # Flatten (H, head_dim) to d_inner for associative_scan_complex
        alpha_flat = alpha.view(B_size, L, -1, N)
        u_flat = u.view(B_size, L, -1, N)
        hs = associative_scan_complex(alpha_flat, u_flat)  # (B, L, d_inner, N)

        # 5. Output (C is Read-Key)
        # hs: (B, L, H, head_dim, N)
        # C: (B, L, H, N) -> (B, L, H, 1, N)
        hs = hs.view(B_size, L, H, head_dim, N)
        y_complex = (hs * C.unsqueeze(3).conj()).sum(dim=-1)  # (B, L, H, head_dim)
        y = y_complex.real
        y = y.view(B_size, L, -1)  # (B, L, d_inner)

        # Add skip connection (D)
        # x is originally (B, L, d_inner) before being viewed as mimo
        x_flat = x.view(B_size, L, -1)
        y = y + self.D * x_flat
        return y  # type: ignore

    def step(
        self,
        x: Tensor,
        conv_state: Tensor | None,
        ssm_state: Tensor | None,
        prev_Bx: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch = x.shape[0]
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)

        # Squeeze sequence dimension for processing one token
        x_in = x_in.squeeze(1)
        z = z.squeeze(1)

        # Conv
        if conv_state is None:
            conv_state = torch.zeros(batch, self.d_inner, self.d_conv, device=x.device)
        conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
        conv_state[:, :, -1] = x_in

        weight = self.conv1d.weight.squeeze(1)
        conv_bias = self.conv1d.bias if self.conv1d.bias is not None else 0
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

        A_real = -torch.exp(self.A_log.float())
        A_imag = self.A_imag.float()
        A = torch.complex(A_real, A_imag)  # (n_heads, d_state)

        B = torch.complex(B_re, B_im).view(batch, self.n_heads, self.d_state)
        C = torch.complex(C_re, C_im).view(batch, self.n_heads, self.d_state)

        # Discretization
        # delta: (batch, n_heads, head_dim) -> (batch, n_heads, head_dim, 1)
        # A: (n_heads, d_state) -> (1, n_heads, 1, d_state)
        dtA = delta.unsqueeze(-1) * A.view(1, self.n_heads, 1, self.d_state)
        dtA_real = torch.clamp(dtA.real, min=-20.0, max=20.0)
        alpha = torch.exp(torch.complex(dtA_real, dtA.imag))

        # SASM Input Bx = x * B
        x_mimo = x_conv.view(batch, self.n_heads, self.head_dim)
        current_Bx = x_mimo.unsqueeze(-1) * B.unsqueeze(2)  # (B, H, head_dim, N)

        if ssm_state is None:
            ssm_state = torch.zeros(
                batch,
                self.n_heads,
                self.head_dim,
                self.d_state,
                dtype=torch.complex64,
                device=x.device,
            )
        if prev_Bx is None:
            prev_Bx = torch.zeros_like(current_Bx)

        # Mamba-3 Trapezoidal update with Evolution Gate
        lg = lambda_gate.unsqueeze(-1)
        eg = evo_gate.unsqueeze(-1)

        beta = (1 - lg) * delta.unsqueeze(-1) * alpha
        gamma = lg * delta.unsqueeze(-1) * eg

        ssm_state = alpha * ssm_state + beta * prev_Bx + gamma * current_Bx

        y_complex = (ssm_state * C.unsqueeze(2).conj()).sum(dim=-1)
        y = y_complex.real
        y = y.view(batch, self.d_inner) + self.D * x_conv

        z = F.silu(z)
        output = y * z
        output = self.out_proj(output)

        return output.unsqueeze(1), conv_state, ssm_state, current_Bx  # type: ignore


def associative_scan_complex(A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Parallelize Mamba's recurrence h_t = A_t * h_{t-1} + X_t using Associative Scan.
    A: (B, L, D, N) - Complex coefficients
    X: (B, L, D, N) - Complex inputs
    """
    B, L, D, N = A.shape
    if L == 1:
        return X

    # Handle odd lengths by padding
    is_odd = L % 2 == 1
    if is_odd:
        A_pad = torch.cat(
            [A, torch.ones(B, 1, D, N, device=A.device, dtype=A.dtype)], dim=1
        )
        X_pad = torch.cat(
            [X, torch.zeros(B, 1, D, N, device=A.device, dtype=A.dtype)], dim=1
        )
    else:
        A_pad, X_pad = A, X

    # Combine adjacent pairs
    A_even = A_pad[:, 0::2]
    X_even = X_pad[:, 0::2]
    A_odd = A_pad[:, 1::2]
    X_odd = X_pad[:, 1::2]

    A_prime = A_odd * A_even
    X_prime = A_odd * X_even + X_odd

    # Recursive scan
    Y_prime = associative_scan_complex(A_prime, X_prime)

    # Expand results
    Y = torch.empty_like(A_pad)
    Y[:, 1::2] = Y_prime
    # y_{2t} = a_{2t} * y_{2t-1} + x_{2t} (with y_{-1} = 0)
    Y_prev = torch.cat(
        [torch.zeros(B, 1, D, N, device=A.device, dtype=A.dtype), Y_prime[:, :-1]],
        dim=1,
    )
    Y[:, 0::2] = A_even * Y_prev + X_even

    if is_odd:
        return Y[:, :L]
    return Y
