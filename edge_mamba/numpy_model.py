import math
from typing import cast

import numpy as np

# --- Activation Functions ---


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU activation function (swish)"""
    return cast(np.ndarray, x * (1.0 / (1.0 + np.exp(-x))))


def softplus(x: np.ndarray) -> np.ndarray:
    """Softplus activation function"""
    return cast(np.ndarray, np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function"""
    return cast(np.ndarray, 1.0 / (1.0 + np.exp(-x)))


class MambaConfig:
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
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.n_heads = n_heads
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.bias = bias
        self.conv_bias = conv_bias
        self.d_inner = expand * d_model
        self.head_dim = self.d_inner // n_heads
        self.dt_rank = math.ceil(d_model / 16)


class MambaNumpy:
    def __init__(self, config: MambaConfig):
        self.config = config
        self.params: dict[str, np.ndarray] = {}
        self._init_random_params()

    def _init_random_params(self) -> None:
        c = self.config
        rng = np.random.default_rng(42)
        # Using float32 for all parameters (64bit total for complex-like pairs)
        dtype = np.float32

        # in_proj weights
        self.params["in_proj.weight"] = (
            rng.standard_normal((2 * c.d_inner, c.d_model), dtype=dtype) * 0.02
        )
        if c.bias:
            self.params["in_proj.bias"] = np.zeros(2 * c.d_inner, dtype=dtype)

        # conv1d weights (depthwise)
        self.params["conv1d.weight"] = (
            rng.standard_normal((c.d_inner, c.d_conv), dtype=dtype) * 0.02
        )
        if c.conv_bias:
            self.params["conv1d.bias"] = np.zeros(c.d_inner, dtype=dtype)

        # x_proj (delta, B_re, B_im, C_re, C_im, lambda_gate, evo_gate)
        out_dim = c.dt_rank + (4 * c.d_state * c.n_heads) + (2 * c.n_heads)
        self.params["x_proj.weight"] = rng.standard_normal((out_dim, c.d_inner), dtype=dtype) * 0.02

        # dt_proj
        self.params["dt_proj.weight"] = (
            rng.standard_normal((c.d_inner, c.dt_rank), dtype=dtype) * 0.02
        )
        self.params["dt_proj.bias"] = rng.standard_normal(c.d_inner, dtype=dtype) * 0.02

        # A parameters (A_re and A_im)
        A_re_init = np.arange(1, c.d_state + 1, dtype=dtype)
        self.params["A_log"] = np.log(A_re_init)[None, :].repeat(c.n_heads, axis=0)
        self.params["A_imag"] = np.pi * rng.random((c.n_heads, c.d_state), dtype=dtype)

        # D parameter
        self.params["D"] = np.ones(c.d_inner, dtype=dtype)

        # out_proj
        self.params["out_proj.weight"] = (
            rng.standard_normal((c.d_model, c.d_inner), dtype=dtype) * 0.02
        )
        if c.bias:
            self.params["out_proj.bias"] = np.zeros(c.d_model, dtype=dtype)

    def load_state_dict(self, state_dict: dict[str, np.ndarray]) -> None:
        """Load weights from a dictionary."""
        for k, v in state_dict.items():
            if k in self.params:
                self.params[k] = v
            else:
                print(f"Warning: Param {k} not found in model.")

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, L, _ = x.shape
        c = self.config

        # 1. in_proj
        xz = x @ self.params["in_proj.weight"].T
        if "in_proj.bias" in self.params:
            xz += self.params["in_proj.bias"]
        x_in, z = np.split(xz, 2, axis=-1)

        # 2. conv1d
        x_t = x_in.transpose(0, 2, 1)
        padding = c.d_conv - 1
        x_padded = np.pad(x_t, ((0, 0), (0, 0), (padding, 0)), mode="constant")

        conv_out = np.zeros_like(x_t)
        weight = self.params["conv1d.weight"]
        bias = self.params["conv1d.bias"] if c.conv_bias else 0

        for i in range(L):
            window = x_padded[:, :, i : i + c.d_conv]
            conv_out[:, :, i] = np.sum(window * weight[None, :, :], axis=2) + bias

        x_conv_pre_act = conv_out.transpose(0, 2, 1)
        x_conv = silu(x_conv_pre_act)

        # 3. SSM (Real-valued calculation)
        y = self.ssm(x_conv)

        # 4. Gating
        z_act = silu(z)
        gated_y = y * z_act

        # 5. out_proj
        output = gated_y @ self.params["out_proj.weight"].T
        if "out_proj.bias" in self.params:
            output += self.params["out_proj.bias"]

        return cast(np.ndarray, output)

    def ssm(self, x: np.ndarray) -> np.ndarray:
        B, L, D_inner = x.shape
        c = self.config

        A_re = -np.exp(self.params["A_log"])
        A_im = self.params["A_imag"]

        proj_out = x @ self.params["x_proj.weight"].T

        offset = 0
        delta_proj = proj_out[..., offset : offset + c.dt_rank]
        offset += c.dt_rank
        B_re = proj_out[..., offset : offset + c.n_heads * c.d_state].reshape(
            B, L, c.n_heads, c.d_state
        )
        offset += c.n_heads * c.d_state
        B_im = proj_out[..., offset : offset + c.n_heads * c.d_state].reshape(
            B, L, c.n_heads, c.d_state
        )
        offset += c.n_heads * c.d_state
        C_re = proj_out[..., offset : offset + c.n_heads * c.d_state].reshape(
            B, L, c.n_heads, c.d_state
        )
        offset += c.n_heads * c.d_state
        C_im = proj_out[..., offset : offset + c.n_heads * c.d_state].reshape(
            B, L, c.n_heads, c.d_state
        )
        offset += c.n_heads * c.d_state
        lambda_gate = sigmoid(
            proj_out[..., offset : offset + c.n_heads].reshape(B, L, c.n_heads, 1)
        )
        offset += c.n_heads
        evo_gate = sigmoid(proj_out[..., offset : offset + c.n_heads].reshape(B, L, c.n_heads, 1))

        delta = softplus(
            delta_proj @ self.params["dt_proj.weight"].T + self.params["dt_proj.bias"]
        ).reshape(B, L, c.n_heads, c.head_dim)
        x_mimo = x.reshape(B, L, c.n_heads, c.head_dim)

        y = self.selective_scan_mimo(
            x_mimo,
            delta,
            A_re,
            A_im,
            B_re,
            B_im,
            C_re,
            C_im,
            lambda_gate,
            evo_gate,
        )

        y = y.reshape(B, L, D_inner)
        y = y + self.params["D"][None, None, :] * x

        return cast(np.ndarray, y)

    def selective_scan_mimo(
        self,
        x: np.ndarray,
        delta: np.ndarray,
        A_re: np.ndarray,
        A_im: np.ndarray,
        B_re: np.ndarray,
        B_im: np.ndarray,
        C_re: np.ndarray,
        C_im: np.ndarray,
        l_gate: np.ndarray,
        e_gate: np.ndarray,
    ) -> np.ndarray:
        B_size, L, H, head_dim = x.shape
        N = A_re.shape[-1]

        # 1. Discretization
        dtA_re = delta[:, :, :, :, None] * A_re[None, None, :, None, :]
        dtA_im = delta[:, :, :, :, None] * A_im[None, None, :, None, :]

        alpha_mag = np.exp(np.clip(dtA_re, -20.0, 20.0))
        alpha_re = alpha_mag * np.cos(dtA_im)
        alpha_im = alpha_mag * np.sin(dtA_im)

        # 2. Input Bx (x is real, B is complex)
        curr_Bx_re = x[:, :, :, :, None] * B_re[:, :, :, None, :]
        curr_Bx_im = x[:, :, :, :, None] * B_im[:, :, :, None, :]

        # 3. Trapezoidal update
        lg = l_gate[:, :, :, :, None]
        eg = e_gate[:, :, :, :, None]

        beta_re = (1.0 - lg) * delta[:, :, :, :, None] * alpha_re
        beta_im = (1.0 - lg) * delta[:, :, :, :, None] * alpha_im
        gamma = lg * delta[:, :, :, :, None] * eg

        # u = beta * Bx_prev + gamma * curr_Bx
        p_Bx_re = np.pad(curr_Bx_re[:, :-1], ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)))
        p_Bx_im = np.pad(curr_Bx_im[:, :-1], ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)))

        u_re = (beta_re * p_Bx_re - beta_im * p_Bx_im) + gamma * curr_Bx_re
        u_im = (beta_re * p_Bx_im + beta_im * p_Bx_re) + gamma * curr_Bx_im

        # 4. Sequential Scan (Simplification for Numpy/Inference focus)
        # For parallel scan, use the same logic as associative_scan_real
        h_re = np.zeros((B_size, L, H, head_dim, N), dtype=np.float32)
        h_im = np.zeros((B_size, L, H, head_dim, N), dtype=np.float32)

        last_re = np.zeros((B_size, H, head_dim, N), dtype=np.float32)
        last_im = np.zeros((B_size, H, head_dim, N), dtype=np.float32)

        for t in range(L):
            # h = alpha * h_prev + u
            re = alpha_re[:, t] * last_re - alpha_im[:, t] * last_im + u_re[:, t]
            im = alpha_re[:, t] * last_im + alpha_im[:, t] * last_re + u_im[:, t]
            h_re[:, t], h_im[:, t] = re, im
            last_re, last_im = re, im

        # 5. Output y = real(h * conj(C)) = h_re * C_re + h_im * C_im
        y = np.sum(h_re * C_re[:, :, :, None, :] + h_im * C_im[:, :, :, None, :], axis=-1)

        return cast(np.ndarray, y)

    def step(
        self,
        x: np.ndarray,
        conv_state: np.ndarray | None,
        ssm_re: np.ndarray | None,
        ssm_im: np.ndarray | None,
        p_Bx_re: np.ndarray | None,
        p_Bx_im: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = x.shape[0]
        c = self.config

        xz = x @ self.params["in_proj.weight"].T
        if "in_proj.bias" in self.params:
            xz += self.params["in_proj.bias"]
        x_in, z = np.split(xz, 2, axis=-1)
        x_in, z = x_in.squeeze(1), z.squeeze(1)

        if conv_state is None:
            conv_state = np.zeros((batch, c.d_inner, c.d_conv), dtype=np.float32)
        conv_state = np.roll(conv_state, shift=-1, axis=-1)
        conv_state[:, :, -1] = x_in
        x_conv = np.sum(conv_state * self.params["conv1d.weight"][None, :, :], axis=2)
        if c.conv_bias:
            x_conv += self.params["conv1d.bias"]
        x_conv = silu(x_conv)

        proj_out = x_conv @ self.params["x_proj.weight"].T
        offset = 0
        delta_proj = proj_out[..., offset : offset + c.dt_rank]
        offset += c.dt_rank
        B_re = proj_out[..., offset : offset + c.n_heads * c.d_state].reshape(
            batch, c.n_heads, c.d_state
        )
        offset += c.n_heads * c.d_state
        B_im = proj_out[..., offset : offset + c.n_heads * c.d_state].reshape(
            batch, c.n_heads, c.d_state
        )
        offset += c.n_heads * c.d_state
        C_re = proj_out[..., offset : offset + c.n_heads * c.d_state].reshape(
            batch, c.n_heads, c.d_state
        )
        offset += c.n_heads * c.d_state
        C_im = proj_out[..., offset : offset + c.n_heads * c.d_state].reshape(
            batch, c.n_heads, c.d_state
        )
        offset += c.n_heads * c.d_state
        l_gate = sigmoid(proj_out[..., offset : offset + c.n_heads]).reshape(batch, c.n_heads, 1)
        offset += c.n_heads
        e_gate = sigmoid(proj_out[..., offset : offset + c.n_heads]).reshape(batch, c.n_heads, 1)

        delta = softplus(
            delta_proj @ self.params["dt_proj.weight"].T + self.params["dt_proj.bias"]
        ).reshape(batch, c.n_heads, c.head_dim)

        A_re = -np.exp(self.params["A_log"])
        A_im = self.params["A_imag"]

        # Discretization
        dtA_re = delta[..., None] * A_re[None, :, None, :]
        dtA_im = delta[..., None] * A_im[None, :, None, :]
        alpha_mag = np.exp(np.clip(dtA_re, -20, 20))
        alpha_re = alpha_mag * np.cos(dtA_im)
        alpha_im = alpha_mag * np.sin(dtA_im)

        curr_Bx_re = x_conv.reshape(batch, c.n_heads, c.head_dim)[..., None] * B_re[:, :, None, :]
        curr_Bx_im = x_conv.reshape(batch, c.n_heads, c.head_dim)[..., None] * B_im[:, :, None, :]

        if ssm_re is None:
            ssm_re = np.zeros((batch, c.n_heads, c.head_dim, c.d_state), dtype=np.float32)
            ssm_im = np.zeros((batch, c.n_heads, c.head_dim, c.d_state), dtype=np.float32)
        if p_Bx_re is None:
            p_Bx_re, p_Bx_im = np.zeros_like(curr_Bx_re), np.zeros_like(curr_Bx_im)

        beta_re = (1.0 - l_gate[..., None]) * delta[..., None] * alpha_re
        beta_im = (1.0 - l_gate[..., None]) * delta[..., None] * alpha_im
        gamma = l_gate[..., None] * delta[..., None] * e_gate[..., None]

        # Update state: h = alpha*h + beta*prev_Bx + gamma*curr_Bx
        new_re = (
            (alpha_re * ssm_re - alpha_im * ssm_im)
            + (beta_re * p_Bx_re - beta_im * p_Bx_im)
            + (gamma * curr_Bx_re)
        )
        new_im = (
            (alpha_re * ssm_im + alpha_im * ssm_re)
            + (beta_re * p_Bx_im + beta_im * p_Bx_re)
            + (gamma * curr_Bx_im)
        )

        y = np.sum(new_re * C_re[:, :, None, :] + new_im * C_im[:, :, None, :], axis=-1)
        y = y.reshape(batch, c.d_inner) + self.params["D"][None, :] * x_conv

        output = (y * silu(z)) @ self.params["out_proj.weight"].T
        if "out_proj.bias" in self.params:
            output += self.params["out_proj.bias"]

        return output[:, None, :], conv_state, new_re, new_im, curr_Bx_re, curr_Bx_im
