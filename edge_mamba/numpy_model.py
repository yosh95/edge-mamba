import math

import numpy as np


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class MambaConfig:
    def __init__(
        self,
        d_model=128,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_min=0.001,
        dt_max=0.1,
        dt_scale=1.0,
        dt_init_floor=1e-4,
        bias=False,
        conv_bias=True,
    ):
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.bias = bias
        self.conv_bias = conv_bias
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)


class MambaNumpy:
    def __init__(self, config: MambaConfig):
        self.config = config
        self.params: dict[str, np.ndarray] = {}
        self._init_random_params()

    def _init_random_params(self):
        c = self.config
        rng = np.random.default_rng(42)

        self.params["in_proj.weight"] = (
            rng.standard_normal((2 * c.d_inner, c.d_model)) * 0.02
        )
        if c.bias:
            self.params["in_proj.bias"] = np.zeros(2 * c.d_inner)

        self.params["conv1d.weight"] = rng.standard_normal((c.d_inner, c.d_conv)) * 0.02
        if c.conv_bias:
            self.params["conv1d.bias"] = np.zeros(c.d_inner)

        # Mamba-3 x_proj: dt_rank + 4*d_state + 1
        out_dim = c.dt_rank + 4 * c.d_state + 1
        self.params["x_proj.weight"] = rng.standard_normal((out_dim, c.d_inner)) * 0.02

        self.params["dt_proj.weight"] = (
            rng.standard_normal((c.d_inner, c.dt_rank)) * 0.02
        )
        self.params["dt_proj.bias"] = rng.standard_normal(c.d_inner) * 0.02

        # Complex A
        A_real = np.arange(1, c.d_state + 1, dtype=np.float32)
        self.params["A_log"] = np.log(A_real)[None, :].repeat(c.d_inner, axis=0)
        self.params["A_imag"] = np.pi * rng.random((c.d_inner, c.d_state))

        self.params["D"] = np.ones(c.d_inner)

        self.params["out_proj.weight"] = (
            rng.standard_normal((c.d_model, c.d_inner)) * 0.02
        )
        if c.bias:
            self.params["out_proj.bias"] = np.zeros(c.d_model)

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            clean_k = k.replace("mamba.", "")
            if clean_k in self.params:
                v_np = v.detach().cpu().numpy() if hasattr(v, "detach") else v
                if v_np.shape == self.params[clean_k].shape:
                    self.params[clean_k] = v_np
                elif clean_k == "conv1d.weight" and v_np.ndim == 3:
                    self.params[clean_k] = v_np.squeeze(1)

    def forward(self, x):
        B, L, _ = x.shape
        c = self.config

        xz = x @ self.params["in_proj.weight"].T
        if "in_proj.bias" in self.params:
            xz += self.params["in_proj.bias"]

        x_in, z = np.split(xz, 2, axis=-1)

        # Conv1d
        x_t = x_in.transpose(0, 2, 1)
        padding = c.d_conv - 1
        x_padded = np.pad(x_t, ((0, 0), (0, 0), (padding, 0)), mode="constant")

        conv_out = np.zeros_like(x_t)
        weight = self.params["conv1d.weight"]
        bias = self.params["conv1d.bias"] if c.conv_bias else 0

        for i in range(L):
            window = x_padded[:, :, i : i + c.d_conv]
            conv_out[:, :, i] = np.sum(window * weight[None, :, :], axis=2) + bias

        x_conv = conv_out.transpose(0, 2, 1)
        x_conv = silu(x_conv)

        # SSM
        y = self.ssm(x_conv)

        # Gating
        z = silu(z)
        output = y * z

        output = output @ self.params["out_proj.weight"].T
        if "out_proj.bias" in self.params:
            output += self.params["out_proj.bias"]

        return output

    def ssm(self, x):
        B, L, D_inner = x.shape
        c = self.config

        A_real = -np.exp(self.params["A_log"])
        A_imag = self.params["A_imag"]
        A = A_real + 1j * A_imag

        proj_out = x @ self.params["x_proj.weight"].T

        delta_proj = proj_out[..., : c.dt_rank]
        B_re = proj_out[..., c.dt_rank : c.dt_rank + c.d_state]
        B_im = proj_out[..., c.dt_rank + c.d_state : c.dt_rank + 2 * c.d_state]
        C_re = proj_out[..., c.dt_rank + 2 * c.d_state : c.dt_rank + 3 * c.d_state]
        C_im = proj_out[..., c.dt_rank + 3 * c.d_state : c.dt_rank + 4 * c.d_state]
        lambda_gate = sigmoid(proj_out[..., -1:])

        delta = (
            delta_proj @ self.params["dt_proj.weight"].T + self.params["dt_proj.bias"]
        )
        delta = softplus(delta)

        B_complex = B_re + 1j * B_im
        C_complex = C_re + 1j * C_im

        y = self.selective_scan_v3(x, delta, A, B_complex, C_complex, lambda_gate)
        return y

    def selective_scan_v3(self, x, delta, A, B, C, lambda_gate):
        bs, L, d_inner = x.shape
        d_state = self.config.d_state

        # alpha = exp(delta * A)
        alpha = np.exp(delta[:, :, :, None] * A[None, None, :, :])  # (B, L, D, N)

        # Bx_t = B_t * x_t (MIMO broadcasting)
        # B: (B, L, N), x: (B, L, D) -> (B, L, D, N)
        Bx = B[:, :, None, :] * x[:, :, :, None]

        # Trapezoidal Terms
        beta = (1 - lambda_gate[:, :, :, None]) * delta[:, :, :, None] * alpha
        gamma = lambda_gate[:, :, :, None] * delta[:, :, :, None]

        # Shift Bx for previous term
        Bx_prev = np.zeros_like(Bx)
        Bx_prev[:, 1:] = Bx[:, :-1]

        u = beta * Bx_prev + gamma * Bx

        h = np.zeros((bs, d_inner, d_state), dtype=np.complex64)
        ys = []

        for t in range(L):
            h = alpha[:, t] * h + u[:, t]
            # y = Re(sum(h * conj(C)))
            y_curr = np.sum(h * np.conj(C[:, t, None, :]), axis=-1)
            ys.append(y_curr.real)

        y = np.stack(ys, axis=1)
        y = y + self.params["D"][None, None, :] * x

        return y

    def step(self, x, conv_state, ssm_state, prev_Bx):
        # x: (B, D_model)
        c = self.config

        xz = x @ self.params["in_proj.weight"].T
        if "in_proj.bias" in self.params:
            xz += self.params["in_proj.bias"]
        x_in, z = np.split(xz, 2, axis=-1)

        # Conv
        if conv_state is None:
            conv_state = np.zeros((x.shape[0], c.d_inner, c.d_conv))
        conv_state = np.roll(conv_state, shift=-1, axis=-1)
        conv_state[:, :, -1] = x_in

        weight = self.params["conv1d.weight"]
        bias = self.params["conv1d.bias"] if c.conv_bias else 0
        x_conv = np.sum(conv_state * weight[None, :, :], axis=-1) + bias
        x_conv = silu(x_conv)

        # SSM
        proj_out = x_conv @ self.params["x_proj.weight"].T
        delta_proj = proj_out[..., : c.dt_rank]
        B_re = proj_out[..., c.dt_rank : c.dt_rank + c.d_state]
        B_im = proj_out[..., c.dt_rank + c.d_state : c.dt_rank + 2 * c.d_state]
        C_re = proj_out[..., c.dt_rank + 2 * c.d_state : c.dt_rank + 3 * c.d_state]
        C_im = proj_out[..., c.dt_rank + 3 * c.d_state : c.dt_rank + 4 * c.d_state]
        lambda_gate = sigmoid(proj_out[..., -1:])

        delta = (
            delta_proj @ self.params["dt_proj.weight"].T + self.params["dt_proj.bias"]
        )
        delta = softplus(delta)

        A = -np.exp(self.params["A_log"]) + 1j * self.params["A_imag"]
        B = B_re + 1j * B_im
        C = C_re + 1j * C_im

        alpha = np.exp(delta[:, :, None] * A[None, :, :])
        current_Bx = B[:, None, :] * x_conv[:, :, None]

        if ssm_state is None:
            ssm_state = np.zeros((x.shape[0], c.d_inner, c.d_state), dtype=np.complex64)
        if prev_Bx is None:
            prev_Bx = np.zeros_like(current_Bx)

        # Trapezoidal
        beta = (1 - lambda_gate[:, :, None]) * delta[:, :, None] * alpha
        gamma = lambda_gate[:, :, None] * delta[:, :, None]

        ssm_state = alpha * ssm_state + beta * prev_Bx + gamma * current_Bx

        y = np.sum(ssm_state * np.conj(C[:, None, :]), axis=-1).real
        y = y + self.params["D"] * x_conv

        output = (y * silu(z)) @ self.params["out_proj.weight"].T
        if "out_proj.bias" in self.params:
            output += self.params["out_proj.bias"]

        return output, conv_state, ssm_state, current_Bx
