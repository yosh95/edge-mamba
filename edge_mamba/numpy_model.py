import math

import numpy as np


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def softplus(x):
    return np.log1p(np.exp(x))


class MambaConfig:
    def __init__(
        self,
        d_model=128,
        d_state=4,
        d_conv=3,
        expand=1,
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
        # Parameters should be loaded using load_state_dict
        # We initialize them randomly here for standalone usage
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

        out_dim = c.dt_rank + 2 * c.d_state
        self.params["x_proj.weight"] = rng.standard_normal((out_dim, c.d_inner)) * 0.02

        self.params["dt_proj.weight"] = (
            rng.standard_normal((c.d_inner, c.dt_rank)) * 0.02
        )
        self.params["dt_proj.bias"] = rng.standard_normal(c.d_inner) * 0.02

        # Initialize A_log ensuring stability
        A = np.arange(1, c.d_state + 1, dtype=np.float32)
        A_log = np.log(A)[None, :].repeat(c.d_inner, axis=0)
        self.params["A_log"] = A_log

        self.params["D"] = np.ones(c.d_inner)

        self.params["out_proj.weight"] = (
            rng.standard_normal((c.d_model, c.d_inner)) * 0.02
        )
        if c.bias:
            self.params["out_proj.bias"] = np.zeros(c.d_model)

    def load_state_dict(self, state_dict):
        """
        Load weights from a dictionary (e.g. from PyTorch state_dict).
        PyTorch Conv1d weight is (out, 1, k) for groups=out. We expect (out, k).
        """
        for k, v in state_dict.items():
            # Adjust key names if necessary (e.g. removing 'mamba.' prefix)
            clean_k = k.replace("mamba.", "")

            if clean_k in self.params:
                target_shape = self.params[clean_k].shape
                v_np = v.detach().cpu().numpy() if hasattr(v, "detach") else v

                if v_np.shape == target_shape:
                    self.params[clean_k] = v_np
                elif clean_k == "conv1d.weight" and v_np.ndim == 3:
                    self.params[clean_k] = v_np.squeeze(1)  # (D, 1, K) -> (D, K)
                elif (
                    clean_k == "in_proj.weight"
                    or clean_k == "out_proj.weight"
                    or clean_k == "x_proj.weight"
                    or clean_k == "dt_proj.weight"
                ):
                    if v_np.shape == target_shape:
                        self.params[clean_k] = v_np
                    elif v_np.ndim == 2 and v_np.shape == target_shape[::-1]:
                        # Depending on how Linear is exported (sometimes Transposed)
                        # PyTorch Linear weight is (Out, In).
                        # In our numpy code we use x @ W.T, so we expect (Out, In)
                        # stored in params.
                        self.params[clean_k] = v_np
                    else:
                        print(
                            f"Warning: Shape mismatch for {clean_k}. "
                            f"Expected {target_shape}, got {v_np.shape}"
                        )
                else:
                    print(
                        f"Warning: Shape mismatch for {clean_k}. "
                        f"Expected {target_shape}, got {v_np.shape}"
                    )

    def forward(self, x):
        """
        x: (B, L, D_model)
        """
        B, L, _ = x.shape
        c = self.config

        # in_proj: (B, L, D) @ (2D_in, D).T -> (B, L, 2D_in)
        xz = x @ self.params["in_proj.weight"].T
        if c.bias and "in_proj.bias" in self.params:
            xz += self.params["in_proj.bias"]

        x, z = np.split(xz, 2, axis=-1)

        # Conv1d
        x_t = x.transpose(0, 2, 1)  # (B, D_in, L)
        padding = c.d_conv - 1
        x_padded = np.pad(x_t, ((0, 0), (0, 0), (padding, 0)), mode="constant")

        conv_out = np.zeros_like(x_t)
        weight = self.params["conv1d.weight"]  # (D_in, K)
        bias = self.params["conv1d.bias"] if c.conv_bias else 0

        # Naive sliding window for depthwise conv
        for i in range(L):
            window = x_padded[:, :, i : i + c.d_conv]  # (B, D_in, K)
            # Element-wise mult with weight (broadcast over B) and sum over K
            conv_out[:, :, i] = np.sum(window * weight[None, :, :], axis=2) + bias

        x = conv_out.transpose(0, 2, 1)  # (B, L, D_in)
        x = silu(x)

        # SSM
        y = self.ssm(x)

        # Gating
        z = silu(z)
        output = y * z

        # Out proj
        output = output @ self.params["out_proj.weight"].T
        if c.bias and "out_proj.bias" in self.params:
            output += self.params["out_proj.bias"]

        return output

    def ssm(self, x):
        B, L, D_inner = x.shape
        c = self.config

        A = -np.exp(self.params["A_log"])
        D = self.params["D"]

        deltaBC = x @ self.params["x_proj.weight"].T

        delta_proj, B_ssm, C_ssm = np.split(
            deltaBC, [c.dt_rank, c.dt_rank + c.d_state], axis=-1
        )

        delta = (
            delta_proj @ self.params["dt_proj.weight"].T + self.params["dt_proj.bias"]
        )
        delta = softplus(delta)

        y = self.selective_scan(x, delta, A, B_ssm, C_ssm, D)
        return y

    def selective_scan(self, x, delta, A, B, C, D):
        bs, L, d_inner = x.shape
        d_state = self.config.d_state

        # deltaA: (B, L, D_inner, D_state)
        deltaA = np.exp(delta[:, :, :, None] * A[None, None, :, :])

        # deltaB: (B, L, D_inner, D_state)
        deltaB = delta[:, :, :, None] * B[:, :, None, :]

        # deltaB_x: (B, L, D_inner, D_state)
        deltaB_x = deltaB * x[:, :, :, None]

        h = np.zeros((bs, d_inner, d_state))
        ys = []

        for t in range(L):
            h = deltaA[:, t] * h + deltaB_x[:, t]
            # y_t: (B, D_inner)
            y_curr = np.sum(h * C[:, t, None, :], axis=-1)
            ys.append(y_curr)

        y = np.stack(ys, axis=1)
        y = y + D[None, None, :] * x

        return y
