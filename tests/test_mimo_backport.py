import torch

from edge_mamba.torch_model import Mamba


def test_mamba_mimo() -> None:
    d_model = 128
    d_state = 16
    n_heads = 8
    seq_len = 64
    batch_size = 2

    model = Mamba(d_model=d_model, d_state=d_state, n_heads=n_heads)
    x = torch.randn(batch_size, seq_len, d_model)

    # Test forward
    output = model(x)
    print(f"Forward output shape: {output.shape}")
    assert output.shape == (batch_size, seq_len, d_model)

    # Test step (inference)
    conv_state = None
    ssm_re = None
    ssm_im = None
    p_Bx_re = None
    p_Bx_im = None

    for i in range(seq_len):
        step_x = x[:, i : i + 1, :]
        res = model.step(step_x, conv_state, ssm_re, ssm_im, p_Bx_re, p_Bx_im)
        step_output, conv_state, ssm_re, ssm_im, p_Bx_re, p_Bx_im = res

    print(f"Last step output shape: {step_output.shape}")
    assert step_output.shape == (batch_size, 1, d_model)
    print("MIMO Backport Test Passed!")


if __name__ == "__main__":
    test_mamba_mimo()
