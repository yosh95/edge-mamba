import sys
import time
from pathlib import Path

import onnxruntime as ort
import torch
from tabulate import tabulate

sys.path.append(str(Path.cwd()))
from edge_mamba.numpy_model import MambaConfig, MambaNumpy
from edge_mamba.torch_model import Mamba


def benchmark() -> None:
    # Model configuration
    d_model = 128
    d_state = 16
    d_conv = 3
    n_heads = 8
    expand = 2
    d_inner = d_model * expand
    head_dim = d_inner // n_heads
    batch_size = 1
    num_steps = 1000
    warmup = 100

    onnx_path = Path("mamba_step_inference.onnx")
    if not onnx_path.exists():
        print(f"Error: {onnx_path} not found. Please run the export script first.")
        return

    # 1. Setup PyTorch
    device = torch.device("cpu")
    torch_model = Mamba(
        d_model=d_model, d_state=d_state, d_conv=d_conv, n_heads=n_heads, expand=expand
    ).to(device)
    torch_model.eval()

    # 2. Setup NumPy
    np_config = MambaConfig(
        d_model=d_model, d_state=d_state, d_conv=d_conv, n_heads=n_heads, expand=expand
    )
    np_model = MambaNumpy(np_config)

    # 3. Setup ONNX Runtime
    # Use CPU Execution Provider for fair comparison
    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # Dummy Inputs (Float32)
    x_torch = torch.randn(batch_size, 1, d_model).to(device)
    conv_state_torch = torch.zeros(batch_size, d_inner, d_conv).to(device)
    ssm_re_torch = torch.zeros(batch_size, n_heads, head_dim, d_state).to(device)
    ssm_im_torch = torch.zeros(batch_size, n_heads, head_dim, d_state).to(device)
    p_Bx_re_torch = torch.zeros(batch_size, n_heads, head_dim, d_state).to(device)
    p_Bx_im_torch = torch.zeros(batch_size, n_heads, head_dim, d_state).to(device)

    x_np = x_torch.numpy()
    conv_state_np = conv_state_torch.numpy()
    ssm_re_np = ssm_re_torch.numpy()
    ssm_im_np = ssm_im_torch.numpy()
    p_Bx_re_np = p_Bx_re_torch.numpy()
    p_Bx_im_np = p_Bx_im_torch.numpy()

    results = []

    # --- PyTorch Benchmark ---
    print("Benchmarking PyTorch...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = torch_model.step(
                x_torch,
                conv_state_torch,
                ssm_re_torch,
                ssm_im_torch,
                p_Bx_re_torch,
                p_Bx_im_torch,
            )

        start_time = time.perf_counter()
        for _ in range(num_steps):
            _ = torch_model.step(
                x_torch,
                conv_state_torch,
                ssm_re_torch,
                ssm_im_torch,
                p_Bx_re_torch,
                p_Bx_im_torch,
            )
        torch_time = (time.perf_counter() - start_time) / num_steps
    results.append(
        [
            "PyTorch (CPU)",
            f"{torch_time * 1000:.4f} ms",
            f"{1 / torch_time:.2f} steps/s",
        ]
    )

    # --- NumPy Benchmark ---
    print("Benchmarking NumPy...")
    for _ in range(warmup):
        _ = np_model.step(x_np, conv_state_np, ssm_re_np, ssm_im_np, p_Bx_re_np, p_Bx_im_np)

    start_time = time.perf_counter()
    for _ in range(num_steps):
        _ = np_model.step(x_np, conv_state_np, ssm_re_np, ssm_im_np, p_Bx_re_np, p_Bx_im_np)
    np_time = (time.perf_counter() - start_time) / num_steps
    results.append(["NumPy", f"{np_time * 1000:.4f} ms", f"{1 / np_time:.2f} steps/s"])

    # --- ONNX Runtime Benchmark ---
    print("Benchmarking ONNX Runtime...")
    ort_inputs = {
        "x": x_np,
        "conv_state": conv_state_np,
        "ssm_re": ssm_re_np,
        "ssm_im": ssm_im_np,
        "p_Bx_re": p_Bx_re_np,
        "p_Bx_im": p_Bx_im_np,
    }
    for _ in range(warmup):
        _ = ort_session.run(None, ort_inputs)

    start_time = time.perf_counter()
    for _ in range(num_steps):
        _ = ort_session.run(None, ort_inputs)
    onnx_time = (time.perf_counter() - start_time) / num_steps
    results.append(
        [
            "ONNX Runtime (CPU)",
            f"{onnx_time * 1000:.4f} ms",
            f"{1 / onnx_time:.2f} steps/s",
        ]
    )

    # Summary table
    table = tabulate(
        results,
        headers=["Implementation", "Latency (1 step)", "Throughput"],
        tablefmt="github",
    )
    print("\nBenchmark Results:")
    print(table)

    # Save to Markdown file
    with Path("benchmark_results.md").open("w") as f:
        f.write("# Mamba-3 Step Inference Benchmark\n\n")
        f.write("Tested on: CPU\n")
        f.write(
            f"Configuration: d_model={d_model}, d_state={d_state}, "
            f"d_conv={d_conv}, n_heads={n_heads}\n\n"
        )
        f.write(table)
        f.write("\n")


if __name__ == "__main__":
    benchmark()
