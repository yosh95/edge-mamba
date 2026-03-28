# Mamba-3 Step Inference Benchmark

Tested on: CPU
Configuration: d_model=128, d_state=16, d_conv=3, n_heads=8

| Implementation     | Latency (1 step)   | Throughput      |
|--------------------|--------------------|-----------------|
| PyTorch (CPU)      | 0.3834 ms          | 2608.44 steps/s |
| NumPy              | 0.1493 ms          | 6699.37 steps/s |
| ONNX Runtime (CPU) | 0.1590 ms          | 6289.36 steps/s |
