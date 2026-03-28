# Mamba Edge Implementation / Mamba エッジ実装

[English](#english) | [日本語](#japanese)

<a name="english"></a>
## English Description

This project provides a lightweight, **real-valued (float32)** implementation of the Mamba architecture, specifically optimized for edge devices and real-time IDS (Intrusion Detection Systems).

### Key Features
- **Real-valued Math**: 32-bit float real-valued pairs (64-bit complex equivalent) for maximum compatibility with ONNX and edge accelerators.
- **Inference Optimized**: Includes a high-performance `step` function for processing data one packet/token at a time.
- **Multi-Backend**: Support for **PyTorch**, **NumPy**, and **ONNX Runtime**.
- **Mathematical Consistency**: Verified 10^-7 precision between all implementations.

### Performance (IDS Step Inference)
Tested with `d_model=128, d_state=16, n_heads=8` on CPU.

| Backend | Latency (1 step) | Throughput |
| :--- | :--- | :--- |
| **PyTorch (CPU)** | ~0.38 ms | 2,600 steps/s |
| **NumPy** | **~0.15 ms** | **6,700 steps/s** |
| **ONNX Runtime** | **~0.16 ms** | **6,300 steps/s** |

### Usage (Makefile)
```bash
make help      # Show available commands
make test      # Verify mathematical consistency
make export    # Export 'step' function to ONNX (mamba_step_inference.onnx)
make benchmark # Run performance benchmark
```

---

<a name="japanese"></a>
## 日本語説明

このプロジェクトは、エッジデバイスやリアルタイムIDS（侵入検知システム）向けに最適化された、**実数演算（float32）ベース**のMamba実装を提供します。

### 特徴
- **実数演算の徹底**: 複素数演算を排除し、32bit実数ペア（64bit複素数相当）で再構成。ONNXやエッジNPUとの高い親和性を確保。
- **推論特化型**: パケットやトークンを1つずつリアルタイムに処理する `step` 関数に最適化。
- **マルチバックエンド**: **PyTorch**, **NumPy**, **ONNX Runtime** の3つをサポート。
- **数学的等価性**: 全実装間で 10^-7 以上の精度で等価であることを検証済み。

### パフォーマンス (IDS ステップ推論)
`d_model=128, d_state=16, n_heads=8` を CPU で実行した結果です。

| バックエンド | レイテンシ (1 step) | スループット |
| :--- | :--- | :--- |
| **PyTorch (CPU)** | 約 0.38 ms | 2,600 steps/s |
| **NumPy** | **約 0.15 ms** | **6,700 steps/s** |
| **ONNX Runtime** | **約 0.16 ms** | **6,300 steps/s** |

*1パケットずつの処理において、NumPy/ONNX版はPyTorch(CPU)より約2.5倍高速です。*

### 実行手順 (Makefile)
```bash
make help      # ヘルプを表示
make test      # 各実装間の数学的等価性をテスト
make export    # 推論用step関数をONNXに出力
make benchmark # 性能ベンチマークを実行
```

### ファイル構成
- `edge_mamba/torch_model.py`: PyTorch版。学習およびONNXへのエクスポートに使用。
- `edge_mamba/numpy_model.py`: NumPy版。エッジでのスタンドアロン推論用（学習済みPyTorchモデルから重みをロードして使用）。
- `scripts/export_onnx_step.py`: ONNXエクスポート用スクリプト（**PyTorchモデル**から推論用step関数のみを抽出）。

### 推奨ワークフロー
1. **学習**: GPU環境にて `torch_model.py` を用いて学習を行います。
2. **ONNX変換**: `scripts/export_onnx_step.py` を使用して、**PyTorchの実装から**推論用ONNXファイルを生成します。（※NumPy実装からONNXへの変換はサポートしていません）
3. **デプロイ**: エッジデバイス上で ONNX Runtime を使用するか、学習済み重みを `numpy_model.py` に読み込ませて推論を実行します。
- `scripts/benchmark.py`: 性能比較用スクリプト。
- `tests/test_step_consistency.py`: 数学的整合性検証テスト。
