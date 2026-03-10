# Mamba Edge Implementation / Mamba エッジ実装

[English](#english) | [日本語](#japanese)

<a name="english"></a>
## English Description

This project provides a lightweight, **NumPy-only implementation** of the Mamba architecture tailored for edge devices. Unlike other implementations, it supports **both inference and training (backpropagation)** using only NumPy, making it ideal for environments where PyTorch cannot be installed.

### Key Features
- **Lightweight Inference & Training**: Runs on pure NumPy. No PyTorch dependency required for both forward and backward passes.
- **On-device Learning**: Fine-tune or train models directly on edge devices (CPU-only, no-Torch environments).
- **Training Compatibility**: Supports a hybrid workflow—train with PyTorch in development and fine-tune with NumPy on the edge.
- **Sequential Scan**: Optimized for CPU-bound edge devices using a sequential loop for both forward and backward selective scans.
- **Built-in Optimizer**: Includes a pure NumPy implementation of the Adam optimizer.

### File Structure
- `edge_mamba/numpy_model.py`: The NumPy-based Mamba model (Inference + **Backward/Training**).
- `edge_mamba/torch_model.py`: PyTorch implementation for high-speed training on GPUs.
- `scripts/train_numpy.py`: Sample script to train the model using **only NumPy**.
- `scripts/train.py`: Training script using PyTorch.
- `scripts/export_to_numpy.py`: Tool to convert PyTorch weights (`.pth`) to NumPy format (`.npz`).
- `tests/test_backward_consistency.py`: Test to ensure gradients match exactly between PyTorch and NumPy.

### Installation

**For Edge Devices (NumPy only):**
```bash
pip install .
```

**For Development (Training/Testing with PyTorch):**
```bash
pip install .[dev]
```

### Workflow

#### Option A: NumPy-only Training (On-device Learning)
Train or fine-tune directly on a device without PyTorch:
```bash
python scripts/train_numpy.py
```

#### Option B: Hybrid Workflow (Train with Torch, Run with NumPy)
1. **Train** with PyTorch: `python scripts/train.py`
2. **Export** weights: `python scripts/export_to_numpy.py`
3. **Inference** with NumPy: `python scripts/run_edge.py`

### Testing
To verify that the NumPy implementation (both forward and backward) matches the PyTorch implementation exactly:
```bash
python tests/test_consistency.py
python tests/test_backward_consistency.py
```

---

<a name="japanese"></a>
## 日本語説明

このプロジェクトは、エッジデバイス向けに最適化された**NumPyのみで動作するMamba実装**を提供します。他の実装とは異なり、推論だけでなく**学習（誤差逆伝播）もNumPyのみでサポート**しているため、PyTorchをインストールできない制限された環境での現地学習に最適です。

### 特徴
- **軽量な推論と学習**: NumPyだけで動作します。順伝播（Forward）と逆伝播（Backward）の両方でPyTorchへの依存はありません。
- **現地学習 (On-device Learning)**: エッジデバイス（CPUのみ、Torchなし環境）で直接モデルの微調整や学習が可能です。
- **ハイブリッドな開発**: 開発環境のGPUでPyTorchを使って高速に学習し、エッジ環境でNumPyを使って動作・微調整させるフローをサポートします。
- **逐次スキャン**: エッジデバイス（CPU）での効率を考慮し、SSMのスキャン処理を順方向・逆方向ともに逐次ループで実装しています。
- **組み込みオプティマイザ**: AdamオプティマイザをNumPyのみで実装して同梱しています。

### ファイル構成
- `edge_mamba/numpy_model.py`: NumPyベースのMambaモデル（推論 + **学習/逆伝播**）。
- `edge_mamba/torch_model.py`: GPU環境での高速学習用PyTorch実装。
- `scripts/train_numpy.py`: **NumPyのみ**で学習を実行するサンプルスクリプト。
- `scripts/train.py`: PyTorchを使用した学習スクリプト。
- `scripts/export_to_numpy.py`: PyTorchの重み（`.pth`）をNumPy形式（`.npz`）に変換するツール。
- `tests/test_backward_consistency.py`: PyTorch版とNumPy版で勾配計算が一致することを確認するテスト。

### インストール

**エッジデバイス向け（NumPyのみ）:**
```bash
pip install .
```

**開発環境向け（学習・テスト）:**
```bash
pip install .[dev]
```

### 実行フロー

#### オプションA：NumPyのみで学習（現地学習）
PyTorchのない環境で直接学習・微調整を行います。
```bash
python scripts/train_numpy.py
```

#### オプションB：ハイブリッド（Torchで学習、NumPyで実行）
1. **学習**: `python scripts/train.py` (PyTorch)
2. **変換**: `python scripts/export_to_numpy.py`
3. **推論**: `python scripts/run_edge.py` (NumPy)

### テスト
NumPy版の計算結果（順伝播・逆伝播）がPyTorch版と完全に一致するか確認するには、以下を実行します。
```bash
python tests/test_consistency.py
python tests/test_backward_consistency.py
```
