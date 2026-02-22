# Mamba Edge Implementation / Mamba エッジ実装

[English](#english) | [日本語](#japanese)

<a name="english"></a>
## English Description

This project provides a lightweight, **NumPy-only implementation** of the Mamba architecture tailored for edge devices. It allows you to train models using PyTorch in a development environment and deploy them to edge devices without requiring a heavy PyTorch installation.

### Key Features
- **Lightweight Inference**: Runs on pure NumPy. No PyTorch dependency required on the edge device.
- **Training Compatibility**: Train with PyTorch, export weights, and run on NumPy.
- **Sequential Scan**: Replaces parallel scan with a sequential loop for efficiency on CPU-bound edge devices.

### File Structure
- `mamba_numpy.py`: The NumPy-based Mamba model implementation (Inference).
- `run_edge.py`: Sample script to run inference using NumPy weights.
- `train.py`: Training script using PyTorch.
- `export_to_numpy.py`: Tool to convert PyTorch weights (`.pth`) to NumPy format (`.npz`).
- `test_mamba.py`: Test to ensure outputs match between PyTorch and NumPy implementations.

### Installation

**For Edge Devices (Inference only):**
```bash
pip install .
```
This installs only `numpy`.

**For Development (Training/Testing):**
```bash
pip install .[dev]
```
This installs `torch` and `numpy`.

### Workflow

#### 1. Train (Development Environment)
Train your model using PyTorch. This generates `mamba_model.pth`.
```bash
python scripts/train.py
```

#### 2. Export Weights (Development Environment)
Convert the trained PyTorch weights to a NumPy-compatible format (`mamba_weights.npz`).
```bash
python scripts/export_to_numpy.py
```

#### 3. Inference (Edge Device)
Transfer `mamba_numpy.py`, `run_edge.py`, and `mamba_weights.npz` to your edge device. Then run:
```bash
python scripts/run_edge.py
```
This script does **not** import torch.

### Testing
To verify that the NumPy implementation matches the PyTorch implementation exactly:
```bash
python tests/test_consistency.py
```

---

<a name="japanese"></a>
## 日本語説明

このプロジェクトは、エッジデバイス向けに最適化された**NumPyのみで動作するMamba実装**を提供します。開発環境ではPyTorchを使用して学習を行い、その学習済みモデルをPyTorchのインストールが不要なエッジデバイス上で動作させることができます。

### 特徴
- **軽量推論**: NumPyだけで動作します。エッジデバイスに巨大なPyTorchライブラリをインストールする必要はありません。
- **学習互換性**: PyTorchで学習し、重みをエクスポートしてNumPyで実行できます。
- **逐次スキャン**: エッジデバイス（CPU）での効率を考慮し、並列スキャン（Parallel Scan）を逐次ループ（Sequential Scan）に置き換えています。

### ファイル構成
- `mamba_numpy.py`: NumPyベースのMambaモデル実装（推論用）。
- `run_edge.py`: NumPy形式の重みを使って推論を行うサンプルスクリプト。
- `train.py`: PyTorchを使用した学習スクリプト。
- `export_to_numpy.py`: PyTorchの重み（`.pth`）をNumPy形式（`.npz`）に変換するツール。
- `test_mamba.py`: PyTorch版とNumPy版の出力が一致することを確認するテスト。

### インストール

**エッジデバイス向け（推論のみ）:**
```bash
pip install .
```
これにより `numpy` のみがインストールされます。

**開発環境向け（学習・テスト）:**
```bash
pip install .[dev]
```
これにより `torch` と `numpy` がインストールされます。

### 実行フロー

#### 1. 学習（開発環境）
PyTorchを使ってモデルを学習させます。`mamba_model.pth` が生成されます。
```bash
python scripts/train.py
```

#### 2. 重みの変換（開発環境）
学習済みのPyTorchの重みを、NumPyで読み込める形式（`mamba_weights.npz`）に変換します。
```bash
python scripts/export_to_numpy.py
```

#### 3. 推論（エッジデバイス）
`mamba_numpy.py`、`run_edge.py`、`mamba_weights.npz` の3ファイルをエッジデバイスに転送し、実行します。
```bash
python scripts/run_edge.py
```
このスクリプトは `torch` を一切インポートせずに動作します。

### テスト
NumPy版の実装がPyTorch版と完全に同じ結果を出すか確認するには、以下を実行します。
```bash
python tests/test_consistency.py
```
