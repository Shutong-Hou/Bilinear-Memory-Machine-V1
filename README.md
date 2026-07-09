# A Bilinear Memory Machine (BMM) - V1

This repository contains the official PyTorch implementation of the paper: 
**"A Bilinear Memory Machine: A Novel Attention-Free Architecture for Efficient Long-Context Language Modeling"**.

## 🎯 Highlights
- **Attention-Free Architecture**: BMM eliminates dot-product attention, softmax, and convolution entirely, relying purely on bilinear state interactions and dynamic memory slots.
- **Strict O(1) Inference Memory**: Maintains a strictly constant 2.33 GB inference memory regardless of context length, remaining OOM-free at 100K tokens.
- **3x Throughput vs Mamba**: Achieves 3x higher throughput than State Space Models like Mamba by avoiding sequential hardware scan bottlenecks.
- **Honest Trade-offs**: We provide a deep theoretical and empirical analysis of the trade-offs between parallel state computation and raw perplexity.

## 🛠️ Environment Setup
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```
*Note: This implementation requires a single NVIDIA RTX 4090 (24GB) GPU for full reproduction. `mamba_ssm` requires CUDA toolkit for compilation.*

## 📂 Repository Structure
- `bmm_model.py`: Core BMM architecture and BPE data loader.
- `train_bmm_bpe.py`, `train_transformer_bpe.py`, `train_mamba_bpe.py`: Training scripts for main experiments.
- `evaluate_main.py`: Script to reproduce main PPL and Efficiency results (Tables 1-3).
- `eval_streaming.py`: Script for 100K extreme streaming generation test (Table 4).
- `run_ablations.py`: Script for architecture ablation studies (Table 6).

## 🚀 Quick Start

### 1. Train BMM (BPE, 50k steps, γ=2.0)
```bash
python train_bmm_bpe.py --gamma 2.0
```

### 2. Evaluate Inference Efficiency
```bash
python evaluate_main.py
```

## 📊 Pre-trained Checkpoints
Due to file size limits, pre-trained checkpoints (BMM, Transformer, Mamba at ~200M params) can be downloaded from:
[Download Link Here - e.g., Google Drive or HuggingFace]

## 📄 Citation
If you find this work useful, please cite:
```bibtex
@article{hou2026bmm,
  title={A Bilinear Memory Machine: A Novel Attention-Free Architecture for Efficient Long-Context Language Modeling},
  author={Hou, Shutong},
  year={2026}
}
```
