# SAW: Scaling Watermarking for Large Language Models

This repository contains the official implementation and experimental code for the paper: **"[Title of Your Paper]"**.

This project implements **SAW** (Scaling Watermarking), a novel watermarking framework, and compares it against state-of-the-art baselines including KGW, SWEET, SIR, and SynthID. The codebase is built upon the [MarkLLM] framework.

## 🌟 Key Features

* **Multi-Algorithm Support**: Implementations of SAW, KGW, SWEET, SIR, SynthID, EWD, DIP, and more.
* **Comprehensive Evaluation**: Pipelines for **Detectability** (TPR/FPR/AUC), **Text Quality** (BLEU, BERTScore, PPL), and **Robustness**.
* **Diverse Datasets**: Support for C4, WMT16 (De-En), HumanEval, ROCStories, etc.
* **Attack Simulations**: Built-in text editing attacks including Word Deletion, Synonym Substitution, Paraphrasing (Dipper/GPT), and Misspellings.

## 📂 Directory Structure

```text
.
├── assess/                 # Scripts for assessing detectability, quality, and robustness
├── config/                 # Configuration files (JSON) for each watermark algorithm
├── evaluation/             # Core evaluation logic
│   ├── pipelines/          # Detection and quality analysis pipelines
│   └── tools/              # Text editors (attacks), oracle, and quality analyzers
├── font/                   # Fonts used for visualization
├── lines/                  # Scripts for generating result charts
├── models/                 # (Optional) Directory for local models
├── output/                 # Directory where experiment logs and results are saved
├── utils/                  # Utility functions and Transformer configurations
├── visualize/              # Visualization tools
├── watermark/              # Source code for watermark algorithms
│   ├── saw/                # SAW algorithm implementation
│   ├── sir/                # SIR algorithm implementation
│   ├── synthid/            # SynthID algorithm implementation
│   └── ...
├── launch_saw.py           # Launcher script for batch experiments
├── pipeline_saw.py         # Main execution pipeline
└── requirements.txt        # Python dependencies

```

## 🛠️ Installation

1. **Clone the repository** (or download the source code):
```bash
git clone https://github.com/yourusername/SAW.git
cd SAW

```


2. **Create a virtual environment** (Recommended):
```bash
conda create -n markllm python=3.10
conda activate markllm

```


3. **Install dependencies**:
```bash
pip install -r requirements.txt

```


*Note: Ensure you have PyTorch installed with CUDA support if you intend to use GPU acceleration.*

## 🚀 Quick Start

The main entry point for running experiments is `pipeline_saw.py`.

### Basic Usage

To run the **SAW** watermark on the **WMT16 (German-to-English)** dataset using the **NLLB-200** model:

```bash
python3 pipeline_saw.py \
    --algorithm SAW \
    --dataset wmt16_de_en \
    --model nllb-200-distilled-600M \
    --max_new_tokens 200 \
    --min_length 200 \
    --data_lines 100

```

### Parameters Explanation

* `--algorithm`: The watermark algorithm to use (e.g., `SAW`, `KGW`, `SWEET`, `SIR`, `SynthID`).
* `--dataset`: Target dataset (e.g., `wmt16_de_en`, `c4`, `rocstories`).
* `--model`: The model checkpoint (e.g., `nllb-200-distilled-600M`, `Llama-3-8B-Instruct`).
* `--data_lines`: Number of samples to evaluate (default: 100).
* `--temperature_inner`: Sampling temperature during generation.

### Configuration

Each algorithm has its specific hyperparameters defined in `config/<AlgorithmName>.json`. For example, `config/SAW.json` allows you to tune:

* `beta`, `std`: Parameters specific to the SAW method.
* `z_threshold`: Threshold for detection.
* `window_scheme`: Watermarking window strategy.

You can modify these JSON files or pass some arguments directly via command line (see `pipeline_saw.py` for supported overrides).

## 📊 Supported Algorithms & Baselines

| Algorithm | Config File | Description |
| --- | --- | --- |
| **SAW** (Ours) | `config/SAW.json` | Scaling Watermarking |
| **KGW** | `config/KGW.json` | Kirchenbauer et al. (2023) |
| **SWEET** | `config/SWEET.json` | Entropy-based watermarking |
| **SIR** | `config/SIR.json` | Robust watermarking with context scaling |
| **SynthID** | `config/SynthID.json` | DeepMind's non-distortionary watermarking |
| **EWD** | `config/EWD.json` | Exponential Weighting Distribution |

## 🛡️ Robustness Evaluation

The pipeline automatically evaluates the watermark against various attacks:

* **Word-D**: Word Deletion (10%, 30%, etc.)
* **Word-S**: Synonym Substitution
* **Paraphrasing**: Using Dipper or GPT models
* **Typos / Misspelling**

Results including TPR (True Positive Rate), F1 Score, and Z-scores under attack are logged in the `output/` directory.

## ⚖️ License

This project is licensed under the Apache 2.0 License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## 🙏 Acknowledgements

This codebase is built upon the open-source framework [MarkLLM]. We thank the original authors for their contribution to the community.

---

**Note for Reviewers:** This repository contains the source code for the algorithms and experiments described in the paper. Sensitive information has been anonymized for the double-blind review process.