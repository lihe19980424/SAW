#!/usr/bin/env python3
"""
Single-step visualization: Original vs SAW (multiplicative) vs SAW_ADD (additive)
logit reweighting on OPT-1.3B Top-K candidates.
Standalone script for rebuttal — does not modify SAW source code.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------- Config -------------------------
MODEL_PATH = "/home/lihe/models/opt-1.3b"
PROMPT = "The rapid development of artificial intelligence has"
TOP_K = 100
DISPLAY_K = 50
SEED = 15485863

# SAW multiplicative noise (near identity scaling)
SAW_MEAN = 1.0
SAW_STD = 0.05

# SAW_ADD additive noise (high-robustness regime for ablation contrast)
SAW_ADD_MEAN = 0.0
SAW_ADD_STD = 1.0

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "lines" / "SAW_Logit_Scaling_Vis.pdf"

# ------------------------- Style -------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.linewidth": 1.0,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

COLOR_ORIG = "#4D4D4D"
COLOR_SAW = "#2166AC"
COLOR_ADD = "#D6604D"


def get_last_token_logits(model, tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, return_dict=True)
    return outputs.logits[0, -1, :].float().cpu()


def topk_subset(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    values, indices = torch.topk(logits, k)
    return values, indices


def apply_saw_multiplicative(logits_topk: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    noise = torch.normal(
        mean=SAW_MEAN,
        std=SAW_STD,
        size=logits_topk.shape,
        generator=generator,
    )
    return logits_topk * noise


def apply_saw_add_additive(logits_topk: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    noise = torch.normal(
        mean=SAW_ADD_MEAN,
        std=SAW_ADD_STD,
        size=logits_topk.shape,
        generator=generator,
    )
    return logits_topk + noise


def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=0).numpy()


def plot_panel(ax, ranks, orig, saw, saw_add, ylabel: str, title: str) -> None:
    ax.plot(ranks, orig, color=COLOR_ORIG, linewidth=2.2, marker="o", markersize=5,
            label="Original Logits", zorder=3)
    ax.plot(ranks, saw, color=COLOR_SAW, linewidth=2.2, marker="s", markersize=5,
            label=r"SAW (Multiplicative, $\mu{=}1.0$, $\sigma{=}0.05$)", zorder=3)
    ax.plot(ranks, saw_add, color=COLOR_ADD, linewidth=2.2, marker="^", markersize=5,
            label=r"SAW$_{\mathrm{ADD}}$ (Additive, $\mu{=}0.0$, $\sigma{=}1.0$)", zorder=3)
    ax.set_xlabel("Token Rank (by Original Logit)", fontweight="medium")
    ax.set_ylabel(ylabel, fontweight="medium")
    ax.set_title(title, fontweight="medium", pad=10)
    ax.set_xlim(0.5, DISPLAY_K + 0.5)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="#CCCCCC")


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading model from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    print(f"Prompt: \"{PROMPT}\"")
    logits = get_last_token_logits(model, tokenizer, PROMPT, device)

    logits_topk, token_indices = topk_subset(logits, TOP_K)

    gen_saw = torch.Generator()
    gen_saw.manual_seed(SEED)
    gen_add = torch.Generator()
    gen_add.manual_seed(SEED + 1)

    logits_saw = apply_saw_multiplicative(logits_topk, gen_saw)
    logits_add = apply_saw_add_additive(logits_topk, gen_add)

    # Display top DISPLAY_K (already sorted descending by original logit)
    n = DISPLAY_K
    orig_np = logits_topk[:n].numpy()
    saw_np = logits_saw[:n].numpy()
    add_np = logits_add[:n].numpy()
    ranks = np.arange(1, n + 1)

    prob_orig = softmax_probs(logits_topk[:n])
    prob_saw = softmax_probs(logits_saw[:n])
    prob_add = softmax_probs(logits_add[:n])

    # Optional: decode top token strings for console
    top_tokens = [tokenizer.decode([idx.item()]) for idx in token_indices[:5]]
    print("Top-5 tokens:", top_tokens)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=150)

    plot_panel(
        axes[0],
        ranks,
        orig_np,
        saw_np,
        add_np,
        ylabel="Logit Value",
        title="(a) Top-50 Logits Before & After Scaling",
    )

    axes[1].plot(ranks, prob_orig, color=COLOR_ORIG, linewidth=2.2, marker="o", markersize=5,
                 label="Original (Softmax)", zorder=3)
    axes[1].plot(ranks, prob_saw, color=COLOR_SAW, linewidth=2.2, marker="s", markersize=5,
                 label="SAW Multiplicative", zorder=3)
    axes[1].plot(ranks, prob_add, color=COLOR_ADD, linewidth=2.2, marker="^", markersize=5,
                 label=r"SAW$_{\mathrm{ADD}}$ Additive", zorder=3)
    axes[1].set_xlabel("Token Rank (by Original Logit)", fontweight="medium")
    axes[1].set_ylabel("Probability", fontweight="medium")
    axes[1].set_title("(b) Softmax Distribution Over Top-50", fontweight="medium", pad=10)
    axes[1].set_xlim(0.5, DISPLAY_K + 0.5)
    axes[1].legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="#CCCCCC")

    fig.suptitle(
        "Visualization of Multiplicative vs Additive Logit Reweighting (Single Step, OPT-1.3B)",
        fontsize=16,
        fontweight="medium",
        y=1.02,
    )
    fig.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
