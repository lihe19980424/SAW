#!/usr/bin/env python3
"""
Empirical check: are Top-K (K=100) logits positive during OPT-1.3B generation?
Supports the rebuttal claim that multiplicative SAW scaling avoids negative-logit bias
because Top-K candidates are almost always positive.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/lihe/models/opt-1.3b"
TOP_K = 100
TOKENS_PER_PROMPT = 20

PROMPTS = [
    "The rapid development of",
    "Once upon a time",
    "In recent years,",
    "Scientists have discovered that",
    "The main advantage of machine learning is",
    "According to the latest report,",
    "Climate change has led to",
    "The history of the internet began",
    "To solve this problem, we need",
    "Artificial intelligence can help",
]


@torch.no_grad()
def analyze_prompt(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    device: torch.device,
    tokens_to_generate: int,
) -> list[dict]:
    """Greedy-decode `tokens_to_generate` steps; record Top-K min logit each step."""
    records: list[dict] = []
    ids = input_ids.clone()

    for step in range(tokens_to_generate):
        outputs = model(ids, return_dict=True)
        logits = outputs.logits[0, -1, :].float()

        topk_vals, topk_idx = torch.topk(logits, TOP_K)
        min_logit = topk_vals.min().item()
        all_positive = bool((topk_vals > 0).all().item())
        num_positive = int((topk_vals > 0).sum().item())

        records.append(
            {
                "step": step + 1,
                "min_topk_logit": min_logit,
                "max_topk_logit": topk_vals.max().item(),
                "mean_topk_logit": topk_vals.mean().item(),
                "all_topk_positive": all_positive,
                "num_positive_in_topk": num_positive,
                "top1_token_id": topk_idx[0].item(),
            }
        )

        next_id = topk_idx[0].view(1, 1)  # greedy: highest logit
        ids = torch.cat([ids, next_id], dim=1)

    return records


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {MODEL_PATH}")
    print(f"Prompts: {len(PROMPTS)}, tokens per prompt: {TOKENS_PER_PROMPT}, Top-K: {TOP_K}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    all_records: list[dict] = []

    for i, prompt in enumerate(PROMPTS, start=1):
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(device)
        recs = analyze_prompt(model, input_ids, device, TOKENS_PER_PROMPT)
        all_records.extend(recs)

        mins = [r["min_topk_logit"] for r in recs]
        pos_steps = sum(1 for r in recs if r["all_topk_positive"])
        print(
            f"[{i:2d}/{len(PROMPTS)}] \"{prompt[:40]}{'...' if len(prompt) > 40 else ''}\" "
            f"| min Top-{TOP_K} logit range: [{min(mins):.4f}, {max(mins):.4f}] "
            f"| steps all-positive: {pos_steps}/{len(recs)}"
        )

    total_steps = len(all_records)
    global_min = min(r["min_topk_logit"] for r in all_records)
    global_max_of_mins = max(r["min_topk_logit"] for r in all_records)
    steps_all_positive = sum(1 for r in all_records if r["all_topk_positive"])
    total_topk_entries = total_steps * TOP_K
    total_positive_entries = sum(r["num_positive_in_topk"] for r in all_records)
    fraction_positive_entries = total_positive_entries / total_topk_entries
    fraction_steps_all_positive = steps_all_positive / total_steps

    print()
    print("=" * 60)
    print("AGGREGATE STATISTICS")
    print("=" * 60)
    print(f"Total generation steps analyzed : {total_steps} "
          f"({len(PROMPTS)} prompts × {TOKENS_PER_PROMPT} tokens)")
    print(f"Global minimum of Top-{TOP_K} min-logit : {global_min:.6f}")
    print(f"Largest per-step min among Top-{TOP_K} : {global_max_of_mins:.6f}")
    print(f"Steps where ALL Top-{TOP_K} logits > 0 : {steps_all_positive} / {total_steps} "
          f"({100.0 * fraction_steps_all_positive:.2f}%)")
    print(f"Top-{TOP_K} entries with logit > 0      : {total_positive_entries} / {total_topk_entries} "
          f"({100.0 * fraction_positive_entries:.4f}%)")

    if steps_all_positive < total_steps:
        bad = [r for r in all_records if not r["all_topk_positive"]]
        worst = min(bad, key=lambda r: r["min_topk_logit"])
        print()
        print("Note: some steps contain non-positive Top-K logits.")
        print(f"  Worst step: min Top-{TOP_K} logit = {worst['min_topk_logit']:.6f}, "
              f"{worst['num_positive_in_topk']}/{TOP_K} positive.")
    else:
        print()
        print("All analyzed steps have strictly positive Top-100 logits.")


if __name__ == "__main__":
    main()
