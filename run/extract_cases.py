#!/usr/bin/env python3
"""
Extract two qualitative SAW generation cases for paper figures:
  1. High-entropy creative text (ROCStories-style)
  2. Low-entropy rigid code generation

Usage (from MarkLLM root):
  PYTHONPATH=/home/lihe/MarkLLM python3 run/extract_cases.py
  PYTHONPATH=/home/lihe/MarkLLM python3 run/extract_cases.py --model Llama-3-8B-Instruct
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.tools.text_editor import TruncatePromptTextEditor, TruncateTaskTextEditor
from utils.transformers_config import TransformersConfig
from utils.utils import load_config_file
from watermark.auto_watermark import AutoWatermark

MODEL_PATHS = {
    "opt-1.3b": "/home/lihe/models/opt-1.3b",
    "Llama-3-8B-Instruct": "/home/lihe/models/Meta-Llama-3-8B-Instruct",
}

HIGH_ENTROPY_PROMPT = (
    "Your task is to continue the story beginning with "
    "The soccer game was tied 3 to 3 and there was a minute left to play. "
    "Incorporate the words goalkeeper penalty kick stadium seamlessly "
    "to develop a coherent and compelling narrative."
)

LOW_ENTROPY_PROMPT = (
    "Write a highly optimized Python function to reverse a linked list. "
    "Do not write any explanations, output code block only."
)


def load_model_and_tokenizer(model_name: str, device: str):
    path = MODEL_PATHS[model_name]
    if model_name == "opt-1.3b":
        model = AutoModelForCausalLM.from_pretrained(path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(path, legacy=False)
    elif model_name == "Llama-3-8B-Instruct":
        model = AutoModelForCausalLM.from_pretrained(path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(path, legacy=False)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    vocab_size = model.get_output_embeddings().weight.shape[0]
    return model, tokenizer, vocab_size


def format_prompt(tokenizer, prompt: str, model_name: str) -> str:
    if "Instruct" in model_name and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def build_watermark(model, tokenizer, vocab_size, device, max_new_tokens, temperature):
    config_dict = load_config_file("config/SAW.json")
    transformers_config = TransformersConfig(
        model=model,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        device=device,
        max_new_tokens=max_new_tokens,
        min_length=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        no_repeat_ngram_size=4,
    )
    return AutoWatermark.load("SAW", algorithm_config=config_dict, transformers_config=transformers_config)


def attach_model_name(watermark, model_name: str) -> None:
    watermark.config._model_name = model_name


def strip_prompt(full_text: str, prompt: str, mode: str) -> str:
    if mode == "story":
        return TruncatePromptTextEditor().edit(full_text, prompt)
    return TruncateTaskTextEditor().edit(full_text, prompt)


def extract_code_block(text: str) -> str:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def check_python_syntax(code: str) -> dict:
    result = {"syntax_valid": True, "syntax_error": None}
    try:
        ast.parse(code)
    except SyntaxError as exc:
        result["syntax_valid"] = False
        result["syntax_error"] = f"{exc.msg} (line {exc.lineno})"
    return result


def generate_watermarked(watermark, formatted_prompt: str) -> tuple[str, str]:
    """Return (full decoded text, generation-only continuation)."""
    tokenizer = watermark.config.generation_tokenizer
    encoded_prompt = tokenizer(
        formatted_prompt, return_tensors="pt", add_special_tokens=True
    ).to(watermark.config.device)

    from functools import partial
    from transformers import LogitsProcessorList

    generate_with_watermark = partial(
        watermark.config.generation_model.generate,
        logits_processor=LogitsProcessorList([watermark.logits_processor]),
        **watermark.config.gen_kwargs,
    )
    encoded_full = generate_with_watermark(
        **encoded_prompt,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = encoded_prompt["input_ids"].shape[1]
    new_ids = encoded_full[0, prompt_len:]
    full_text = tokenizer.batch_decode(encoded_full, skip_special_tokens=True)[0]
    generation_only = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return full_text, generation_only


def run_trial(watermark, prompt: str, truncate_mode: str, check_code: bool = False) -> dict:
    tokenizer = watermark.config.generation_tokenizer
    model_name = getattr(watermark.config, "_model_name", "")
    formatted = format_prompt(tokenizer, prompt, model_name)
    full_text, generation_only = generate_watermarked(watermark, formatted)
    if not generation_only:
        generation_only = strip_prompt(full_text, formatted, truncate_mode)

    detect_result = watermark.detect_watermark(generation_only)
    score = float(detect_result["score"])
    threshold = watermark.config.mean + watermark.config.std

    record = {
        "prompt": prompt,
        "formatted_prompt": formatted,
        "watermarked_text_full": full_text,
        "generation_only": generation_only,
        "empirical_score": score,
        "detection_threshold": threshold,
        "is_watermarked": bool(detect_result["is_watermarked"]),
    }

    if check_code:
        code = extract_code_block(generation_only)
        syntax = check_python_syntax(code)
        record["extracted_code"] = code
        record.update(syntax)
        trailing_after_fence = bool(
            re.search(r"```[\s\S]*?```\s*\S", generation_only)
        )
        record["looks_awkward"] = (
            not syntax["syntax_valid"]
            or "def " not in code
            or generation_only.count("```") > 2
            or trailing_after_fence
            or len(generation_only.strip()) < 20
            or code.count("def reverse") + code.count("def reverseList") > 1
        )

    return record


def print_latex_block(case_name: str, record: dict) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {case_name}")
    print(f"{'=' * 72}")
    print(f"Empirical score: {record['empirical_score']:.4f}")
    print(f"Threshold (mean+std): {record['detection_threshold']:.4f}")
    print(f"Detected as watermarked: {record['is_watermarked']}")
    print("\n--- Prompt ---")
    print(record["prompt"])
    print("\n--- Generated text (continuation only) ---")
    print(record["generation_only"])
    if "extracted_code" in record:
        print("\n--- Extracted code ---")
        print(record["extracted_code"])
        print(f"\nSyntax valid: {record['syntax_valid']}")
        if record.get("syntax_error"):
            print(f"Syntax error: {record['syntax_error']}")
        print(f"Looks awkward / structurally weak: {record['looks_awkward']}")


def main():
    parser = argparse.ArgumentParser(description="Extract SAW qualitative paper cases")
    parser.add_argument("--model", type=str, default="opt-1.3b", choices=list(MODEL_PATHS))
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(ROOT / "outputs" / "case_studies"),
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    model, tokenizer, vocab_size = load_model_and_tokenizer(args.model, device)
    watermark = build_watermark(
        model, tokenizer, vocab_size, device, args.max_new_tokens, args.temperature
    )
    attach_model_name(watermark, args.model)

    high = run_trial(watermark, HIGH_ENTROPY_PROMPT, truncate_mode="story")
    low = run_trial(
        watermark, LOW_ENTROPY_PROMPT, truncate_mode="code", check_code=True
    )

    payload = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "algorithm": "SAW",
            "model": args.model,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "device": device,
            "saw_config": load_config_file("config/SAW.json"),
        },
        "high_entropy_success_case": high,
        "low_entropy_failure_case": low,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model.replace('/', '_')}_tokens{args.max_new_tokens}"
    json_path = out_dir / f"cases_{tag}.json"
    txt_path = out_dir / f"cases_{tag}.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved TXT:  {txt_path}")

    print_latex_block("Trial 1 — High-Entropy (Creative Story)", high)
    print_latex_block("Trial 2 — Low-Entropy (Code Generation)", low)


if __name__ == "__main__":
    main()
