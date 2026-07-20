#!/usr/bin/env python3
"""Plot Agent-SAW metrics exported by run_agent_saw.py."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt


def load_metrics(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Agent-SAW metrics")
    parser.add_argument("--metrics", type=str, default="outputs/agent_saw/metrics.csv")
    parser.add_argument("--output", type=str, default="outputs/agent_saw/figures/detection_z_by_task.png")
    args = parser.parse_args()

    rows = load_metrics(Path(args.metrics))
    if not rows:
        raise SystemExit(f"No metrics found at {args.metrics}")

    tasks = sorted({row["task"] for row in rows})
    values = []
    for task in tasks:
        task_rows = [float(row["detection_z"]) for row in rows if row["task"] == task]
        values.append(sum(task_rows) / len(task_rows))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.bar(tasks, values)
    plt.ylabel("Mean detection z-score")
    plt.title("Agent-SAW detection by task")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    main()
