#!/usr/bin/env python3
"""Run formal Agent-SAW experiments and export paper-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from agent_watermark.action_space import ActionSpace
from agent_watermark.agent_saw import AgentSAW, AgentSAWConfig
from agent_watermark.attacks import drop_random_steps, paraphrase_observations, rename_actions
from agent_watermark.detector import TrajectoryDetector, TrajectoryStep
from agent_watermark.metrics import auc_score, best_f1_threshold, tool_accuracy, tpr_at_fpr
from agent_watermark.tasks import (
    TravelPlanningScenario,
    ToolSelectionScenario,
    generate_tool_selection_scenarios,
    generate_travel_planning_scenarios,
)
from agent_watermark.toy_agent import ToyAgent


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_agent_saw_config(cfg: dict) -> AgentSAWConfig:
    return AgentSAWConfig(
        hash_key=cfg.get("hash_key", 15485863),
        private_key_seed=cfg.get("private_key_seed", 42),
        beta=cfg.get("beta", 0.9),
        std=cfg.get("std", 0.08),
        mean=cfg.get("mean", 1.0),
        noise=cfg.get("noise", "uniform"),
        mean_threshold=cfg.get("mean_threshold", cfg.get("z_threshold", 1.03)),
        left_rand=cfg.get("left_rand", 0.0),
        selection_mode=cfg.get("selection_mode", "sample"),
        temperature=cfg.get("temperature", 0.7),
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_scores(
    pos: list[float],
    neg: list[float],
    attacked_pos: dict[str, list[float]],
) -> dict:
    auc = auc_score(pos, neg)
    tpr1, thr1 = tpr_at_fpr(pos, neg, target_fpr=0.01)
    best_f1, best_thr, best_acc = best_f1_threshold(pos, neg)
    out = {
        "n_pos": len(pos),
        "n_neg": len(neg),
        "mean_score_w": sum(pos) / len(pos) if pos else 0.0,
        "mean_score_nw": sum(neg) / len(neg) if neg else 0.0,
        "auc": auc,
        "tpr_at_1fpr": tpr1,
        "thr_at_1fpr": thr1,
        "best_f1": best_f1,
        "best_thr": best_thr,
        "best_acc": best_acc,
    }
    for name, scores in attacked_pos.items():
        f1, _, acc = best_f1_threshold(scores, neg)
        out[f"f1_{name}"] = f1
        out[f"acc_{name}"] = acc
        out[f"mean_score_{name}"] = sum(scores) / len(scores) if scores else 0.0
    return out


def run_tool_selection(
    agent: ToyAgent,
    detector: TrajectoryDetector,
    scenarios: list[ToolSelectionScenario],
    rng: random.Random,
) -> tuple[list[dict], list[dict], dict]:
    trajectory_rows: list[dict] = []
    per_scenario: list[dict] = []
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    attacked = {"drop": [], "rename": [], "paraphrase": []}
    acc_w = 0.0
    acc_nw = 0.0

    rename_map = {
        "check_weather": "weather",
        "book_flight": "flight_booking",
        "search_web": "web_search",
        "calculator": "calc",
        "ask_user": "clarify",
    }

    for scenario in scenarios:
        golds = scenario.gold_actions or [scenario.gold_action]
        step_defs = scenario.steps or [
            {
                "observation": scenario.observation,
                "candidate_actions": scenario.candidate_actions,
                "original_scores": scenario.original_scores,
                "gold_action": scenario.gold_action,
            }
        ]

        # Non-watermarked session
        agent.reset()
        nw_records = agent.act_tool_session(scenario, watermark_on=False)
        nw_steps = [
            TrajectoryStep(observation=step_defs[i]["observation"], action=nw_records[i].selected_action)
            for i in range(len(nw_records))
        ]
        nw_det = detector.detect(nw_steps)
        neg_scores.append(nw_det.mean_score)
        nw_acc = tool_accuracy([r.selected_action for r in nw_records], golds)
        acc_nw += nw_acc
        for idx, record in enumerate(nw_records):
            trajectory_rows.append(
                {
                    "task": "tool_selection",
                    "scenario_id": scenario.scenario_id,
                    "step_idx": idx,
                    "watermark_on": False,
                    "gold_action": golds[idx] if idx < len(golds) else scenario.gold_action,
                    "trajectory_mean_score": nw_det.mean_score,
                    **asdict(record),
                }
            )

        # Watermarked session
        agent.reset()
        w_records = agent.act_tool_session(scenario, watermark_on=True)
        w_steps = [
            TrajectoryStep(observation=step_defs[i]["observation"], action=w_records[i].selected_action)
            for i in range(len(w_records))
        ]
        w_det = detector.detect(w_steps)
        pos_scores.append(w_det.mean_score)
        w_acc = tool_accuracy([r.selected_action for r in w_records], golds)
        acc_w += w_acc
        for idx, record in enumerate(w_records):
            trajectory_rows.append(
                {
                    "task": "tool_selection",
                    "scenario_id": scenario.scenario_id,
                    "step_idx": idx,
                    "watermark_on": True,
                    "gold_action": golds[idx] if idx < len(golds) else scenario.gold_action,
                    "trajectory_mean_score": w_det.mean_score,
                    **asdict(record),
                }
            )

        drop_det = detector.detect(drop_random_steps(w_steps, drop_ratio=0.3, rng=rng))
        rename_det = detector.detect(rename_actions(w_steps, rename_map))
        para_det = detector.detect(paraphrase_observations(w_steps))
        attacked["drop"].append(drop_det.mean_score)
        attacked["rename"].append(rename_det.mean_score)
        attacked["paraphrase"].append(para_det.mean_score)

        per_scenario.append(
            {
                "task": "tool_selection",
                "scenario_id": scenario.scenario_id,
                "n_steps": len(w_steps),
                "success_nw": nw_acc,
                "success_w": w_acc,
                "score_nw": nw_det.mean_score,
                "score_w": w_det.mean_score,
                "score_drop": drop_det.mean_score,
                "score_rename": rename_det.mean_score,
                "score_paraphrase": para_det.mean_score,
            }
        )

    n = len(scenarios)
    summary = summarize_scores(pos_scores, neg_scores, attacked)
    summary.update(
        {
            "task": "tool_selection",
            "n_scenarios": n,
            "tool_acc_nw": acc_nw / n if n else 0.0,
            "tool_acc_w": acc_w / n if n else 0.0,
        }
    )
    return trajectory_rows, per_scenario, summary


def run_travel_planning(
    agent: ToyAgent,
    detector: TrajectoryDetector,
    scenarios: list[TravelPlanningScenario],
    rng: random.Random,
) -> tuple[list[dict], list[dict], dict]:
    trajectory_rows: list[dict] = []
    per_scenario: list[dict] = []
    pos_scores: list[float] = []
    neg_scores: list[float] = []
    attacked = {"drop": [], "rename": [], "paraphrase": []}
    acc_w = 0.0
    acc_nw = 0.0

    rename_map = {
        "search_flight": "flight_search",
        "search_hotel": "hotel_search",
        "check_weather": "weather",
        "search_attractions": "poi_search",
        "summarize_plan": "final_plan",
        "ask_user": "clarify",
    }

    for scenario in scenarios:
        # NW trajectory
        agent.reset()
        nw_records = agent.act_travel_plan(scenario, watermark_on=False)
        nw_steps = [
            TrajectoryStep(observation=step["observation"], action=nw_records[i].selected_action)
            for i, step in enumerate(scenario.steps)
        ]
        nw_det = detector.detect(nw_steps)
        neg_scores.append(nw_det.mean_score)
        nw_acc = tool_accuracy([r.selected_action for r in nw_records], scenario.gold_actions)
        acc_nw += nw_acc
        for idx, record in enumerate(nw_records):
            trajectory_rows.append(
                {
                    "task": "travel_planning",
                    "scenario_id": scenario.scenario_id,
                    "step_idx": idx,
                    "watermark_on": False,
                    "gold_action": scenario.gold_actions[idx],
                    "trajectory_mean_score": nw_det.mean_score,
                    **asdict(record),
                }
            )

        # W trajectory
        agent.reset()
        w_records = agent.act_travel_plan(scenario, watermark_on=True)
        w_steps = [
            TrajectoryStep(observation=step["observation"], action=w_records[i].selected_action)
            for i, step in enumerate(scenario.steps)
        ]
        w_det = detector.detect(w_steps)
        pos_scores.append(w_det.mean_score)
        w_acc = tool_accuracy([r.selected_action for r in w_records], scenario.gold_actions)
        acc_w += w_acc
        for idx, record in enumerate(w_records):
            trajectory_rows.append(
                {
                    "task": "travel_planning",
                    "scenario_id": scenario.scenario_id,
                    "step_idx": idx,
                    "watermark_on": True,
                    "gold_action": scenario.gold_actions[idx],
                    "trajectory_mean_score": w_det.mean_score,
                    **asdict(record),
                }
            )

        drop_det = detector.detect(drop_random_steps(w_steps, drop_ratio=0.3, rng=rng))
        rename_det = detector.detect(rename_actions(w_steps, rename_map))
        para_det = detector.detect(paraphrase_observations(w_steps))
        attacked["drop"].append(drop_det.mean_score)
        attacked["rename"].append(rename_det.mean_score)
        attacked["paraphrase"].append(para_det.mean_score)

        per_scenario.append(
            {
                "task": "travel_planning",
                "scenario_id": scenario.scenario_id,
                "n_steps": len(scenario.steps),
                "success_nw": nw_acc,
                "success_w": w_acc,
                "score_nw": nw_det.mean_score,
                "score_w": w_det.mean_score,
                "score_drop": drop_det.mean_score,
                "score_rename": rename_det.mean_score,
                "score_paraphrase": para_det.mean_score,
            }
        )

    n = len(scenarios)
    summary = summarize_scores(pos_scores, neg_scores, attacked)
    summary.update(
        {
            "task": "travel_planning",
            "n_scenarios": n,
            "tool_acc_nw": acc_nw / n if n else 0.0,
            "tool_acc_w": acc_w / n if n else 0.0,
        }
    )
    return trajectory_rows, per_scenario, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run formal Agent-SAW experiments")
    parser.add_argument("--config", type=str, default="configs/agent_saw.yaml")
    parser.add_argument("--n_tool", type=int, default=None)
    parser.add_argument("--n_travel", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    rng = random.Random(seed)
    output_dir = Path(cfg.get("output_dir", "outputs/agent_saw"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    n_tool = args.n_tool if args.n_tool is not None else int(cfg.get("n_tool", 200))
    n_travel = args.n_travel if args.n_travel is not None else int(cfg.get("n_travel", 200))

    saw_config = build_agent_saw_config(cfg)
    tool_space = ActionSpace.tool_selection()
    travel_space = ActionSpace.travel_planning()
    tool_agent = ToyAgent(tool_space, AgentSAW(tool_space, saw_config))
    travel_agent = ToyAgent(travel_space, AgentSAW(travel_space, saw_config))
    tool_detector = TrajectoryDetector(tool_space, saw_config)
    travel_detector = TrajectoryDetector(travel_space, saw_config)

    all_traj: list[dict] = []
    all_per: list[dict] = []
    summaries: list[dict] = []

    if "tool_selection" in cfg.get("tasks", []):
        scenarios = generate_tool_selection_scenarios(n_tool, seed=seed)
        traj, per, summary = run_tool_selection(tool_agent, tool_detector, scenarios, rng)
        all_traj.extend(traj)
        all_per.extend(per)
        summaries.append(summary)
        print(
            f"[tool_selection] n={summary['n_scenarios']} "
            f"acc_nw={summary['tool_acc_nw']:.3f} acc_w={summary['tool_acc_w']:.3f} "
            f"AUC={summary['auc']:.3f} TPR@1%FPR={summary['tpr_at_1fpr']:.3f} "
            f"mean_w={summary['mean_score_w']:.4f} mean_nw={summary['mean_score_nw']:.4f}"
        )

    if "travel_planning" in cfg.get("tasks", []):
        scenarios = generate_travel_planning_scenarios(n_travel, seed=seed + 1)
        traj, per, summary = run_travel_planning(travel_agent, travel_detector, scenarios, rng)
        all_traj.extend(traj)
        all_per.extend(per)
        summaries.append(summary)
        print(
            f"[travel_planning] n={summary['n_scenarios']} "
            f"acc_nw={summary['tool_acc_nw']:.3f} acc_w={summary['tool_acc_w']:.3f} "
            f"AUC={summary['auc']:.3f} TPR@1%FPR={summary['tpr_at_1fpr']:.3f} "
            f"mean_w={summary['mean_score_w']:.4f} mean_nw={summary['mean_score_nw']:.4f}"
        )

    traj_path = output_dir / "trajectories.jsonl"
    per_path = output_dir / "per_scenario.csv"
    summary_path = output_dir / "summary.csv"
    paper_path = output_dir / "paper_table.md"

    write_jsonl(traj_path, all_traj)
    write_csv(per_path, all_per)
    write_csv(summary_path, summaries)

    lines = [
        "# Agent-SAW Formal Results",
        "",
        f"- seed: {seed}",
        f"- noise: {saw_config.noise}, beta={saw_config.beta}, std={saw_config.std}",
        f"- selection: {saw_config.selection_mode}, temperature={saw_config.temperature}",
        f"- detection: mean reconstructed noise (threshold={saw_config.mean_threshold})",
        "",
        "| Task | N | Acc(NW) | Acc(W) | AUC | TPR@1%FPR | Mean(W) | Mean(NW) | F1(best) | F1(rename) | F1(paraphrase) | F1(drop) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s['task']} | {s['n_scenarios']} | {s['tool_acc_nw']:.3f} | {s['tool_acc_w']:.3f} | "
            f"{s['auc']:.3f} | {s['tpr_at_1fpr']:.3f} | {s['mean_score_w']:.4f} | {s['mean_score_nw']:.4f} | "
            f"{s['best_f1']:.3f} | {s.get('f1_rename', 0):.3f} | {s.get('f1_paraphrase', 0):.3f} | "
            f"{s.get('f1_drop', 0):.3f} |"
        )
    paper_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote trajectories: {traj_path} ({len(all_traj)} rows)")
    print(f"Wrote per-scenario: {per_path} ({len(all_per)} rows)")
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote paper table: {paper_path}")


if __name__ == "__main__":
    main()
