"""Metrics for formal Agent-SAW experiments."""

from __future__ import annotations

from dataclasses import dataclass


def tool_accuracy(selected: list[str], gold: list[str]) -> float:
    if not gold:
        return 0.0
    matches = sum(1 for pred, target in zip(selected, gold) if pred == target)
    return matches / len(gold)


def auc_score(pos_scores: list[float], neg_scores: list[float]) -> float:
    """Mann–Whitney AUC for watermarked (pos) vs non-watermarked (neg)."""
    if not pos_scores or not neg_scores:
        return 0.0
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (n_pos * n_neg)


def tpr_at_fpr(pos_scores: list[float], neg_scores: list[float], target_fpr: float = 0.01) -> tuple[float, float]:
    """
    Choose the highest threshold such that FPR <= target_fpr on negatives,
    then report TPR on positives. Returns (tpr, threshold).
    """
    if not pos_scores or not neg_scores:
        return 0.0, 0.0
    candidates = sorted(set(neg_scores + pos_scores), reverse=True)
    best_tpr = 0.0
    best_thr = candidates[0]
    n_neg = len(neg_scores)
    n_pos = len(pos_scores)
    for thr in candidates:
        fpr = sum(1 for s in neg_scores if s > thr) / n_neg
        if fpr <= target_fpr:
            tpr = sum(1 for s in pos_scores if s > thr) / n_pos
            if tpr >= best_tpr:
                best_tpr = tpr
                best_thr = thr
    return best_tpr, best_thr


def best_f1_threshold(pos_scores: list[float], neg_scores: list[float]) -> tuple[float, float, float]:
    """Scan thresholds and return (best_f1, threshold, accuracy)."""
    if not pos_scores or not neg_scores:
        return 0.0, 0.0, 0.0
    candidates = sorted(set(neg_scores + pos_scores))
    best = (0.0, candidates[0], 0.0)
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    for thr in candidates:
        tp = sum(1 for s in pos_scores if s > thr)
        fp = sum(1 for s in neg_scores if s > thr)
        fn = n_pos - tp
        tn = n_neg - fp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / n_pos if n_pos else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        acc = (tp + tn) / (n_pos + n_neg)
        if f1 > best[0]:
            best = (f1, thr, acc)
    return best


@dataclass
class TaskSummary:
    task: str
    n_scenarios: int
    tool_acc_nw: float
    tool_acc_w: float
    mean_score_nw: float
    mean_score_w: float
    auc: float
    tpr_at_1fpr: float
    best_f1: float
    best_acc: float
    f1_drop: float
    f1_rename: float
    f1_paraphrase: float
