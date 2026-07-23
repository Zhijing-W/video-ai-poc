"""Identity scoring, aggregate metrics, and paired statistics."""
from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np

from common import mevid_eval_common as common


def _templates(
    gallery_records: list[dict],
    gallery_vectors: list[np.ndarray | None],
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for row, vector in zip(gallery_records, gallery_vectors):
        if vector is not None:
            grouped[row["pid"]].append(vector)
    return {
        pid: common.l2norm(np.mean(np.stack(vectors), axis=0))
        for pid, vectors in grouped.items()
    }


def _score(vector: np.ndarray | None, templates: dict[str, np.ndarray], gt: str) -> dict:
    if vector is None or not templates:
        return {
            "pred": None,
            "score": None,
            "rank": None,
            "rank1_correct": False,
            "rank5_correct": False,
            "gt_template_available": gt in templates,
            "reciprocal_rank": 0.0,
            "gt_cosine": None,
            "max_other_cosine": None,
            "margin": None,
        }
    ranked = sorted(
        ((pid, float(template @ vector)) for pid, template in templates.items()),
        key=lambda item: (-item[1], item[0]),
    )
    rank = next(
        (index + 1 for index, (pid, _) in enumerate(ranked) if pid == gt),
        None,
    )
    scores = dict(ranked)
    gt_cosine = scores.get(gt)
    max_other = max(
        (score for pid, score in ranked if pid != gt),
        default=None,
    )
    return {
        "pred": ranked[0][0],
        "score": round(ranked[0][1], 6),
        "rank": rank,
        "rank1_correct": rank == 1,
        "rank5_correct": rank is not None and rank <= 5,
        "gt_template_available": gt in templates,
        "reciprocal_rank": round(1.0 / rank, 6) if rank else 0.0,
        "gt_cosine": round(gt_cosine, 6) if gt_cosine is not None else None,
        "max_other_cosine": (
            round(max_other, 6) if max_other is not None else None
        ),
        "margin": (
            round(gt_cosine - max_other, 6)
            if gt_cosine is not None and max_other is not None
            else None
        ),
    }


def summarize_scores(rows: list[dict], threshold: float) -> dict:
    count = len(rows)
    correct_accept = sum(
        row["rank1_correct"]
        and row["score"] is not None
        and row["score"] >= threshold
        for row in rows
    )
    wrong_accept = sum(
        not row["rank1_correct"]
        and row["pred"] is not None
        and row["score"] is not None
        and row["score"] >= threshold
        for row in rows
    )
    return {
        "queries": count,
        "vector_available": sum(row["pred"] is not None for row in rows),
        "gt_template_available": sum(
            row["gt_template_available"] for row in rows
        ),
        "rank1": sum(row["rank1_correct"] for row in rows),
        "rank1_rate": (
            round(sum(row["rank1_correct"] for row in rows) / count, 6)
            if count
            else None
        ),
        "rank5": sum(row["rank5_correct"] for row in rows),
        "rank5_rate": (
            round(sum(row["rank5_correct"] for row in rows) / count, 6)
            if count
            else None
        ),
        "mean_reciprocal_rank": (
            round(
                sum(float(row.get("reciprocal_rank") or 0.0) for row in rows)
                / count,
                6,
            )
            if count
            else None
        ),
        "fixed_threshold": threshold,
        "correct_accept": correct_accept,
        "correct_accept_rate": round(correct_accept / count, 6) if count else None,
        "wrong_accept_among_genuine_queries": wrong_accept,
        "wrong_accept_rate_among_genuine_queries": (
            round(wrong_accept / count, 6) if count else None
        ),
        "genuine_wrong_identity": wrong_accept,
        "genuine_wrong_identity_rate": (
            round(wrong_accept / count, 6) if count else None
        ),
        "note": "wrong accept among genuine Query samples; this is not true FMR",
    }


def _exact_paired_p(improved: int, degraded: int) -> float:
    discordant = improved + degraded
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(improved, degraded) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def pid_cluster_bootstrap_rate(
    rows: list[dict],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("pid", row.get("sample_id")))].append(
            bool(row["rank1_correct"])
        )
    clusters = list(grouped.values())
    rng = random.Random(seed)
    rates = []
    for _ in range(max(0, bootstrap_samples)):
        sampled = [
            clusters[rng.randrange(len(clusters))]
            for _ in range(len(clusters))
        ] if clusters else []
        values = [value for cluster in sampled for value in cluster]
        if values:
            rates.append(sum(values) / len(values))
    rates.sort()
    lower = rates[int(0.025 * (len(rates) - 1))] if rates else None
    upper = rates[int(0.975 * (len(rates) - 1))] if rates else None
    return {
        "rate": (
            round(sum(row["rank1_correct"] for row in rows) / len(rows), 6)
            if rows
            else None
        ),
        "pid_cluster_bootstrap_95ci": (
            [round(lower, 6), round(upper, 6)]
            if lower is not None and upper is not None
            else None
        ),
        "bootstrap_samples": max(0, bootstrap_samples),
        "pid_clusters": len(clusters),
        "inference_label": "exploratory",
    }


def paired_uncertainty(
    before: list[dict],
    after: list[dict],
    *,
    bootstrap_samples: int,
    seed: int,
    permutation_samples: int = 0,
) -> dict:
    if [row["sample_id"] for row in before] != [
        row["sample_id"] for row in after
    ]:
        raise ValueError("paired rows顺序不一致")
    deltas = [
        int(candidate["rank1_correct"]) - int(base["rank1_correct"])
        for base, candidate in zip(before, after)
    ]
    improved = sum(delta == 1 for delta in deltas)
    degraded = sum(delta == -1 for delta in deltas)
    correct_to_correct = sum(
        base["rank1_correct"] and candidate["rank1_correct"]
        for base, candidate in zip(before, after)
    )
    wrong_to_wrong = sum(
        not base["rank1_correct"] and not candidate["rank1_correct"]
        for base, candidate in zip(before, after)
    )
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, (base, delta) in enumerate(zip(before, deltas)):
        grouped[str(base.get("pid", base.get("sample_id", index)))].append(delta)
    clusters = list(grouped.values())
    rng = random.Random(seed)
    boot = []
    if clusters:
        for _ in range(max(0, bootstrap_samples)):
            sampled = [
                clusters[rng.randrange(len(clusters))]
                for _ in range(len(clusters))
            ]
            flat = [delta for cluster in sampled for delta in cluster]
            boot.append(sum(flat) / len(flat))
    boot.sort()
    lower = boot[int(0.025 * (len(boot) - 1))] if boot else None
    upper = boot[int(0.975 * (len(boot) - 1))] if boot else None
    observed = abs(sum(deltas))
    permutations = max(0, int(permutation_samples))
    extreme = 0
    for _ in range(permutations):
        permuted = sum(
            delta * sign
            for cluster in clusters
            for sign in [(-1 if rng.random() < 0.5 else 1)]
            for delta in cluster
        )
        extreme += abs(permuted) >= observed
    return {
        "samples": len(deltas),
        "rank1_improved": improved,
        "rank1_degraded": degraded,
        "correct_to_correct": correct_to_correct,
        "correct_to_wrong": degraded,
        "wrong_to_correct": improved,
        "wrong_to_wrong": wrong_to_wrong,
        "prediction_changed": sum(
            base["pred"] != candidate["pred"]
            for base, candidate in zip(before, after)
        ),
        "rank1_net_delta": round(sum(deltas) / len(deltas), 6) if deltas else None,
        "paired_bootstrap_95ci": (
            [round(lower, 6), round(upper, 6)]
            if lower is not None and upper is not None
            else None
        ),
        "bootstrap_samples": max(0, bootstrap_samples),
        "bootstrap_unit": "pid_cluster",
        "pid_clusters": len(clusters),
        "exact_paired_two_sided_p": round(_exact_paired_p(improved, degraded), 8),
        "pid_sign_flip_permutation_samples": permutations,
        "pid_sign_flip_two_sided_p": (
            round((extreme + 1) / (permutations + 1), 8)
            if permutations
            else None
        ),
        "inference_label": "exploratory",
    }
