from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from io import StringIO
from pathlib import Path
from typing import Any

from .common import (
    EXPERIMENT_DIR,
    SCHEMA_VERSION,
    atomic_write_json,
    atomic_write_text,
    load_protocol,
    stable_hash,
)
from .runner import DEFAULT_BACKENDS


BASE_TABLE_FIELDS = (
    "backend",
    "device_request",
    "actual_device",
    "embedding_dimension",
    "rank1_rate",
    "rank5_rate",
    "mAP_identity_template",
    "p2_rank1_rate",
    "p3_rank1_rate",
    "latency_ms_mean",
    "latency_ms_p95",
    "throughput_images_per_second",
    "model_load_seconds",
    "cuda_memory_after_load_bytes",
    "peak_cuda_memory_bytes",
    "model_weight_bytes",
)


def _result_path(
    root: Path,
    protocol_id: str,
    backend: str,
) -> Path:
    direct = root / f"{backend}.json"
    nested = root / protocol_id / f"{backend}.json"
    if direct.is_file():
        return direct
    if nested.is_file():
        return nested
    raise FileNotFoundError(
        f"缺少{backend}正式结果；检查过：{direct}, {nested}"
    )


def load_results(
    root: Path,
    protocol: dict[str, Any],
    backends: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    protocol_id = protocol["protocol_id"]
    manifest_sha256 = protocol["sample_manifest"]["sha256"]
    results = {}
    for backend in backends:
        path = _result_path(root, protocol_id, backend)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{backend}结果schema版本不支持：{path}")
        if payload.get("kind") != "body_reid_backend_result":
            raise ValueError(f"{backend}文件不是正式后端结果：{path}")
        if payload.get("protocol_id") != protocol_id:
            raise ValueError(
                f"{backend}协议不一致："
                f"expected={protocol_id}, actual={payload.get('protocol_id')}"
            )
        if payload.get("manifest_sha256") != manifest_sha256:
            raise ValueError(
                f"{backend}冻结样本哈希不一致："
                f"expected={manifest_sha256}, "
                f"actual={payload.get('manifest_sha256')}"
            )
        if payload.get("backend") != backend:
            raise ValueError(
                f"结果后端名称不一致：expected={backend}, "
                f"actual={payload.get('backend')}"
            )
        if payload.get("active_backend") != backend:
            raise ValueError(f"{backend}结果发生了后端回退")
        results[backend] = payload
    return results


def _validated_identity_hash(
    payload: dict[str, Any] | None,
    *,
    backend: str,
    label: str,
) -> str:
    if not payload or not payload.get("sha256"):
        raise ValueError(f"{backend}结果缺少{label}")
    identity = {key: value for key, value in payload.items() if key != "sha256"}
    actual = stable_hash(identity)
    if payload["sha256"] != actual:
        raise ValueError(
            f"{backend}的{label}哈希无效："
            f"expected={payload['sha256']}, actual={actual}"
        )
    return actual


def _query_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = result.get("queries") or []
    mapping = {row["query_id"]: row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError(f"{result['backend']}结果包含重复query_id")
    return mapping


def _validate_query_universe(
    results: dict[str, dict[str, Any]],
) -> list[str]:
    reference = None
    for backend, result in results.items():
        query_ids = sorted(_query_map(result))
        if reference is None:
            reference = query_ids
        elif query_ids != reference:
            raise ValueError(f"{backend}与其他后端的Query集合不一致")
    return reference or []


def _transition(
    baseline_rows: dict[str, dict[str, Any]],
    candidate_rows: dict[str, dict[str, Any]],
    subset_ids: set[str],
) -> dict[str, Any]:
    selected = sorted(
        query_id
        for query_id in subset_ids
        if baseline_rows[query_id]["genuine"]
    )
    rescued = [
        query_id
        for query_id in selected
        if not baseline_rows[query_id]["rank1_correct"]
        and candidate_rows[query_id]["rank1_correct"]
    ]
    harmed = [
        query_id
        for query_id in selected
        if baseline_rows[query_id]["rank1_correct"]
        and not candidate_rows[query_id]["rank1_correct"]
    ]
    return {
        "queries": len(selected),
        "rescued_count": len(rescued),
        "harmed_count": len(harmed),
        "net_rank1_change": len(rescued) - len(harmed),
        "rescued_query_ids": rescued,
        "harmed_query_ids": harmed,
    }


def _metric_row(result: dict[str, Any]) -> dict[str, Any]:
    protocols = result["protocols"]
    p1 = protocols["P1_overall"]["closed_set"]
    performance = result["performance"]
    row = {
        "backend": result["backend"],
        "device_request": result["device_request"],
        "actual_device": result["actual_device"],
        "embedding_dimension": result["embedding_dimension"],
        "rank1_rate": p1["rank1_rate"],
        "rank5_rate": p1["rank5_rate"],
        "mAP_identity_template": p1["mAP_identity_template"],
        "p2_rank1_rate": protocols["P2_cross_outfit"]["closed_set"][
            "rank1_rate"
        ],
        "p3_rank1_rate": protocols[
            "P3_same_outfit_cross_camera"
        ]["closed_set"]["rank1_rate"],
        "latency_ms_mean": performance["latency_ms_mean"],
        "latency_ms_p95": performance["latency_ms_p95"],
        "throughput_images_per_second": performance[
            "throughput_images_per_second"
        ],
        "model_load_seconds": performance["model_load_seconds"],
        "cuda_memory_after_load_bytes": performance[
            "cuda_memory_after_load_bytes"
        ],
        "peak_cuda_memory_bytes": performance["peak_cuda_memory_bytes"],
        "model_weight_bytes": result["provenance"]["model_weight_bytes"],
    }
    for key, point in protocols["P1_overall"]["open_set"][
        "operating_points"
    ].items():
        row[f"tpir_at_{key}"] = point["tpir"]
        row[f"observed_fmr_at_{key}"] = point["fmr"]
    return row


def build_summary(
    protocol: dict[str, Any],
    results: dict[str, dict[str, Any]],
    *,
    baseline: str = "osnet",
) -> dict[str, Any]:
    if baseline not in results:
        raise ValueError(f"基线结果不存在：{baseline}")
    all_query_ids = set(_validate_query_universe(results))
    baseline_rows = _query_map(results[baseline])
    genuine_ids = {
        query_id
        for query_id, row in baseline_rows.items()
        if row["genuine"]
    }
    p2_ids = {
        f"test:t{int(track):05d}"
        for track in protocol["test"]["p2_cross_outfit_query_tracks"]
    }
    p3_ids = {
        f"test:t{int(track):05d}"
        for track in protocol["test"][
            "p3_same_outfit_cross_camera_query_tracks"
        ]
    }
    for name, subset in (("P2", p2_ids), ("P3", p3_ids)):
        missing = subset - genuine_ids
        if missing:
            raise ValueError(
                f"{name}冻结Query不在正式genuine集合：{sorted(missing)[:3]}"
            )

    baseline_gallery = set(results[baseline]["gallery"]["test_template_pids"])
    baseline_environment = _validated_identity_hash(
        results[baseline].get("comparison_environment"),
        backend=baseline,
        label="比较环境",
    )
    baseline_evaluation = _validated_identity_hash(
        results[baseline].get("evaluation_config"),
        backend=baseline,
        label="评测配置",
    )
    manifest_sha256 = protocol["sample_manifest"]["sha256"]
    operating_keys = set(
        results[baseline]["protocols"]["P1_overall"]["open_set"][
            "operating_points"
        ]
    )
    for backend, result in results.items():
        if result.get("manifest_sha256") != manifest_sha256:
            raise ValueError(f"{backend}的冻结样本哈希与协议不一致")
        environment = _validated_identity_hash(
            result.get("comparison_environment"),
            backend=backend,
            label="比较环境",
        )
        if environment != baseline_environment:
            raise ValueError(f"{backend}与基线的运行环境或实验代码不一致")
        evaluation_config = _validated_identity_hash(
            result.get("evaluation_config"),
            backend=backend,
            label="评测配置",
        )
        if evaluation_config != baseline_evaluation:
            raise ValueError(f"{backend}与基线的评测配置不一致")
        if set(result["gallery"]["test_template_pids"]) != baseline_gallery:
            raise ValueError(f"{backend}的Gallery身份模板集合与基线不一致")
        result_p2 = set(
            result["protocols"]["P2_cross_outfit"]["query_ids"]
        )
        result_p3 = set(
            result["protocols"]["P3_same_outfit_cross_camera"]["query_ids"]
        )
        if result_p2 != p2_ids or result_p3 != p3_ids:
            raise ValueError(f"{backend}的P2/P3 Query集合与冻结协议不一致")
        candidate_keys = set(
            result["protocols"]["P1_overall"]["open_set"][
                "operating_points"
            ]
        )
        if candidate_keys != operating_keys:
            raise ValueError(f"{backend}的FMR目标集合与基线不一致")

    comparisons = {}
    for backend, result in results.items():
        if backend == baseline:
            continue
        candidate_rows = _query_map(result)
        comparisons[backend] = {
            "P1_overall": _transition(
                baseline_rows,
                candidate_rows,
                all_query_ids,
            ),
            "P2_cross_outfit": _transition(
                baseline_rows,
                candidate_rows,
                p2_ids,
            ),
            "P3_same_outfit_cross_camera": _transition(
                baseline_rows,
                candidate_rows,
                p3_ids,
            ),
        }

    table = [_metric_row(result) for result in results.values()]
    open_set = {
        backend: result["protocols"]["P1_overall"]["open_set"][
            "operating_points"
        ]
        for backend, result in results.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "body_reid_matrix_summary",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "manifest_sha256": manifest_sha256,
        "baseline": baseline,
        "backends": list(results),
        "evaluation_config": results[baseline]["evaluation_config"],
        "comparison_environment": results[baseline][
            "comparison_environment"
        ],
        "query_universe": {
            "all": len(all_query_ids),
            "genuine": len(genuine_ids),
            "imposter": len(all_query_ids - genuine_ids),
            "P2_cross_outfit": len(p2_ids),
            "P3_same_outfit_cross_camera": len(p3_ids),
        },
        "table": table,
        "open_set": open_set,
        "paired_vs_baseline": comparisons,
    }


def _serialize_table(rows: list[dict[str, Any]]) -> str:
    extra_fields = sorted(
        {
            field
            for row in rows
            for field in row
            if field not in BASE_TABLE_FIELDS
        }
    )
    fields = (*BASE_TABLE_FIELDS, *extra_fields)
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    return output.getvalue()


def _parse_backends(value: str) -> tuple[str, ...]:
    backends = tuple(
        dict.fromkeys(item.strip() for item in value.split(",") if item.strip())
    )
    if not backends:
        raise ValueError("--backends不能为空")
    return backends


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="校验并汇总MEVID人形ReID四后端正式结果"
    )
    parser.add_argument(
        "--protocol",
        default=str(EXPERIMENT_DIR / "manifests" / "frozen_protocol.json"),
    )
    parser.add_argument(
        "--results",
        default=str(EXPERIMENT_DIR / "results" / "runs"),
    )
    parser.add_argument(
        "--backends",
        default=",".join(DEFAULT_BACKENDS),
    )
    parser.add_argument("--baseline", default="osnet")
    parser.add_argument(
        "--output",
        default=str(
            EXPERIMENT_DIR
            / "results"
            / "final"
            / "body_reid_matrix_summary.json"
        ),
    )
    parser.add_argument(
        "--table",
        default=str(
            EXPERIMENT_DIR
            / "results"
            / "final"
            / "body_reid_matrix_summary.csv"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol = load_protocol(Path(args.protocol).resolve())
    backends = _parse_backends(args.backends)
    results = load_results(
        Path(args.results).resolve(),
        protocol,
        backends,
    )
    payload = build_summary(protocol, results, baseline=args.baseline)
    output_path = Path(args.output).resolve()
    table_path = Path(args.table).resolve()
    existing = [path for path in (output_path, table_path) if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "最终汇总已存在，使用--force覆盖："
            + ", ".join(str(path) for path in existing)
        )
    atomic_write_json(output_path, payload)
    atomic_write_text(table_path, _serialize_table(payload["table"]))
    print(f"[saved] {output_path}")
    print(f"[saved] {table_path}")
    return 0
