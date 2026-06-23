"""Phase 3 评估体系（Step 19）—— 量化证明"既省钱、又没掉精度"。

定位：Phase 2/3 风险清单反复点名"评估缺位"——没有量化指标，就无法证明"逐轨迹识别 +
主体记忆 + 缓存复用"真的既省 LLM 调用、又没掉识别精度。本脚本就是补这个缺口：跑通
`tracker → reid → gallery → track_fusion` 这条叶子管线，算出一组对 manager 有说服力的指标。

**与编排解耦**：Step 12 三时钟编排尚未接线，本脚本**直接驱动各叶子模块**（不依赖 /analyze
的串联），因此现在就能独立跑，也不和任何在做的模块冲突。

两种模式：
  1. **synthetic（默认，零数据）**：自动造若干"身份×多次出现×多帧"的合成行人（带颜色身份、
     运动、偶发单帧噪声），跑完整识别管线 → 出指标 + 报告。用于验证评估骨架本身，并给一份样例报告。
  2. **manifest（真实数据）**：`--manifest data.json`，喂带标注的 crop 序列，出真报告。
     manifest 格式（每条 = 一帧里某 track 的一个目标 crop + 真值身份）：
        {"observations": [
            {"crop": "crops/0001.jpg", "identity": "alice", "track_id": 1, "frame_idx": 0},
            ...
        ]}

指标：
  - **识别精度/召回/F1**：用 purity / inverse-purity（把"预测 subject_id"当聚类，对"真值身份"评）。
  - **ID switch**：每个真值身份的时间线上，预测身份翻了几次（越低越稳）。
  - **单位视频 LLM 调用数 + 省比**：track-and-identify 只在 new/grey 触发 LLM；对比"每帧每目标都调"的 baseline。
  - **记忆命中率**：gallery 命中（hit）占比。
  - **逐帧 vs 融合**：同时报"逐帧 gallery 判定"与"track 融合后判定"，证明多帧投票/融合纠错有效。

用法：
    python scripts/eval_phase3.py                          # 合成自检（默认 coarse 后端）
    python scripts/eval_phase3.py --identities 6 --noise 0.25
    python scripts/eval_phase3.py --manifest data/eval.json --backend resnet50
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Windows 控制台 UTF-8（同 vertical_slice.py）
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUT_DIR = Path(__file__).resolve().parents[1] / "out"


# --------------------------------------------------------------------------
# 数据：合成 / manifest
# --------------------------------------------------------------------------
def _synthetic_observations(identities: int, appearances: int, frames: int, noise: float):
    """造合成观测：每个身份一种基色，出现 `appearances` 次（每次一个新 track_id，考验跨 track 记忆），
    每次 `frames` 帧（带抖动 + 偶发单帧噪声）。返回 [(crop, identity, track_id, frame_idx)]。"""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(42)
    # 在色相环上均匀取 identities 个高饱和颜色，保证可区分
    import colorsys
    base_colors = []
    for i in range(identities):
        r, g, b = colorsys.hsv_to_rgb(i / identities, 0.85, 0.9)
        base_colors.append((int(r * 255), int(g * 255), int(b * 255)))

    obs = []
    track_id = 0
    for ident in range(identities):
        color = base_colors[ident]
        for _ in range(appearances):
            track_id += 1
            for fi in range(frames):
                corrupt = rng.random() < noise
                if corrupt:
                    # 单帧噪声：极大抖动 + 偏小 + 偶尔串到邻近颜色（模拟糊帧/遮挡/误检）
                    jitter = rng.integers(-80, 80, (200, 80, 3))
                    if rng.random() < 0.5:
                        bad = base_colors[(ident + 1) % identities]
                        patch = np.full((200, 80, 3), bad, dtype=np.int16) + jitter
                    else:
                        patch = np.full((200, 80, 3), color, dtype=np.int16) + jitter
                    size = (40, 100)  # 偏小 → 质量门控/最佳帧会降权
                else:
                    patch = np.full((200, 80, 3), color, dtype=np.int16) + rng.integers(-12, 12, (200, 80, 3))
                    size = (80, 200)
                img = Image.fromarray(np.clip(patch, 0, 255).astype("uint8")).resize(size)
                # 给个随平移的 box（让运动连续性线索有意义）
                x = 100 + fi * 5 + (int(rng.integers(-3, 3)))
                box = [x, 60, x + size[0], 60 + size[1]]
                obs.append((img, f"id{ident}", track_id, fi, box))
    return obs


def _manifest_observations(path: Path):
    """从 manifest 读取真实标注观测，返回 [(crop, identity, track_id, frame_idx, box)]。"""
    from PIL import Image

    spec = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    obs = []
    for o in spec.get("observations", []):
        crop_path = (base / o["crop"]) if not Path(o["crop"]).is_absolute() else Path(o["crop"])
        img = Image.open(crop_path).convert("RGB")
        box = o.get("box") or [0, 0, img.width, img.height]
        obs.append((img, str(o["identity"]), int(o["track_id"]), int(o.get("frame_idx", 0)), box))
    return obs


# --------------------------------------------------------------------------
# 跑管线：reid → gallery → fusion
# --------------------------------------------------------------------------
def run_pipeline(observations, session_id="eval"):
    """对每条观测：提 ReID 指纹 → 查/登记主体记忆 → 累积进融合。返回逐帧记录 + 各 track 融合结论。"""
    from app import gallery as gallery_mod
    from app import reid, track_fusion
    from app.utils import dominant_color

    gallery_mod.reset_gallery(session_id)
    track_fusion.reset_fusion(session_id)
    g = gallery_mod.get_gallery(session_id, reid.embed_dim())

    records = []  # 每条：dict(identity, track_id, frame_idx, pred_subject, decision, score)
    decision_counts = Counter()
    for crop, identity, track_id, frame_idx, box in observations:
        vec = reid.embed(crop)
        quality = reid.assess_quality(crop)
        color = dominant_color(crop, [0, 0, crop.width, crop.height], "whole")
        res = g.identify_or_enroll(vec, quality, auto_enroll=True)
        decision_counts[res["decision"]] += 1
        track_fusion.add_observation(
            session_id, track_id, frame_idx=frame_idx, box=box, quality=quality,
            reid_subject=res.get("subject_id"), reid_decision=res["decision"],
            reid_score=res.get("score", 0.0), color=color,
        )
        records.append({
            "identity": identity, "track_id": track_id, "frame_idx": frame_idx,
            "pred_subject": res.get("subject_id"), "decision": res["decision"],
            "score": res.get("score", 0.0),
        })

    # 各 track 的融合裁决（track 级身份）
    fused = {}
    for track_id in {r["track_id"] for r in records}:
        fused[track_id] = track_fusion.resolve_track(session_id, track_id).get("subject_id")
    return records, fused, decision_counts, g.stats()["subjects"]


# --------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------
def _cluster_prf(pairs):
    """pairs=[(gt, pred)]，用 purity/inverse-purity 算 (precision, recall, F1)。

    precision(purity)：每个预测簇里"占多数的真值"的占比之和 / N —— 簇有多纯。
    recall(inverse-purity)：每个真值里"占多数的预测簇"的占比之和 / N —— 真值有没有被打散。
    pred 为 None 视作各自独立的错误簇。
    """
    pairs = [(gt, (pred if pred is not None else f"_none_{i}")) for i, (gt, pred) in enumerate(pairs)]
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0, 0.0
    by_pred = defaultdict(Counter)
    by_gt = defaultdict(Counter)
    for gt, pred in pairs:
        by_pred[pred][gt] += 1
        by_gt[gt][pred] += 1
    precision = sum(c.most_common(1)[0][1] for c in by_pred.values()) / n
    recall = sum(c.most_common(1)[0][1] for c in by_gt.values()) / n
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def _id_switches(records, pred_key):
    """每个真值身份的时间线上，预测身份发生切换的次数之和（越低越稳）。"""
    by_gt = defaultdict(list)
    for r in sorted(records, key=lambda r: (r["identity"], r["track_id"], r["frame_idx"])):
        by_gt[r["identity"]].append(r[pred_key])
    switches = 0
    for seq in by_gt.values():
        prev = None
        for p in seq:
            if p is not None and prev is not None and p != prev:
                switches += 1
            if p is not None:
                prev = p
    return switches


def compute_metrics(records, fused, decision_counts, n_subjects):
    n = len(records)
    gt_identities = len({r["identity"] for r in records})

    # 逐帧（gallery 直接判定）
    frame_pairs = [(r["identity"], r["pred_subject"]) for r in records]
    fp, fr, ff1 = _cluster_prf(frame_pairs)

    # 融合（track 级裁决回填到每条观测）
    fused_pairs = [(r["identity"], fused.get(r["track_id"])) for r in records]
    up, ur, uf1 = _cluster_prf(fused_pairs)

    # 成本：track-and-identify 只在 new/grey 触发 LLM；baseline = 每帧每目标都调
    llm_calls = decision_counts.get("new", 0) + decision_counts.get("grey", 0)
    baseline = n
    savings = (1 - llm_calls / baseline) if baseline else 0.0
    hit_rate = decision_counts.get("hit", 0) / n if n else 0.0

    return {
        "observations": n,
        "gt_identities": gt_identities,
        "discovered_subjects": n_subjects,
        "per_frame": {"precision": round(fp, 4), "recall": round(fr, 4), "f1": round(ff1, 4),
                      "id_switches": _id_switches(records, "pred_subject")},
        "fused": {"precision": round(up, 4), "recall": round(ur, 4), "f1": round(uf1, 4),
                  "id_switches": _id_switches(
                      [{**r, "fused_pred": fused.get(r["track_id"])} for r in records], "fused_pred")},
        "cost": {
            "llm_calls_track_identify": llm_calls,
            "llm_calls_baseline_every_frame": baseline,
            "savings_pct": round(savings * 100, 1),
            "gallery_hit_rate": round(hit_rate, 4),
            "decision_breakdown": dict(decision_counts),
        },
    }


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------
def print_report(m: dict) -> None:
    pf, fu, c = m["per_frame"], m["fused"], m["cost"]
    print("\n================  Phase 3 评估报告  ================")
    print(f"观测数 {m['observations']}  |  真值身份 {m['gt_identities']}  |  发现主体 {m['discovered_subjects']}")
    print("\n--- 识别精度（越高越好；融合应 ≥ 逐帧）---")
    print(f"  逐帧 gallery : precision={pf['precision']}  recall={pf['recall']}  "
          f"F1={pf['f1']}  ID切换={pf['id_switches']}")
    print(f"  track 融合   : precision={fu['precision']}  recall={fu['recall']}  "
          f"F1={fu['f1']}  ID切换={fu['id_switches']}")
    print("\n--- 成本（核心卖点：省 LLM 调用而不掉精度）---")
    print(f"  track-and-identify LLM 调用 : {c['llm_calls_track_identify']}")
    print(f"  每帧 baseline LLM 调用       : {c['llm_calls_baseline_every_frame']}")
    print(f"  ==> 省比 {c['savings_pct']}%   记忆命中率 {c['gallery_hit_rate']}")
    print(f"  判定分布 : {c['decision_breakdown']}")
    print("====================================================\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase 3 识别+省钱评估")
    ap.add_argument("--manifest", type=str, default=None, help="真实标注 manifest（不传则合成自检）")
    ap.add_argument("--backend", type=str, default=None, help="reid 后端 coarse|resnet50|osnet|auto")
    ap.add_argument("--identities", type=int, default=5, help="合成：身份数")
    ap.add_argument("--appearances", type=int, default=2, help="合成：每个身份出现次数（不同 track）")
    ap.add_argument("--frames", type=int, default=6, help="合成：每次出现的帧数")
    ap.add_argument("--noise", type=float, default=0.2, help="合成：单帧噪声概率")
    ap.add_argument("--out", type=str, default=str(OUT_DIR / "eval_phase3.json"))
    args = ap.parse_args()

    if args.backend:
        import os
        os.environ["REID_BACKEND"] = args.backend

    if args.manifest:
        path = Path(args.manifest)
        if not path.exists():
            print(f"[X] 找不到 manifest：{path}")
            return 1
        print(f"[1/3] 读取标注 manifest：{path}")
        observations = _manifest_observations(path)
    else:
        print(f"[1/3] 合成自检：{args.identities} 身份 × {args.appearances} 次出现 × {args.frames} 帧"
              f"（噪声 {args.noise}）")
        observations = _synthetic_observations(args.identities, args.appearances, args.frames, args.noise)

    from app import reid
    print(f"[2/3] 跑管线（ReID 后端 = {reid.active_backend()}, dim = {reid.embed_dim()}）...")
    records, fused, decisions, n_subjects = run_pipeline(observations)

    print("[3/3] 计算指标...")
    metrics = compute_metrics(records, fused, decisions, n_subjects)
    metrics["reid_backend"] = reid.active_backend()
    print_report(metrics)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"指标已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
