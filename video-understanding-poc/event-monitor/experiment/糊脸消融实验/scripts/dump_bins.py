# -*- coding: utf-8 -*-
"""把 probe 图按人脸质量桶落盘，便于人眼查看「筛出来的糊脸长什么样」。

只做：划分 gallery/probe（同 run_eval 逻辑）→ 人脸质量分桶 → 复制图到
    dataset/probe_by_bin/{clear,blur,tiny,none}/  和 dataset/gallery/
不提认人特征（不跑 ReID/多 arm），所以比 run_eval 快很多。

用法（从仓库根运行）：
    python experiment/糊脸消融实验/scripts/dump_bins.py --data data/external/Market-1501-v15.09.15 --max-subjects 25
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

HERE = Path(__file__).resolve().parent
EXPERIMENT_DIR = HERE.parent
ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(ROOT))

# 复用 run_eval 的产品调用 / 分桶 / 加载逻辑（按文件路径加载，避开中文包名 import）
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("run_eval", HERE / "run_eval.py")
run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval)


def main() -> int:
    ap = argparse.ArgumentParser(description="把 probe 图按人脸质量桶落盘查看")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="bounding_box_test")
    ap.add_argument("--max-subjects", type=int, default=25)
    ap.add_argument("--gallery-per-subject", type=int, default=3)
    ap.add_argument("--probe-per-subject", type=int, default=8)
    ap.add_argument("--per-bin-cap", type=int, default=40, help="每桶最多复制多少张（够看即可）")
    ap.add_argument("--name", default=None,
                    help="运行名（dataset/<name>/ 下落盘，区分不同实验）；默认自动 数据集_split_人数_日期")
    args = ap.parse_args()

    from datetime import datetime

    from PIL import Image
    from app import face as face_mod
    from app.core.config import settings

    settings.face_superres = "off"

    # 每次跑落盘到独立命名子文件夹，互不覆盖，便于区分对比
    ds_short = Path(args.data).name.split("-")[0].lower()  # 如 Market1501-... → market1501
    run_name = args.name or f"{ds_short}_{args.split}_{args.max_subjects}subj_{datetime.now():%Y%m%d-%H%M}"
    out_root = EXPERIMENT_DIR / "dataset" / run_name
    if out_root.exists():
        shutil.rmtree(out_root)  # 只清同名这一次的，保留其他运行
    (out_root / "gallery").mkdir(parents=True, exist_ok=True)
    print(f"[*] 本次落盘目录：dataset/{run_name}/")

    data_dir = Path(args.data)
    by_pid = run_eval.load_market(data_dir, args.split, args.max_subjects)
    print(f"[*] 身份数={len(by_pid)}；划分 + 分桶 + 落盘 …")

    bin_counts: dict[str, int] = defaultdict(int)
    gallery_n = 0
    manifest_lines = ["path,subject,camera,bin"]

    for pid, items in by_pid.items():
        scored = []
        for (path, cam) in items:
            try:
                pil = Image.open(path).convert("RGB")
            except Exception:
                continue
            bf = run_eval.face_detect(face_mod, settings, pil, "arcface", False, with_quality=True)
            side = (min(bf["bbox"][2] - bf["bbox"][0], bf["bbox"][3] - bf["bbox"][1]) if bf else 0.0)
            det = float(bf.get("det_score", 0.0)) if bf else 0.0
            qbin = run_eval.quality_bin(bf)
            scored.append(((path, cam), det * 100 + side, qbin))
        scored.sort(key=lambda x: -x[1])

        gsel = scored[:args.gallery_per_subject]
        psel = scored[args.gallery_per_subject:args.gallery_per_subject + args.probe_per_subject]

        for (path, cam), _, qbin in gsel:
            dst = out_root / "gallery" / f"{pid}_c{cam}_{path.name}"
            shutil.copy2(path, dst)
            gallery_n += 1
            manifest_lines.append(f"{path.name},{pid},{cam},gallery")

        for (path, cam), _, qbin in psel:
            if bin_counts[qbin] >= args.per_bin_cap:
                manifest_lines.append(f"{path.name},{pid},{cam},{qbin}")
                bin_counts[qbin] += 1
                continue
            bdir = out_root / "probe_by_bin" / qbin
            bdir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, bdir / f"{pid}_c{cam}_{path.name}")
            bin_counts[qbin] += 1
            manifest_lines.append(f"{path.name},{pid},{cam},{qbin}")

    (out_root / "manifest.csv").write_text("\n".join(manifest_lines), encoding="utf-8")

    base = f"dataset/{run_name}"
    print(f"[✓] gallery 图：{gallery_n} 张 → {base}/gallery/")
    print("[✓] probe 按桶：")
    for b in run_eval.BIN_ORDER:
        n = bin_counts.get(b, 0)
        capped = min(n, args.per_bin_cap)
        print(f"      {b:<8} 共 {n} 张（落盘 {capped}） → {base}/probe_by_bin/{b}/")
    print(f"[✓] 清单：{base}/manifest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
