"""人脸识别消融评测（Phase 4 · Step 28 / 3.5）—— 在 TinyFace 上量化"每加一个 feature 提升多少"。

定位：CSA demo 的"弹药"。在正规低分辨率人脸基准 **TinyFace**（QMUL, ACCV'18）上，按它的
开集识别协议算 **Rank-1 / Rank-5**，并逐步叠加攻"人脸模糊"的 feature，产出一张消融对比表/图
（baseline → +多帧融合 → +超分 → +3D 几何 → +AdaFace），证明每一步救回了多少识别率。

为什么是脚本而不是前端：前端吃的是"视频流 → 事件叙述"，TinyFace 是带身份标签的孤立人脸小图，
评测要做的是"批量提特征 → 按标签算准确率"这种定量打分，前端不做这件事。

TinyFace 结构（Testing_Set）：
  Gallery_Match/      已知身份的 gallery 人脸
  Probe/              待识别的低清 probe 人脸（与 gallery 同身份）
  Gallery_Distractor/ 15 万张干扰（无对应 probe，增加检索难度）
  gallery_match_img_ID_pairs.mat / probe_img_ID_pairs.mat  图名→身份ID 映射

协议：对每张 probe 提特征 → 在 (gallery_match + distractor) 里按余弦找最近 → 看 Top-K 里有没有
同身份 gallery → Rank-1/Rank-5 命中率。特征用 InsightFace ArcFace（512 维，已归一化）。

用法（数据在 git 仓库外）：
    python scripts/eval_face_recognition.py --data <TinyFace 根目录>
    python scripts/eval_face_recognition.py --data <dir> --features baseline,fusion --max-distractor 10000
    python scripts/eval_face_recognition.py --data <dir> --max-probe 500   # 先小样本跑通
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import face as face_mod  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs"


# ---------------- 数据集加载 ----------------
def _find_testing_set(data_dir: Path) -> Path:
    """容忍解压多套一层：自动找含 Testing_Set 的目录。"""
    for cand in (data_dir, *data_dir.glob("*"), *data_dir.glob("*/*")):
        if (cand / "Testing_Set").is_dir():
            return cand / "Testing_Set"
        if cand.name == "Testing_Set" and cand.is_dir():
            return cand
    raise FileNotFoundError(
        f"在 {data_dir} 下找不到 Testing_Set/。请确认 TinyFace 已解压，"
        f"目录里应有 Testing_Set\\Gallery_Match 等。"
    )


def _load_id_pairs(mat_path: Path) -> dict[str, int]:
    """读 TinyFace 的 *_img_ID_pairs.mat → {图名(无扩展名): 身份ID}。

    .mat 里通常是两个并列数组（图名 cell + ID 向量）；做容错解析。
    """
    from scipy import io as sio

    m = sio.loadmat(str(mat_path))
    keys = [k for k in m if not k.startswith("__")]
    names = ids = None
    for k in keys:
        arr = np.asarray(m[k]).squeeze()
        if arr.dtype.kind in ("U", "S", "O"):
            names = arr
        elif arr.dtype.kind in ("i", "u", "f"):
            ids = arr
    if names is None or ids is None:
        raise ValueError(f"无法解析 {mat_path.name} 的图名/ID 字段：keys={keys}")

    def _name(x) -> str:
        s = x[0] if isinstance(x, np.ndarray) and x.size else x
        return Path(str(s)).stem

    out: dict[str, int] = {}
    for nm, i in zip(names, np.asarray(ids).ravel()):
        out[_name(nm)] = int(i)
    return out


def _id_from_filename(stem: str) -> int | None:
    """TinyFace 文件名形如 [PersonID]_[xxx].jpg；从文件名兜底取身份。"""
    head = stem.split("_", 1)[0]
    return int(head) if head.isdigit() else None


def _list_images(folder: Path) -> list[Path]:
    return sorted([p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])


def _img_id(stem: str, pairs: dict[str, int]) -> int | None:
    return pairs.get(stem, _id_from_filename(stem))


# ---------------- 特征提取（baseline = InsightFace ArcFace）----------------
def _arcface_embed(path: Path, preprocess=None) -> np.ndarray | None:
    """对一张人脸图提 ArcFace 512 维归一化向量。

    TinyFace 的脸已裁好且很小，SCRFD 检测常失败 → 兜底：检测不到就把整图当脸送识别。
    preprocess: 可选的预处理 hook（超分/对齐等增强武器），输入/输出都是 PIL.Image。
    """
    from PIL import Image

    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None
    if preprocess is not None:
        try:
            img = preprocess(img)
        except Exception:
            pass
    faces = []
    try:
        faces = face_mod.detect(img, with_quality=False)
    except Exception:
        faces = []
    if faces:
        best = max(faces, key=lambda f: float(f.get("det_score", 0.0)))
        emb = best.get("embedding")
        if emb is not None:
            return np.asarray(emb, dtype=np.float32).reshape(-1)
    # 兜底：整图直接走识别模型（TinyFace 脸已对齐裁好）
    return _embed_whole(img)


def _embed_whole(img) -> np.ndarray | None:
    """检测失败时，把整张（已裁好的）人脸图直接喂 ArcFace 识别模型，提归一化向量。"""
    import numpy as _np

    face_mod._ensure_backend()
    app = face_mod._state["model"]["app"]
    rec = None
    for m in app.models.values():
        if getattr(m, "taskname", "") == "recognition":
            rec = m
            break
    if rec is None:
        return None
    bgr = _np.asarray(img.convert("RGB"))[:, :, ::-1]
    try:
        feat = rec.get_feat(bgr).reshape(-1).astype(_np.float32)
    except Exception:
        return None
    n = float(_np.linalg.norm(feat))
    return feat / n if n > 0 else feat


# ---------------- 增强武器（feature 开关）----------------
# 每个 feature 提供一个"特征提取器"。baseline=纯 ArcFace；其余为攻糊脸的进阶武器。
# 未实现的武器先返回 None hook（跑评测时会被跳过并提示），实现后填入真正的 preprocess。
def _make_extractor(feature: str):
    """返回 (extractor_fn(path)->vec, note)。extractor 内部决定如何提特征。"""
    if feature == "baseline":
        return (lambda p: _arcface_embed(p), "InsightFace ArcFace（纯 baseline）")
    if feature == "superres":
        pre = _try_superres_preprocess()
        if pre is None:
            return (None, "超分 GFP-GAN/CodeFormer（未安装，跳过）")
        return (lambda p: _arcface_embed(p, preprocess=pre), "ArcFace + 人脸超分预处理")
    if feature == "fusion":
        # 多帧融合在视频里才有意义；TinyFace 是单图，这里占位（同 baseline），仅为表格完整。
        return (lambda p: _arcface_embed(p), "多帧融合（单图基准下等同 baseline，占位）")
    if feature == "adaface":
        return (None, "AdaFace/MagFace（未接入，跳过）")
    if feature == "geom3d":
        return (None, "3D-68 几何 cue（未接入，跳过）")
    return (None, f"未知 feature: {feature}")


def _try_superres_preprocess():
    """若装了 gfpgan 则返回一个 PIL→PIL 的超分预处理；否则 None。"""
    try:
        # 占位：实现时在此加载 GFPGAN/CodeFormer 并返回 enhance 函数
        return None
    except Exception:
        return None


# ---------------- 评测核心 ----------------
def _build_matrix(paths: list[Path], pairs: dict[str, int], extractor, label: str,
                  max_n: int | None = None) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    """对一批图提特征，返回 (feats[N,512], ids[N], kept_paths)。提不出特征的丢弃。"""
    if max_n:
        paths = paths[:max_n]
    feats, ids, kept = [], [], []
    t0 = time.time()
    total = len(paths)
    print(f"    [{label}] 开始提特征：{total} 张（CPU 较慢，每 50 张报一次）", flush=True)
    for i, p in enumerate(paths):
        v = extractor(p)
        iid = _img_id(p.stem, pairs)
        if v is None or iid is None:
            continue
        feats.append(v)
        ids.append(iid)
        kept.append(p)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            rate = (i + 1) / el if el > 0 else 0
            eta = (total - (i + 1)) / rate if rate > 0 else 0
            print(f"    [{label}] {i+1}/{total}  {rate:.1f} 张/s  已用 {el:.0f}s  预计还需 {eta:.0f}s", flush=True)
    if not feats:
        return np.zeros((0, face_mod.FACE_DIM), np.float32), np.zeros((0,), np.int64), []
    return np.stack(feats).astype(np.float32), np.asarray(ids, np.int64), kept


def _rank_metrics(probe: np.ndarray, probe_ids: np.ndarray,
                  gallery: np.ndarray, gallery_ids: np.ndarray, ks=(1, 5, 10)) -> dict:
    """开集识别：每个 probe 在 gallery 里按余弦找最近，算 Rank-K 命中率。"""
    if probe.shape[0] == 0 or gallery.shape[0] == 0:
        return {f"rank{k}": 0.0 for k in ks} | {"n_probe": 0}
    sims = probe @ gallery.T  # 都已 L2 归一化 → 内积即余弦
    kmax = max(ks)
    topk = np.argpartition(-sims, kth=min(kmax, gallery.shape[0] - 1), axis=1)[:, :kmax]
    # 对 topk 再精确排序
    rows = np.arange(probe.shape[0])[:, None]
    order = np.argsort(-sims[rows, topk], axis=1)
    topk_sorted = topk[rows, order]
    hit_ids = gallery_ids[topk_sorted]  # [N, kmax]
    correct = hit_ids == probe_ids[:, None]
    out = {"n_probe": int(probe.shape[0])}
    for k in ks:
        out[f"rank{k}"] = round(float(correct[:, :k].any(axis=1).mean()) * 100, 2)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="人脸识别消融评测（TinyFace, Rank-1/5）")
    ap.add_argument("--data", required=True, help="TinyFace 根目录（含 Testing_Set）")
    ap.add_argument("--features", default="baseline",
                    help="逗号分隔，按顺序叠加评测：baseline,fusion,superres,geom3d,adaface")
    ap.add_argument("--max-probe", type=int, default=None, help="只评测前 N 张 probe（先跑通用小样本）")
    ap.add_argument("--max-distractor", type=int, default=0,
                    help="加入多少张干扰图（0=不加，先用 gallery_match 跑通；完整评测用大数/全量）")
    args = ap.parse_args()

    data_dir = Path(args.data)
    test = _find_testing_set(data_dir)
    print(f"[*] TinyFace Testing_Set: {test}")

    gm_dir = test / "Gallery_Match"
    pr_dir = test / "Probe"
    di_dir = test / "Gallery_Distractor"
    gm_pairs = _load_id_pairs(test / "gallery_match_img_ID_pairs.mat") if (test / "gallery_match_img_ID_pairs.mat").exists() else {}
    pr_pairs = _load_id_pairs(test / "probe_img_ID_pairs.mat") if (test / "probe_img_ID_pairs.mat").exists() else {}

    gm_imgs = _list_images(gm_dir)
    pr_imgs = _list_images(pr_dir)
    di_imgs = _list_images(di_dir) if (args.max_distractor and di_dir.is_dir()) else []
    print(f"[*] gallery_match={len(gm_imgs)}  probe={len(pr_imgs)}  distractor(用)={min(len(di_imgs), args.max_distractor)}")

    feature_list = [f.strip() for f in args.features.split(",") if f.strip()]
    rows = []
    print(f"[*] 后端：InsightFace {face_mod.active_backend()}  (CPU 提特征较慢，请耐心)")

    for feat in feature_list:
        extractor, note = _make_extractor(feat)
        print(f"\n===== feature = {feat} =====  {note}")
        if extractor is None:
            rows.append({"feature": feat, "note": note, "rank1": None, "rank5": None, "n_probe": 0})
            print("    （跳过：该 feature 尚未接入）")
            continue

        g_feat, g_ids, _ = _build_matrix(gm_imgs, gm_pairs, extractor, "gallery")
        if args.max_distractor and di_imgs:
            d_feat, d_ids, _ = _build_matrix(di_imgs, {}, extractor, "distractor", max_n=args.max_distractor)
            if d_feat.shape[0]:
                g_feat = np.concatenate([g_feat, d_feat], axis=0)
                g_ids = np.concatenate([g_ids, d_ids], axis=0)  # 干扰 ID 与 probe 永不相同
        p_feat, p_ids, _ = _build_matrix(pr_imgs, pr_pairs, extractor, "probe", max_n=args.max_probe)

        metrics = _rank_metrics(p_feat, p_ids, g_feat, g_ids)
        metrics.update({"feature": feat, "note": note, "gallery_size": int(g_feat.shape[0])})
        rows.append(metrics)
        print(f"    Rank-1={metrics['rank1']}%  Rank-5={metrics['rank5']}%  "
              f"(probe={metrics['n_probe']}, gallery={metrics.get('gallery_size')})")

    # ---- 输出表 + 图 ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _print_table(rows)
    _save_json(rows, OUT_DIR / "face_eval_results.json")
    _plot(rows, OUT_DIR / "face_eval_ablation.png")
    print(f"\n[✓] 结果表 → {OUT_DIR / 'face_eval_results.json'}")
    print(f"[✓] 消融图 → {OUT_DIR / 'face_eval_ablation.png'}")
    return 0


def _print_table(rows: list[dict]) -> None:
    print("\n================ 消融结果（TinyFace Rank-1/5）================")
    print(f"{'feature':<12}{'Rank-1':>9}{'Rank-5':>9}{'probe':>8}   note")
    for r in rows:
        r1 = "-" if r.get("rank1") is None else f"{r['rank1']}%"
        r5 = "-" if r.get("rank5") is None else f"{r['rank5']}%"
        print(f"{r['feature']:<12}{r1:>9}{r5:>9}{r.get('n_probe', 0):>8}   {r.get('note','')}")


def _save_json(rows: list[dict], path: Path) -> None:
    import json

    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _plot(rows: list[dict], path: Path) -> None:
    done = [r for r in rows if r.get("rank1") is not None]
    if not done:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
        plt.rcParams["axes.unicode_minus"] = False
        labels = [r["feature"] for r in done]
        r1 = [r["rank1"] for r in done]
        r5 = [r["rank5"] for r in done]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.6), 5))
        ax.bar(x - 0.2, r1, 0.4, label="Rank-1", color="#0078D4")
        ax.bar(x + 0.2, r5, 0.4, label="Rank-5", color="#107C10")
        for i, v in enumerate(r1):
            ax.text(i - 0.2, v + 0.5, f"{v}%", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("准确率 (%)")
        ax.set_title("人脸识别消融评测 · TinyFace（逐步叠加攻糊脸 feature）")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=120, facecolor="white")
    except Exception as exc:  # noqa: BLE001
        print(f"    （绘图跳过：{exc}）")


if __name__ == "__main__":
    raise SystemExit(main())
