"""MEVID 多模态消融评测（人脸 + 人形 + 步态）。

历史设计见 `../docs/实验设计_MEVID多模态.md`。要点：
  · baseline = 完整多模态（ArcFace + OSNet 人形 + SkeletonGaitPP 步态）；主线只换人脸模型。
  · 分桶用产品 `app.face.assess_quality`（经 face.detect(with_quality)）；
    人脸建档走真门控（category==clear 才入库）；融合复刻产品 multimodal_identity_fusion 的软性连续加权。
  · 开集：一部分身份只作冒充者（不建档），用阈值判命中/拒识 → 误识率 FMR / 拒识率 + DET。
  · 每次 run 独立、内存化模板库（不落持久 gallery），杜绝跨 run 串味。

数据：MEVID test（`--data <mevid 根>`，含 bbox_test/ 与 annotation/mevid-v1-annotation-data/）。
用法示例（小规模验证）：
    python run_mevid.py --data /data/external/mevid \
        --arms M0,M1,M2,M3,Fonly,FB --enroll-subjects 10 --imposter-subjects 10 \
        --faces-per-track 8 --gait-frames 24 --anchor monitor
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(ROOT))
OUT_DIR = EXPERIMENT_DIR / "results" / "runs"

# ---------------------------------------------------------------------------
# arm 定义：人脸后端 / 超分 / 人形 / 步态。主线 M0–M3 固定人形+步态；辅助线 Fonly/FB 消融模态。
# ---------------------------------------------------------------------------
ARMS = {
    "M0":    {"backend": "arcface", "superres": False, "body": True,  "gait": True,  "note": "baseline: ArcFace+人形+步态"},
    "M1":    {"backend": "adaface", "superres": False, "body": True,  "gait": True,  "note": "+AdaFace"},
    "M2":    {"backend": "arcface", "superres": True,  "body": True,  "gait": True,  "note": "+超分"},
    "M3":    {"backend": "adaface", "superres": True,  "body": True,  "gait": True,  "note": "full: AdaFace+超分+人形+步态"},
    "Fonly": {"backend": "arcface", "superres": False, "body": False, "gait": False, "note": "辅助线: 纯人脸"},
    "FB":    {"backend": "arcface", "superres": False, "body": True,  "gait": False, "note": "辅助线: 人脸+人形"},
}
BIN_ORDER = ["clear", "marginal", "poor", "none"]

# 文件名形如 0201O003C330T016F00365.jpg
_NAME_RE = re.compile(r"^(?P<pid>\d+)O(?P<outfit>\d+)C(?P<cam>\d+)T(?P<track>\d+)F(?P<frame>\d+)$")


# ---------------------------------------------------------------------------
# MEVID 官方标注加载
# ---------------------------------------------------------------------------
class Tracklet:
    __slots__ = ("pid", "cam", "outfit", "track", "frames", "is_query")

    def __init__(self, pid, cam, outfit, track):
        self.pid = pid
        self.cam = cam
        self.outfit = outfit
        self.track = track
        self.frames: list[Path] = []
        self.is_query = False


def load_mevid(data_dir: Path):
    """读官方标注，返回 (tracklets: list[Tracklet], ann_dir)。

    track_test_info.txt：每行 [start_idx, end_idx, pid, cam, outfit]（浮点写法）。
    test_name.txt：行号→图片名（<pid>O<o>C<c>T<t>F<f>.jpg）。
    query_IDX.txt：作为 query 的轨迹行号（指向 track_test_info 的行，0-based）。
    """
    ann = data_dir / "annotation" / "mevid-v1-annotation-data"
    bbox = data_dir / "bbox_test"
    names = (ann / "test_name.txt").read_text().split()
    info_lines = (ann / "track_test_info.txt").read_text().strip().splitlines()
    query_idx = {int(float(x)) for x in (ann / "query_IDX.txt").read_text().split()}

    tracklets: list[Tracklet] = []
    for row, line in enumerate(info_lines):
        parts = line.split()
        if len(parts) < 4:
            continue
        start, end = int(float(parts[0])), int(float(parts[1]))
        pid, cam = int(float(parts[2])), int(float(parts[3]))
        outfit = int(float(parts[4])) if len(parts) > 4 else 0
        pid_s = f"{pid:04d}"
        tk = Tracklet(pid=pid_s, cam=cam, outfit=outfit, track=row)
        # 帧路径：test_name 里 start..end 行；图片放在 bbox_test/<pid>/<name>
        for gi in range(start, end + 1):
            if gi >= len(names):
                break
            nm = names[gi]
            p = bbox / pid_s / nm
            tk.frames.append(p)
        tk.is_query = row in query_idx
        if tk.frames:
            tracklets.append(tk)
    return tracklets, ann


def load_checkin_anchors(data_dir: Path) -> dict[str, list[Path]]:
    """登记照锚点：actor_checkin/<...>/NNN-date-...-f.jpg。前缀数字 NNN → pid（补零到 4 位）。
    只取正面 `-f`。返回 {pid: [照片路径]}。"""
    base = data_dir / "actor_checkin"
    photos = list(base.rglob("*.jpg")) + list(base.rglob("*.png"))
    by_pid: dict[str, list[Path]] = defaultdict(list)
    for p in photos:
        m = re.match(r"^(\d+)-.*", p.stem)
        if not m:
            continue
        if p.stem.endswith("-b"):  # 跳过背面
            continue
        pid = f"{int(m.group(1)):04d}"
        by_pid[pid].append(p)
    return by_pid


# ---------------------------------------------------------------------------
# 采样：每轨迹均匀抽 n 帧
# ---------------------------------------------------------------------------
def _sample(seq: list, n: int) -> list:
    if n <= 0 or len(seq) <= n:
        return seq
    step = len(seq) / n
    return [seq[int(i * step)] for i in range(n)]


def _norm(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    nrm = float(np.linalg.norm(v))
    return v / nrm if nrm > 0 else v


# ---------------------------------------------------------------------------
# 特征提取（call 产品代码）
# ---------------------------------------------------------------------------
def _face_of(face_mod, settings, pil, backend: str, superres: bool):
    """call 产品 face.detect：返回该图最佳脸 dict（含 embedding/category/quality），无脸→None。"""
    settings.face_rec_backend = backend
    try:
        faces = face_mod.detect(pil, with_quality=True, enhance_blurry=superres)
    except Exception:
        return None
    if not faces:
        return None
    return max(faces, key=lambda f: float(f.get("det_score", 0.0)))


def extract_track_features(tk, face_mod, reid_mod, gait_mod, settings, face_cfgs,
                           need_body, need_gait, faces_per_track: int, gait_frames: int):
    """一次遍历轨迹采样帧，算好：每种人脸配置 (backend,sr) 的帧级 emb/cat/q + 最佳脸；
    以及**共享**的人形均值、步态向量（与 arm 无关，只算一次）。

    返回 rec：
      face_cfg: {(backend,sr): {"frames":[{emb,cat,q}], "best_emb","best_cat","best_q"}}
      body / gait: 归一化向量 或 None
    """
    from PIL import Image

    rec = {"pid": tk.pid, "face_cfg": {}, "body": None, "gait": None}
    for key in face_cfgs:
        rec["face_cfg"][key] = {"frames": [], "best_emb": None, "best_cat": "none",
                                "best_q": None, "_best_det": -1.0}

    face_imgs = _sample(tk.frames, faces_per_track)
    body_vecs = []
    for p in face_imgs:
        try:
            pil = Image.open(p).convert("RGB")
        except Exception:
            continue
        for (backend, sr) in face_cfgs:
            bf = _face_of(face_mod, settings, pil, backend, sr)
            slot = rec["face_cfg"][(backend, sr)]
            cat, emb, q = "none", None, None
            if bf is not None:
                qd = bf.get("quality") or {}
                cat = qd.get("category", "poor")
                q = qd.get("quality")
                e = bf.get("embedding")
                emb = _norm(e) if e is not None else None
                det = float(bf.get("det_score", 0.0))
                if emb is not None and det > slot["_best_det"]:
                    slot["_best_det"] = det
                    slot["best_emb"], slot["best_cat"], slot["best_q"] = emb, cat, q
            slot["frames"].append({"emb": emb, "cat": cat, "q": q})
        if need_body:
            try:
                body_vecs.append(_norm(reid_mod.embed(pil)))
            except Exception:
                pass
    if need_body and body_vecs:
        rec["body"] = _norm(np.mean(body_vecs, axis=0))
    if need_gait:
        rec["gait"] = _gait_of(tk, gait_mod, gait_frames)
    return rec


def _gait_of(tk, gait_mod, gait_frames: int):
    """对一条轨迹的采样帧逐帧提姿态+剪影，累积成序列 → 步态向量。crop 即单人，取最大人。"""
    import cv2
    from PIL import Image

    frames = _sample(tk.frames, gait_frames)
    pose_seq, sil_seq = [], []
    for p in frames:
        try:
            bgr = cv2.cvtColor(np.asarray(Image.open(p).convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            continue
        try:
            persons = gait_mod.extract_persons(bgr)
        except Exception:
            continue
        if not persons:
            continue
        # crop 即该人：取框面积最大者
        best = max(persons, key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
        pose_seq.append(best["kpts"])
        sil_seq.append(best["mask"])
    if len(pose_seq) < 1:
        return None
    try:
        return gait_mod.embed_track(pose_seq, sil_seq)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 建档（真门控）：人脸 clear 才入库；人形/步态有向量即入库
# ---------------------------------------------------------------------------
def build_gallery(enroll_recs_by_pid, cfg, fkey):
    """enroll_recs_by_pid: {pid: [track_rec,...]}；fkey=(backend,sr) 选人脸配置。
    返回各模态模板 + 建档覆盖统计。人脸真门控：仅 category==clear 帧入库。"""
    face_tpl, body_tpl, gait_tpl = {}, {}, {}
    cov = {"face": 0, "body": 0, "gait": 0, "total": len(enroll_recs_by_pid)}
    for pid, recs in enroll_recs_by_pid.items():
        clear_embs = []
        for r in recs:
            for ff in r["face_cfg"][fkey]["frames"]:
                if ff["emb"] is not None and ff["cat"] == "clear":
                    clear_embs.append(ff["emb"])
        if clear_embs:
            face_tpl[pid] = _norm(np.mean(clear_embs, axis=0))
            cov["face"] += 1
        if cfg["body"]:
            bvs = [r["body"] for r in recs if r["body"] is not None]
            if bvs:
                body_tpl[pid] = _norm(np.mean(bvs, axis=0)); cov["body"] += 1
        if cfg["gait"]:
            gvs = [r["gait"] for r in recs if r["gait"] is not None]
            if gvs:
                gait_tpl[pid] = _norm(np.mean(gvs, axis=0)); cov["gait"] += 1
    return face_tpl, body_tpl, gait_tpl, cov


def _fused_scores(rec, cfg, fkey, tpls, subjects, weights):
    """复刻产品 multimodal_identity_fusion 软性连续加权：返回 {pid: fused_score}。"""
    wf, wb, wg, floor, agree_bonus = weights
    face_tpl, body_tpl, gait_tpl = tpls
    slot = rec["face_cfg"][fkey]
    fv = slot["best_emb"]
    q = slot["best_q"]
    bv = rec["body"] if cfg["body"] else None
    gv = rec["gait"] if cfg["gait"] else None
    if fv is not None:
        ef = wf * (floor + (1.0 - floor) * max(0.0, min(1.0, float(q)))) if q is not None else wf * floor
    else:
        ef = 0.0
    eb = wb if bv is not None else 0.0
    eg = wg if gv is not None else 0.0
    out = {}
    for pid in subjects:
        num = wsum = 0.0
        nroutes = 0
        if ef > 0 and pid in face_tpl:
            num += ef * float(face_tpl[pid] @ fv); wsum += ef; nroutes += 1
        if eb > 0 and pid in body_tpl:
            num += eb * float(body_tpl[pid] @ bv); wsum += eb; nroutes += 1
        if eg > 0 and pid in gait_tpl:
            num += eg * float(gait_tpl[pid] @ gv); wsum += eg; nroutes += 1
        if wsum <= 0:
            continue
        sc = num / wsum
        if nroutes >= 2:
            sc = min(1.0, sc + agree_bonus)
        out[pid] = sc
    return out


def evaluate(probe_recs, cfg, fkey, tpls, weights, thresh):
    """开集评测。probe_recs: [(rec, gt_pid, is_genuine)]。"""
    face_tpl, body_tpl, gait_tpl = tpls
    subjects = sorted(set(face_tpl) | set(body_tpl) | set(gait_tpl))
    bins = defaultdict(lambda: {"correct": 0, "total": 0})
    imp = {"false_match": 0, "total": 0, "reject": 0}
    det_rows = []
    for rec, gt, genuine in probe_recs:
        scores = _fused_scores(rec, cfg, fkey, tpls, subjects, weights)
        if scores:
            pred = max(scores, key=scores.get); best = scores[pred]
        else:
            pred, best = None, -1.0
        hit = best >= thresh
        correct = hit and (pred == gt)
        det_rows.append((best, genuine, pred == gt))
        if genuine:
            qbin = rec["face_cfg"][fkey]["best_cat"]
            bins[qbin]["total"] += 1
            if correct:
                bins[qbin]["correct"] += 1
        else:
            imp["total"] += 1
            imp["false_match" if hit else "reject"] += 1
    return bins, imp, det_rows


def _rate(d):
    return round(100.0 * d["correct"] / d["total"], 1) if d and d["total"] else None


def det_curve(det_rows):
    """扫阈值出 (thresh, TPIR, FMR)。TPIR=真人命中且正确率；FMR=冒充者被接受率。"""
    gen = [(s, ok) for (s, g, ok) in det_rows if g]
    imp = [s for (s, g, ok) in det_rows if not g]
    ng, ni = len(gen), len(imp)
    out = []
    for t in np.linspace(0.0, 1.0, 51):
        tpir = (sum(1 for s, ok in gen if s >= t and ok) / ng) if ng else 0.0
        fmr = (sum(1 for s in imp if s >= t) / ni) if ni else 0.0
        out.append({"thresh": round(float(t), 3), "tpir": round(tpir, 4), "fmr": round(fmr, 4)})
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    print(
        "[!] 旧版 run_mevid.py 已停用：它把产品端到端门控与人脸模型对比混在同一协议中，"
        "会造成只有极少数人脸模板、融合阈值失配等问题。\n"
        "    端到端多模态实验请运行 run_mevid_e2e.py；\n"
        "    固定登记照的人脸模型实验请运行 run_mevid_face.py。"
    )
    return 2

    ap = argparse.ArgumentParser(description="MEVID 多模态消融（人脸+人形+步态）")
    ap.add_argument("--data", required=True, help="MEVID 根（含 bbox_test/ 与 annotation/）")
    ap.add_argument("--arms", default="M0,M1,M2,M3,Fonly,FB", help="逗号分隔")
    ap.add_argument("--enroll-subjects", type=int, default=10, help="建档身份数（真人）")
    ap.add_argument("--imposter-subjects", type=int, default=10, help="冒充者身份数（只查不建）")
    ap.add_argument("--faces-per-track", type=int, default=8, help="每轨迹人脸/人形抽帧数")
    ap.add_argument("--gait-frames", type=int, default=24, help="每轨迹步态抽帧数")
    ap.add_argument("--max-gallery-tracks", type=int, default=3, help="每身份最多入库轨迹数")
    ap.add_argument("--max-query-tracks", type=int, default=4, help="每身份最多 probe 轨迹数")
    ap.add_argument("--anchor", default="both", choices=["monitor", "checkin", "both"],
                    help="人脸建档锚点：monitor=监控高质帧 / checkin=登记照 / both=两种都跑")
    ap.add_argument("--thresh", type=float, default=None, help="命中阈值（默认用 IDENTITY_RESOLVE_THRESH）")
    ap.add_argument("--superres-regate", action="store_true", help="超分后在修复图上重评 category（M2/M3）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip() in ARMS]
    if not arms:
        print("[!] 没有合法 arm"); return 1

    from app import face as face_mod
    from app import body_reid as reid_mod
    from app import gait as gait_mod
    from app.core.config import settings

    if args.superres_regate:
        # 让产品在超分后重评 category（若 face.detect 支持该开关）
        try:
            settings.face_superres_regate = True
        except Exception:
            pass

    face_cfgs = {(ARMS[a]["backend"], ARMS[a]["superres"]) for a in arms}
    need_body = any(ARMS[a]["body"] for a in arms)
    need_gait = any(ARMS[a]["gait"] for a in arms)
    settings.face_superres = "gfpgan" if any(sr for (_, sr) in face_cfgs) else "off"

    data_dir = Path(args.data)
    print(f"[*] 加载 MEVID：{data_dir}")
    tracklets, _ = load_mevid(data_dir)
    print(f"[*] 轨迹数={len(tracklets)}  身份数={len(set(t.pid for t in tracklets))}")

    # 按身份聚合，分 gallery/query 轨迹
    by_pid = defaultdict(lambda: {"gallery": [], "query": []})
    for t in tracklets:
        by_pid[t.pid]["query" if t.is_query else "gallery"].append(t)
    # 只保留同时有 gallery 和 query 的身份，稳定排序
    valid_pids = sorted(p for p, d in by_pid.items() if d["gallery"] and d["query"])
    import random
    rnd = random.Random(args.seed)
    rnd.shuffle(valid_pids)
    enroll_pids = valid_pids[:args.enroll_subjects]
    imposter_pids = valid_pids[args.enroll_subjects:args.enroll_subjects + args.imposter_subjects]
    print(f"[*] 建档身份={len(enroll_pids)}  冒充者身份={len(imposter_pids)}")

    checkin = load_checkin_anchors(data_dir) if args.anchor in ("checkin", "both") else {}

    # 收集要提特征的轨迹：enroll 的 gallery + query；imposter 的 query
    def cap(lst, n):
        return lst[:n]

    ex_tracks = []  # (role, pid, tracklet)  role ∈ {enroll_gallery, genuine_query, imposter_query}
    for pid in enroll_pids:
        for t in cap(by_pid[pid]["gallery"], args.max_gallery_tracks):
            ex_tracks.append(("enroll_gallery", pid, t))
        for t in cap(by_pid[pid]["query"], args.max_query_tracks):
            ex_tracks.append(("genuine_query", pid, t))
    for pid in imposter_pids:
        for t in cap(by_pid[pid]["query"], args.max_query_tracks):
            ex_tracks.append(("imposter_query", pid, t))

    print(f"[*] 待提特征轨迹={len(ex_tracks)}（人脸配置={sorted(face_cfgs)} body={need_body} gait={need_gait}）")
    print(f"[*] 后端：face={face_mod.active_backend()} reid={reid_mod.active_backend() if need_body else '-'} "
          f"gait={'on' if need_gait else '-'}")

    # 提特征（共享 body/gait，只算一次；人脸按 cfg）
    t0 = time.time()
    feats = []  # (role, pid, rec)
    for i, (role, pid, tk) in enumerate(ex_tracks):
        rec = extract_track_features(tk, face_mod, reid_mod, gait_mod, settings,
                                     face_cfgs, need_body, need_gait,
                                     args.faces_per_track, args.gait_frames)
        feats.append((role, pid, rec))
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"    {i+1}/{len(ex_tracks)}  {el:.0f}s  {(i+1)/el:.2f} track/s", flush=True)
    print(f"[*] 提特征完成 {time.time()-t0:.0f}s")

    thresh = args.thresh if args.thresh is not None else settings.identity_resolve_thresh
    weights = (settings.identity_w_face, settings.identity_w_body, settings.identity_w_gait,
               settings.identity_face_quality_floor, settings.identity_agree_bonus)

    anchors = ["monitor", "checkin"] if args.anchor == "both" else [args.anchor]
    all_results = {}
    for anchor in anchors:
        results = {}
        for a in arms:
            cfg = ARMS[a]
            fkey = (cfg["backend"], cfg["superres"])
            # 建档记录
            enroll_by_pid = defaultdict(list)
            for role, pid, rec in feats:
                if role == "enroll_gallery":
                    enroll_by_pid[pid].append(rec)
            face_tpl, body_tpl, gait_tpl, cov = build_gallery(enroll_by_pid, cfg, fkey)
            # checkin 锚点：人脸模板改用登记照（body/gait 仍用监控 gallery）
            if anchor == "checkin":
                face_tpl2 = _checkin_face_templates(enroll_pids, checkin, face_mod, settings, cfg, fkey)
                cov = dict(cov); cov["face"] = len(face_tpl2)
                face_tpl = face_tpl2
            tpls = (face_tpl, body_tpl, gait_tpl)
            # probe
            probe_recs = []
            for role, pid, rec in feats:
                if role == "genuine_query":
                    probe_recs.append((rec, pid, True))
                elif role == "imposter_query":
                    probe_recs.append((rec, pid, False))
            bins, imp, det_rows = evaluate(probe_recs, cfg, fkey, tpls, weights, thresh)
            overall = {"correct": sum(b["correct"] for b in bins.values()),
                       "total": sum(b["total"] for b in bins.values())}
            results[a] = {
                "note": cfg["note"],
                "coverage": cov,
                "by_bin": {k: dict(v) for k, v in bins.items()},
                "overall": dict(overall),
                "imposter": imp,
                "fmr": round(100.0 * imp["false_match"] / imp["total"], 1) if imp["total"] else None,
                "reject_rate": round(100.0 * imp["reject"] / imp["total"], 1) if imp["total"] else None,
                "det": det_curve(det_rows),
            }
        all_results[anchor] = results
        _print_table(anchor, results, thresh)

    _save(all_results, args, enroll_pids, imposter_pids, thresh)
    return 0


def _checkin_face_templates(enroll_pids, checkin, face_mod, settings, cfg, fkey):
    """用登记照建人脸模板（仍走真门控：clear 才入库）。"""
    from PIL import Image
    tpl = {}
    for pid in enroll_pids:
        embs = []
        for p in checkin.get(pid, []):
            try:
                pil = Image.open(p).convert("RGB")
            except Exception:
                continue
            bf = _face_of(face_mod, settings, pil, cfg["backend"], cfg["superres"])
            if bf is None:
                continue
            qd = bf.get("quality") or {}
            if qd.get("category") == "clear" and bf.get("embedding") is not None:
                embs.append(_norm(bf["embedding"]))
        if embs:
            tpl[pid] = _norm(np.mean(embs, axis=0))
    return tpl


def _print_table(anchor, results, thresh):
    print(f"\n================ MEVID 多模态消融 · anchor={anchor} · thresh={thresh} ================")
    head = f"{'arm':<7}" + "".join(f"{b:>9}" for b in BIN_ORDER) + f"{'overall':>9}{'FMR':>7}{'rej':>7}{'覆盖(f/b/g)':>14}   note"
    print(head)
    for a, r in results.items():
        row = f"{a:<7}"
        for b in BIN_ORDER:
            row += f"{(_rate(r['by_bin'].get(b)) if r['by_bin'].get(b) else '-'):>9}"
        cov = r["coverage"]
        covs = f"{cov['face']}/{cov['body']}/{cov['gait']}"
        row += f"{_rate(r['overall']):>9}{str(r['fmr']):>7}{str(r['reject_rate']):>7}{covs:>14}   {r['note']}"
        print(row)
    print("（差脸桶 poor/none 看识别率↑；FMR=冒充者误识率↓；覆盖=各模态成功建档身份数/总）")


def _save(all_results, args, enroll_pids, imposter_pids, thresh):
    import datetime
    import json
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"mevid_{'-'.join(a for a in args.arms.split(','))}_e{len(enroll_pids)}_i{len(imposter_pids)}_{args.anchor}_{ts}"
    payload = {
        "dataset": "MEVID", "run_at": ts,
        "config": {"arms": args.arms, "enroll_subjects": len(enroll_pids),
                   "imposter_subjects": len(imposter_pids), "faces_per_track": args.faces_per_track,
                   "gait_frames": args.gait_frames, "anchor": args.anchor, "thresh": thresh,
                   "superres_regate": args.superres_regate},
        "results": all_results,
    }
    path = OUT_DIR / f"{stem}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[✓] 结果 → {path}")


if __name__ == "__main__":
    raise SystemExit(main())
