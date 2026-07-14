"""糊脸消融实验 · 端到端评测（Market-1501 + 人脸质量分桶 + arm 矩阵）。

实验目的（见 results/实验流程.svg）：
  在真实监控低清行人图上，先**只用人脸**按质量分桶（clear/blur/tiny/none），
  再在**差脸桶**上比各 arm 的「认人」准确率，证明每个手段救回多少：
    S0 baseline 纯 ArcFace 仅人脸 → S1 +AdaFace / S2 +超分（做强脸）
    → S5 +人形 ReID（脸糊兜底）→ full 全栈（天花板）

设计原则：
  · 重活全部 call 产品代码：人脸 = app.face.detect(with_quality)（检测+嵌入+质量门控），
    人形 = app.body_reid.embed；arm 之间只切产品开关（FACE_REC_BACKEND / 超分 enhance_blurry）。
  · 质量分桶只用人脸（arm 无关，跑一次）；人形只在 S5/full 内部才提。
  · gallery/probe 图像互斥（gallery 取走的图不进 probe），无数据泄漏；不做训练（模型预训练冻结）。
  · 打分（闭集 Rank-1）是评测胶水，产品里没有。

用法（数据在 git 仓库外）：
    python run_eval.py --data <Market-1501 根目录> --arms S0,S1,S2,S5,full
    python run_eval.py --data <dir> --max-subjects 20 --gallery-per-subject 3 --probe-per-subject 8
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

# 本文件在 experiment/糊脸消融实验/scripts/ 下。
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(ROOT))
OUT_DIR = EXPERIMENT_DIR / "results" / "legacy_market"

# arm 配置：人脸识别后端 / 是否超分 / 是否加人形
ARMS = {
    "S0":   {"backend": "arcface", "superres": False, "body": False, "note": "纯 ArcFace 仅人脸（baseline）"},
    "S1":   {"backend": "adaface", "superres": False, "body": False, "note": "+AdaFace 质量自适应识别"},
    "S2":   {"backend": "arcface", "superres": True,  "body": False, "note": "+超分 GFPGAN 糊脸预处理"},
    "S5":   {"backend": "arcface", "superres": False, "body": True,  "note": "+人形 ReID 兜底"},
    "full": {"backend": "adaface", "superres": True,  "body": True,  "note": "全栈：超分+AdaFace+人形"},
}
BIN_ORDER = ["clear", "marginal", "poor", "none"]


# --------------------------------------------------------------------------
# Market-1501 数据加载（文件名形如 0001_c1s1_001051_00.jpg）
# --------------------------------------------------------------------------
_NAME_RE = re.compile(r"^(?P<pid>-?\d+)_c(?P<cam>\d+)")


def _parse(p: Path):
    m = _NAME_RE.match(p.stem)
    if not m:
        return None, None
    pid = m.group("pid")
    if pid in ("-1", "0000"):  # 干扰/背景，丢
        return None, None
    return pid, int(m.group("cam"))


def load_market(data_dir: Path, split: str, max_subjects: int | None):
    """读 Market 某 split → {pid: [ (path, cam) ]}。"""
    folder = data_dir / split
    if not folder.is_dir():
        # 容忍多套一层
        cands = list(data_dir.glob(f"*/{split}"))
        folder = cands[0] if cands else folder
    by_pid: dict[str, list] = defaultdict(list)
    for p in sorted(folder.glob("*.jpg")):
        pid, cam = _parse(p)
        if pid is None:
            continue
        by_pid[pid].append((p, cam))
    pids = sorted(by_pid)
    if max_subjects:
        pids = pids[:max_subjects]
    return {pid: by_pid[pid] for pid in pids}


# --------------------------------------------------------------------------
# ChokePoint 数据加载（gt XML 逐帧标出 person id + 眼睛坐标）
# --------------------------------------------------------------------------
def _parse_chokepoint_xml(xml_path: Path):
    """解析 groundtruth/<seq>_<cam>.xml → [(frame_num_str, person_id_str)]。"""
    import xml.etree.ElementTree as ET

    out = []
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return out
    for fr in root.iter("frame"):
        num = fr.get("number")
        for p in fr.findall("person"):
            pid = p.get("id")
            if num and pid:
                out.append((num, pid))
    return out


def load_chokepoint(data_dir: Path, sequences: list[str], gallery_cam: int,
                    max_subjects: int | None):
    """读 ChokePoint 多个 sequence，返回 {global_pid: [(path, cam_int)]}。

    global_pid = "<seq>#<local_pid>"，避免不同序列 person id 撞车。
    gallery_cam = 视作 gallery 机位（其余机位当 probe）——记录在 cam 字段里，划分时用。
    sequences 支持 P1E_S1 / P2E_S1 及 P2 的 .1/.2 分段（会自动匹配 <seq>_C*.xml 和 <seq>.<n>_C*.xml）。
    """
    gt_dir = data_dir / "groundtruth"
    by_pid: dict[str, list] = defaultdict(list)

    # 枚举所有 <seq>_C*.xml；P2 的分段 xml 命名是 <seq>_C1.1.xml / <seq>_C1.2.xml
    for seq in sequences:
        for xml in sorted(gt_dir.glob(f"{seq}_C*.xml")):
            stem = xml.stem  # e.g. P2E_S1_C1.1  或  P1E_S1_C1
            # 解出 cam 编号（去掉 .1/.2 分段后缀）与帧目录名
            m = re.match(rf"^{re.escape(seq)}_C(\d)(?:\.(\d+))?$", stem)
            if not m:
                continue
            cam = int(m.group(1))
            frame_dir_name = stem  # 帧目录名与 xml stem 完全一致
            frame_dir = data_dir / frame_dir_name
            if not frame_dir.is_dir():
                continue
            for frame_num, local_pid in _parse_chokepoint_xml(xml):
                jpg = frame_dir / f"{frame_num}.jpg"
                if not jpg.exists():
                    continue
                global_pid = f"{seq}#{local_pid}"
                by_pid[global_pid].append((jpg, cam))

    pids = sorted(by_pid)
    if max_subjects:
        pids = pids[:max_subjects]
    return {pid: by_pid[pid] for pid in pids}


# --------------------------------------------------------------------------
# 产品 call：人脸（检测+质量+嵌入） / 人形
# --------------------------------------------------------------------------
def face_detect(face_mod, settings, pil, backend: str, superres: bool, with_quality: bool):
    """call 产品 face.detect。返回 (best_face dict 或 None)。按 arm 切识别后端 / 超分。"""
    settings.face_rec_backend = backend
    try:
        faces = face_mod.detect(pil, with_quality=with_quality, enhance_blurry=superres)
    except Exception:
        return None
    if not faces:
        return None
    return max(faces, key=lambda f: float(f.get("det_score", 0.0)))


def _norm(v):
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def quality_bin(best_face) -> str:
    """直接采用产品 assess_quality 的分级类别（唯一真源，对齐客户人脸过滤逻辑）。
    没检测到脸 → none；否则取 category ∈ {clear, marginal, poor}。"""
    if best_face is None:
        return "none"
    q = best_face.get("quality") or {}
    return q.get("category", "poor")


# --------------------------------------------------------------------------
# 提特征：质量分桶(一次) + 各 arm 人脸 + 人形(一次)
# --------------------------------------------------------------------------
def embed_split(images, face_mod, reid_mod, settings, face_cfgs, need_body, label):
    """对一批 (path,cam) 提：base 质量桶 + 每种人脸配置的 emb + (可选)人形 emb。

    face_cfgs: set[(backend, superres)] —— 选中 arm 需要的人脸配置集合。
    返回 list[dict(path, cam, bin, face_embs:{(b,sr):vec}, body)]。
    """
    from PIL import Image

    out = []
    t0 = time.time()
    total = len(images)
    print(f"    [{label}] 提特征 {total} 张（CPU 较慢，每 25 张报一次）", flush=True)
    for i, (path, cam) in enumerate(images):
        try:
            pil = Image.open(path).convert("RGB")
        except Exception:
            continue
        # base：arcface 无超分 → 定质量桶 + 连续质量分(0~1，供融合软加权，对齐产品)
        base = face_detect(face_mod, settings, pil, "arcface", False, with_quality=True)
        q_scalar = (base.get("quality") or {}).get("quality") if base else None
        rec = {"path": path, "cam": cam, "bin": quality_bin(base),
               "q_scalar": q_scalar, "face_embs": {}, "body": None}
        for (backend, sr) in face_cfgs:
            if backend == "arcface" and sr is False and base is not None:
                f = base
            else:
                f = face_detect(face_mod, settings, pil, backend, sr, with_quality=False)
            rec["face_embs"][(backend, sr)] = _norm(f["embedding"]) if (f and f.get("embedding") is not None) else None
        if need_body:
            try:
                rec["body"] = _norm(reid_mod.embed(pil))
            except Exception:
                rec["body"] = None
        out.append(rec)
        if (i + 1) % 25 == 0:
            el = time.time() - t0
            rate = (i + 1) / el if el > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"    [{label}] {i+1}/{total}  {rate:.2f} 张/s  已用 {el:.0f}s  预计还需 {eta:.0f}s", flush=True)
    return out


# --------------------------------------------------------------------------
# 模板 + 打分
# --------------------------------------------------------------------------
def build_templates(gallery_recs, cfg, by_subject):
    """每 subject 均值成 face/body 模板（按 arm 的人脸配置）。"""
    key = (cfg["backend"], cfg["superres"])
    face_tpl, body_tpl = {}, {}
    g_by_subj = defaultdict(list)
    for r, subj in zip(gallery_recs, by_subject):
        g_by_subj[subj].append(r)
    for subj, recs in g_by_subj.items():
        fe = [r["face_embs"].get(key) for r in recs if r["face_embs"].get(key) is not None]
        if fe:
            face_tpl[subj] = _norm(np.mean(fe, axis=0))
        if cfg["body"]:
            be = [r["body"] for r in recs if r["body"] is not None]
            if be:
                body_tpl[subj] = _norm(np.mean(be, axis=0))
    return face_tpl, body_tpl, key


def score_arm(probe_recs, probe_subj, cfg, face_tpl, body_tpl, key, weights):
    """闭集 Rank-1，按桶统计。融合复刻产品 multimodal_identity_fusion 的**软性连续加权**。"""
    wf, wb, blur_factor, floor = weights
    f_ids = list(face_tpl)
    b_ids = list(body_tpl)
    f_mat = np.stack([face_tpl[s] for s in f_ids]) if f_ids else None
    b_mat = np.stack([body_tpl[s] for s in b_ids]) if b_ids else None
    subjects = sorted(set(f_ids) | set(b_ids))

    bins = defaultdict(lambda: {"correct": 0, "total": 0})
    for r, gt in zip(probe_recs, probe_subj):
        qbin = r["bin"]
        fv = r["face_embs"].get(key)
        bv = r["body"] if cfg["body"] else None
        has_face = fv is not None
        has_body = bv is not None

        pred = None
        if not cfg["body"]:
            if has_face and f_mat is not None:
                pred = f_ids[int(np.argmax(f_mat @ fv))]
        else:
            # 人脸有效权重：软性连续加权 wf×(floor+(1-floor)×质量分)；无质量分回退两档
            if has_face:
                qs = r.get("q_scalar")
                if qs is not None:
                    ef = wf * (floor + (1.0 - floor) * max(0.0, min(1.0, float(qs))))
                else:
                    ef = wf * (1.0 if qbin == "clear" else blur_factor)
            else:
                ef = 0.0
            eb = wb if has_body else 0.0
            if ef + eb > 0:
                best, best_s = None, -1e9
                for s in subjects:
                    sc = 0.0
                    if ef > 0 and s in face_tpl:
                        sc += ef * float(f_mat[f_ids.index(s)] @ fv)
                    if eb > 0 and s in body_tpl:
                        sc += eb * float(b_mat[b_ids.index(s)] @ bv)
                    sc /= (ef + eb)
                    if sc > best_s:
                        best, best_s = s, sc
                pred = best

        bins[qbin]["total"] += 1
        if pred == gt:
            bins[qbin]["correct"] += 1
    return bins


def _rate(d):
    return round(100.0 * d["correct"] / d["total"], 1) if d and d["total"] else None


def _rank_by_clarity(face_mod, settings, items):
    """把一组 (path, cam) 按人脸清晰度（det_score×100 + 人脸边长）从清到糊排序。
    gallery/干扰身份都用它选"最清晰 N 张"入库。"""
    from PIL import Image

    scored = []
    for (path, cam) in items:
        try:
            pil = Image.open(path).convert("RGB")
        except Exception:
            continue
        bf = face_detect(face_mod, settings, pil, "arcface", False, with_quality=True)
        side = (min(bf["bbox"][2] - bf["bbox"][0], bf["bbox"][3] - bf["bbox"][1]) if bf else 0.0)
        det = float(bf.get("det_score", 0.0)) if bf else 0.0
        scored.append(((path, cam), det * 100 + side))
    scored.sort(key=lambda x: -x[1])
    return [it for it, _ in scored]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="糊脸消融实验（Market-1501 / ChokePoint，质量分桶 + arm 矩阵）")
    ap.add_argument("--dataset", default="market", choices=["market", "chokepoint"],
                    help="数据集：market（行人ReID，含 tiny/none 桶）/ chokepoint（门廊监控，含 clear→poor 梯度）")
    ap.add_argument("--data", required=True, help="数据集根目录")
    ap.add_argument("--arms", default="S0,S1,S2,S5,full", help="逗号分隔：S0,S1,S2,S5,full")
    # Market 参数
    ap.add_argument("--split", default="bounding_box_test", help="Market 用哪个 split 取图")
    ap.add_argument("--max-subjects", type=int, default=25, help="评测前 N 个身份（控时长）")
    ap.add_argument("--gallery-per-subject", type=int, default=3, help="每人入库（最清晰）张数")
    ap.add_argument("--probe-per-subject", type=int, default=8, help="每人探针张数上限")
    # ChokePoint 参数
    ap.add_argument("--sequences", default="P1E_S1,P2E_S1",
                    help="ChokePoint 序列列表（逗号分隔），如 P1E_S1,P2E_S1")
    ap.add_argument("--gallery-cam", type=int, default=1,
                    help="ChokePoint 视作 gallery 的机位（1=C1 最正脸；其余机位当 probe）")
    ap.add_argument("--out-suffix", default="", help="输出文件名后缀（避免覆盖，如 chokepoint_p1e_p2e）")
    ap.add_argument("--distractors", type=int, default=0,
                    help="干扰身份数：额外 N 个只入库(gallery)不作 probe 的身份，扩大候选集 → 开集更难更真实")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip() in ARMS]
    if not arms:
        print("[!] 没有合法 arm")
        return 1
    face_cfgs = {(ARMS[a]["backend"], ARMS[a]["superres"]) for a in arms}
    need_body = any(ARMS[a]["body"] for a in arms)

    from app import face as face_mod
    from app import body_reid as reid_mod
    from app.core.config import settings

    data_dir = Path(args.data)
    from PIL import Image
    g_imgs, g_subj, p_imgs, p_subj = [], [], [], []

    if args.dataset == "market":
        print(f"[*] Market-1501: {data_dir}  split={args.split}  arms={arms}  distractors={args.distractors}")
        total = (args.max_subjects + args.distractors) if args.max_subjects else None
        by_pid = load_market(data_dir, args.split, total)
        if not by_pid:
            print("[!] 没读到图：确认 --data 指向含 bounding_box_test 的目录。")
            return 1
        all_pids = list(by_pid)
        eval_pids = all_pids[:args.max_subjects] if args.max_subjects else all_pids
        distractor_pids = all_pids[args.max_subjects:] if args.max_subjects else []
        # gallery/probe 划分：每人按「人脸清晰度」选最清晰几张入库，其余当探针（图像互斥）
        print(f"[*] 评测身份={len(eval_pids)}  干扰身份={len(distractor_pids)}（仅入库不作 probe）；按人脸清晰度划分 …")
        for pid in eval_pids:
            ranked = _rank_by_clarity(face_mod, settings, by_pid[pid])
            gsel = ranked[:args.gallery_per_subject]
            psel = ranked[args.gallery_per_subject:args.gallery_per_subject + args.probe_per_subject]
            for it in gsel:
                g_imgs.append(it); g_subj.append(pid)
            for it in psel:
                p_imgs.append(it); p_subj.append(pid)
        for pid in distractor_pids:  # 干扰身份：只取最清晰 N 张入库，不进 probe
            for it in _rank_by_clarity(face_mod, settings, by_pid[pid])[:args.gallery_per_subject]:
                g_imgs.append(it); g_subj.append(pid)
    else:  # chokepoint
        seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]
        print(f"[*] ChokePoint: {data_dir}  sequences={seqs}  gallery_cam=C{args.gallery_cam}  arms={arms}  distractors={args.distractors}")
        total = (args.max_subjects + args.distractors) if args.max_subjects else None
        by_pid = load_chokepoint(data_dir, seqs, args.gallery_cam, total)
        if not by_pid:
            print("[!] 没读到图：确认 --data 指向含 groundtruth/ 与序列帧目录的 chokepoint 根。")
            return 1
        all_pids = list(by_pid)
        eval_pids = all_pids[:args.max_subjects] if args.max_subjects else all_pids
        distractor_pids = all_pids[args.max_subjects:] if args.max_subjects else []
        # 跨机位划分：gallery = 每人 C1 里最清晰 N 张；probe = 该人在其他机位的全部帧（cap 上限）
        print(f"[*] 评测身份={len(eval_pids)}  干扰身份={len(distractor_pids)}；C{args.gallery_cam} 入库，其余机位当 probe …")

        def _ck_gallery(items):
            gal_pool = [it for it in items if it[1] == args.gallery_cam]
            return _rank_by_clarity(face_mod, settings, gal_pool)[:args.gallery_per_subject]

        for pid in eval_pids:
            items = by_pid[pid]
            gal_pool = [it for it in items if it[1] == args.gallery_cam]
            probe_pool = [it for it in items if it[1] != args.gallery_cam]
            if not gal_pool or not probe_pool:
                continue
            gsel = _ck_gallery(items)
            # probe: 其他机位均匀抽样，避免同一 track 相邻帧堆到 probe（对模型太容易）
            if len(probe_pool) > args.probe_per_subject:
                step = len(probe_pool) / args.probe_per_subject
                psel = [probe_pool[int(i * step)] for i in range(args.probe_per_subject)]
            else:
                psel = probe_pool
            for it in gsel:
                g_imgs.append(it); g_subj.append(pid)
            for it in psel:
                p_imgs.append(it); p_subj.append(pid)
        for pid in distractor_pids:  # 干扰身份：只入库（gallery_cam 最清晰 N 张），不进 probe
            for it in _ck_gallery(by_pid[pid]):
                g_imgs.append(it); g_subj.append(pid)

    print(f"[*] gallery={len(g_imgs)} 张  probe={len(p_imgs)} 张")

    print(f"[*] 人脸后端就绪={face_mod.active_backend()}  人形后端={reid_mod.active_backend()}")
    if ("adaface", False) in face_cfgs or ("adaface", True) in face_cfgs:
        err = face_mod.adaface_error()
        print(f"[*] AdaFace 状态：{'可用' if err is None else ('不可用→回退 ArcFace：' + str(err))}")

    # 超分：仅当有 arm 需要时启用真实后端（否则 enhance() 会因 face_superres=off 空转，超分白跑）
    need_sr = any((sr for (_, sr) in face_cfgs))
    settings.face_superres = "gfpgan" if need_sr else "off"
    if need_sr:
        print(f"[*] 超分后端：{settings.face_superres}（S2/full 用；仅对糊+非极端侧脸的脸触发）")

    g_recs = embed_split(g_imgs, face_mod, reid_mod, settings, face_cfgs, need_body, "gallery")
    p_recs = embed_split(p_imgs, face_mod, reid_mod, settings, face_cfgs, need_body, "probe")

    # 探针桶分布
    bin_dist = defaultdict(int)
    for r in p_recs:
        bin_dist[r["bin"]] += 1
    print(f"[*] probe 质量桶分布：" + "  ".join(f"{b}={bin_dist.get(b,0)}" for b in BIN_ORDER))

    weights = (settings.identity_w_face, settings.identity_w_body,
               settings.identity_face_blurry_factor, settings.identity_face_quality_floor)
    results = {}
    for a in arms:
        cfg = ARMS[a]
        face_tpl, body_tpl, key = build_templates(g_recs, cfg, g_subj)
        bins = score_arm(p_recs, p_subj, cfg, face_tpl, body_tpl, key, weights)
        overall = {"correct": sum(b["correct"] for b in bins.values()),
                   "total": sum(b["total"] for b in bins.values())}
        results[a] = {"note": cfg["note"], "by_bin": {k: dict(v) for k, v in bins.items()},
                      "overall": dict(overall)}

    _print_table(results)
    import datetime
    runs_dir = OUT_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ds = "market" if args.dataset == "market" else "chokepoint"
    parts = [ds, "-".join(arms), f"s{args.max_subjects}"]
    if args.distractors:
        parts.append(f"d{args.distractors}")
    if args.out_suffix:
        parts.append(args.out_suffix)
    parts.append(ts)
    stem = "_".join(parts)  # 唯一命名（含数据集/arm/身份数/干扰数/时间戳），不覆盖历史
    _save_json(results, args, dict(bin_dist), runs_dir / f"{stem}.json", ts,
               n_gallery=len(g_imgs), n_probe=len(p_imgs),
               n_eval=len(set(p_subj)), n_distractor=args.distractors)
    _plot(results, runs_dir / f"{stem}.svg")
    print(f"\n[✓] 运行名 → {stem}")
    print(f"[✓] 结果 → {runs_dir / f'{stem}.json'}")
    print(f"[✓] 消融图 → {runs_dir / f'{stem}.svg'}（同名 .png）")
    return 0


def _print_table(results):
    print("\n================ 糊脸消融 · 各 arm 分桶 Rank-1（%）================")
    head = f"{'arm':<6}" + "".join(f"{b:>9}" for b in BIN_ORDER) + f"{'overall':>9}   note"
    print(head)
    for a, r in results.items():
        row = f"{a:<6}"
        for b in BIN_ORDER:
            row += f"{(_rate(r['by_bin'].get(b)) if r['by_bin'].get(b) else '-'):>9}"
        row += f"{_rate(r['overall']):>9}   {r['note']}"
        print(row)
    print("（差脸桶 = blur+tiny+none；看各 arm 相对 S0 在差脸桶救回多少）")


def _save_json(results, args, bin_dist, path, run_at=None, n_gallery=0, n_probe=0,
               n_eval=0, n_distractor=0):
    import json

    payload = {
        "dataset": "Market-1501" if args.dataset == "market" else "ChokePoint",
        "run_at": run_at,
        "config": {
            "arms": list(results.keys()),
            "gallery_per_subject": args.gallery_per_subject,
            "probe_per_subject": args.probe_per_subject,
            "max_subjects": args.max_subjects,
            "distractors": args.distractors,
            **({"split": args.split} if args.dataset == "market"
               else {"sequences": args.sequences, "gallery_cam": f"C{args.gallery_cam}"}),
        },
        "counts": {"n_eval_subjects": n_eval, "n_distractor_subjects": n_distractor,
                   "n_gallery_imgs": n_gallery, "n_probe_imgs": n_probe},
        "probe_bin_dist": bin_dist,
        "arms": {a: {"note": r["note"],
                     "by_bin": {k: {**v, "rank1": _rate(v)} for k, v in r["by_bin"].items()},
                     "overall": {**r["overall"], "rank1": _rate(r["overall"])}}
                 for a, r in results.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _plot(results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
        plt.rcParams["axes.unicode_minus"] = False
        cats = BIN_ORDER + ["overall"]
        arms = list(results.keys())
        x = np.arange(len(cats))
        w = 0.8 / max(1, len(arms))
        palette = ["#9AA0A6", "#F7630C", "#FFB900", "#5C2D91", "#107C10", "#0078D4"]
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, a in enumerate(arms):
            vals = []
            for c in cats:
                d = results[a]["overall"] if c == "overall" else results[a]["by_bin"].get(c)
                vals.append((_rate(d) or 0) if d else 0)
            ax.bar(x + i * w, vals, w, label=a, color=palette[i % len(palette)])
        ax.set_xticks(x + w * (len(arms) - 1) / 2)
        ax.set_xticklabels(["清晰", "糊脸", "极糊", "无脸", "总体"])
        ax.set_ylabel("Rank-1 准确率 (%)")
        ax.set_ylim(0, 105)
        ax.set_title("糊脸消融 · 各 arm 按人脸质量分桶（Market-1501）")
        ax.legend(ncol=len(arms))
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, facecolor="white")
        plt.savefig(str(path).replace(".svg", ".png"), dpi=130, facecolor="white")
    except Exception as exc:  # noqa: BLE001
        print(f"    （绘图跳过：{exc}）")


if __name__ == "__main__":
    raise SystemExit(main())
