"""身份感知·多帧事件理解 端到端编排（Phase 4 · Step 24）—— 把叶子串成一条链路。

定位：这是 Phase 4 的**集成步**。前面写好的叶子模块都是"只做一件事"的纯函数式组件：
  - `video_processor.extract_frames`（选帧① 定时密采样）
  - `tracker.track_objects`（YOLO + ByteTrack，给每个人稳定 track_id）
  - `reid` + `gallery`（人形指纹 + 主体记忆库：认过一次就记住 → 身份）
  - `face`（可选：人脸指纹，清晰正脸时的最强身份信号）
  - `keyframe.select_keyframes`（选帧② 事件驱动地把几百帧砍到几十帧）
  - `identity_context.format_identity_context`（把身份打包成 LLM 能读的文本）
  - `event_understanding.understand_event`（多帧 + 身份 → 跨帧事件叙述，本阶段灵魂）

本模块负责**它们之间的编排**——尤其是"**流式开/关窗**"：把一段视频流按"活动段"切成若干
**事件窗**，每个窗各自选关键帧、打包身份、调一次多模态 LLM，最终汇成一条**事件时间线**。

⚠️ 与设计一致的边界：
  - "事件"由**语义信号**定义（新轨迹进入 / 轨迹离开 / 计数变化 / 身份命中），**不是**像素/
    ffmpeg 场景突变。本模块在跟踪结果之上**算这些语义事件**，再交给 keyframe 选帧。
  - 身份由传统 CV（人脸 / 人形 ReID + 库比对）**外部给定**；LLM 看图理解"做了什么"，不重新认人。

本地 Demo 把"视频文件逐帧处理"当作"流"来跑（按时序喂同一 tracker/gallery 会话）。上云后把
帧来源换成实时拉流即可，编排逻辑不变。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import face as face_mod
from . import gait as gait_mod
from . import gallery as gallery_mod
from . import reid as reid_mod
from . import tracker as tracker_mod
from .core.config import settings
from .keyframe import FrameMeta, select_keyframes
from .services.event_understanding import summarize_event_windows, understand_event
from .services.identity_context import format_identity_context
from .services.identity_fusion import fuse_identity
from .utils.image_utils import image_to_data_uri, seconds_to_timestamp
from .video_processor import Frame, extract_frames


# ---------------- 小工具 ----------------
def _signature(img: Image.Image, size: int = 16) -> np.ndarray:
    """整帧灰度缩略指纹（size×size），仅供 keyframe 去重用（不定义事件）。"""
    g = img.convert("L").resize((size, size))
    return np.asarray(g, dtype=np.float32).reshape(-1)


def _ts_seconds(frame: Frame, idx: int, step: float) -> float:
    """帧的秒级时间（优先用 Frame.timestamp，回退 idx*step）。"""
    return round(idx * step, 3)


def _crop(img: Image.Image, box: list[float]) -> Image.Image | None:
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return img.crop((x1, y1, x2, y2))


def _iou_xyxy(a, b) -> float:
    """两个 [x1,y1,x2,y2] 框的 IoU（步态把 YOLO-Pose 的人关联到 track 的 person box）。"""
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


# ---------------- 主编排 ----------------
def analyze_event_stream(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    fps: float = 2.0,
    max_frames: int = 300,
    session_id: str = "event-demo",
    run_llm: bool = True,
    with_face: bool = False,
    with_gait: bool = False,
    objective: str | None = None,
    quiet_seconds: float = 2.0,
    max_keyframes: int | None = None,
    include_keyframe_images: bool = False,
    max_window_seconds: float | None = None,
    stitch_thresh: float | None = None,
    overall_summary: bool | None = None,
) -> dict:
    """对一段视频做"身份感知·多帧事件理解"的完整端到端处理。

    Args:
        video_path: 输入视频。
        out_dir: 输出目录（帧放在 out_dir/frames）。
        fps: 选帧① 定时采样密度（每秒几帧）。
        max_frames: 抽帧硬上限。
        session_id: 跟踪/记忆会话标识（隔离 tracker & gallery）。
        run_llm: True 真调多模态 LLM 做事件理解；False 只跑到 LLM 边界（dry-run，不花额度）。
        with_face: 是否启用人脸分支（InsightFace，较慢；此夜间蒙面片多半无脸，默认关）。
        with_gait: 是否启用步态分支（SkeletonGait++，CPU 较慢；需 OpenGait+权重就绪）。无脸/背身时兜底。
        objective: 可选关注点，写进事件理解 prompt（如"留意陌生人/包裹被取走"）。
        quiet_seconds: 连续无人多久算"活动结束"，用于切分事件窗。
        max_keyframes: 喂 LLM 的关键帧上限（覆盖 settings.keyframe_max）。
        include_keyframe_images: 是否在每个窗里附带关键帧的 data URI（前端展示用；脚本默认关）。
        max_window_seconds: 单个事件窗的时长上限（秒）；超过则冲刷开新窗，避免长连续事件欠采样。
            默认取 settings.event_window_max_seconds。
        stitch_thresh: 同视频内"轨迹缝合"的余弦阈值；灰区孤立 track 与某主体相似度 ≥ 此值即并入。
            默认取 settings.event_stitch_thresh；设 0 关闭缝合。
        overall_summary: 是否在所有窗理解完后做一次"跨窗整段事件总结"（纯文本、便宜）。
            默认取 settings.event_overall_summary；dry-run（run_llm=False）下不做。

    Returns:
        dict：含 tracks（每条轨迹的身份裁决）、windows（每个事件窗的关键帧/身份/事件叙述）。
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    t_start = time.time()

    # ---- 选帧①：定时密采样（当作"流"的来源）----
    step = 1.0 / float(fps)
    frames = extract_frames(video_path, frames_dir, max_frames=max_frames, fps=fps)

    # ---- 换视频：清掉旧的跟踪/记忆状态，保证干净的流 ----
    tracker_mod.reset_tracker(session_id)
    gallery_mod.reset_gallery(session_id)
    dim = reid_mod.embed_dim()

    # 逐帧累积的语义元数据 + 每条 track 的最佳 crop（给认人用）
    metas: list[FrameMeta] = []
    quiet_frames = max(1, int(round(quiet_seconds * fps)))
    img_w = img_h = 0

    # track_id -> {"first": i, "last": i, "best_q": q, "best_crop": PIL, "best_idx": i,
    #              "best_box": box, "centers": [(i,(cx,cy))], "boxes": {i: box}}
    tracks: dict[int, dict] = {}
    prev_person_count = 0
    # 步态可用性（OpenGait+权重就绪才采集，避免每帧白跑 pose/seg）
    gait_use = bool(with_gait and settings.gait_enabled and gait_mod.available())
    gait_collect_error = None

    for i, fr in enumerate(frames):
        pil = Image.open(fr.local_path).convert("RGB")
        if not img_w:
            img_w, img_h = pil.size
        raw = Path(fr.local_path).read_bytes()
        res = tracker_mod.track_objects(raw, session_id=session_id)

        persons = [d for d in res["detections"] if d.get("label") == "person" and d.get("track_id") is not None]
        active = [int(d["track_id"]) for d in persons]
        events: list[str] = []
        track_quality: dict[int, float] = {}

        for d in persons:
            tid = int(d["track_id"])
            box = d["box"]
            crop = _crop(pil, box)
            q = 0.0
            if crop is not None:
                qa = reid_mod.assess_quality(crop)
                # 帧内 track 清晰度（给 keyframe 选最佳帧）：清晰度 × 面积饱和
                q = float(qa["blur_var"]) * min(1.0, qa["area"] / 20000.0)
            track_quality[tid] = q

            t = tracks.get(tid)
            if t is None:
                t = {"first": i, "last": i, "best_q": -1.0, "best_crop": None,
                     "best_idx": i, "best_box": box, "centers": [], "boxes": {}}
                tracks[tid] = t
                events.append(f"new_track:{tid}")
            t["last"] = i
            t["boxes"][i] = box
            cx = (box[0] + box[2]) / 2.0 / max(1, img_w)
            cy = (box[1] + box[3]) / 2.0 / max(1, img_h)
            t["centers"].append((i, (round(cx, 3), round(cy, 3))))
            if crop is not None and q > t["best_q"]:
                t["best_q"], t["best_crop"], t["best_idx"], t["best_box"] = q, crop, i, box

        # 计数变化 = 语义事件
        if len(active) != prev_person_count:
            events.append("count_change")
        prev_person_count = len(active)

        # ---- 步态：本帧跑一次 YOLO-Pose+Seg，按 box 关联到 track，累积姿态/剪影序列 ----
        if with_gait and gait_use and persons:
            try:
                bgr = np.asarray(pil)[:, :, ::-1]  # PIL RGB → BGR
                gp = gait_mod.extract_persons(bgr)
                for d in persons:
                    tid = int(d["track_id"])
                    pb = d["box"]
                    best, best_iou = None, 0.30  # 至少 0.3 IoU 才认为是同一人
                    for gpitem in gp:
                        iou = _iou_xyxy(pb, gpitem["box"])
                        if iou > best_iou:
                            best_iou, best = iou, gpitem
                    if best is not None:
                        t = tracks[tid]
                        t.setdefault("pose_seq", []).append(best["kpts"])
                        t.setdefault("sil_seq", []).append(best["mask"])
            except Exception as exc:  # 步态采集失败不致命
                gait_collect_error = str(exc)

        metas.append(FrameMeta(
            index=i,
            timestamp=_ts_seconds(fr, i, step),
            signature=_signature(pil),
            active_tracks=active,
            track_quality=track_quality,
            events=events,
        ))

    # ---- 轨迹离开：在每条 track 的 last 帧补 track_left 事件 ----
    for tid, t in tracks.items():
        li = t["last"]
        if "track_left" not in "".join(metas[li].events):
            metas[li].events.append(f"track_left:{tid}")

    # ---- 认人：每条 track 用最佳 crop 提指纹、查/登记主体记忆库 → 身份 ----
    identities: dict[int, dict] = {}
    track_emb: dict[int, np.ndarray] = {}   # track_id -> ReID 向量（缝合用）
    for tid, t in tracks.items():
        ident = {"track_id": tid, "subject_id": None, "decision": None,
                 "score": None, "reused": False, "face": None}
        crop = t["best_crop"]
        if crop is not None:
            try:
                vec = reid_mod.embed(crop)
                track_emb[tid] = np.asarray(vec, dtype=np.float32).reshape(-1)
                qa = reid_mod.assess_quality(crop)
                res = gallery_mod.with_gallery_locked(
                    session_id, dim,
                    lambda g: g.identify_or_enroll(vec, qa, auto_enroll=True),
                )
                ident["subject_id"] = res.get("subject_id")
                ident["decision"] = res.get("decision")
                ident["score"] = res.get("score")
                ident["reused"] = res.get("decision") == "hit"
            except Exception as exc:  # 认人失败不致命：身份留空，仍可做事件理解
                ident["error"] = str(exc)
        identities[tid] = ident

    # ---- 同视频内"轨迹缝合"：把灰区孤立 track 并进最相近的已建主体 ----
    # 动机：gallery 阈值是为"跨摄像头开放集"的安全设的；同一段视频里 ByteTrack 把一个连续的人
    # 断成几段，先验很强（同场景、时间连续），可更大胆地合并。只在编排层做，不动 gallery 语义。
    thr = settings.event_stitch_thresh if stitch_thresh is None else stitch_thresh
    if thr and thr > 0:
        _stitch_orphans(tracks, identities, track_emb, thr)

    # 身份命中 → 在该 track 首帧补 identity_hit 语义事件
    for tid, ident in identities.items():
        if ident.get("reused") and ident.get("subject_id") is not None:
            fi = tracks[tid]["first"]
            metas[fi].events.append(f"identity_hit:主体#{ident['subject_id']}")

    # ---- 可选人脸分支：仅在每条 track 的最佳帧上稀疏跑（攻"人脸模糊"的同时不拖慢全片）----
    if with_face:
        _attach_faces(frames, tracks, identities, session_id)

    # ---- 步态认人：每条 track 用累积的(姿态+剪影)序列提步态向量 → 步态库 → 写 gait_cue ----
    gait_dim = None
    if gait_use:
        gait_sess = f"{session_id}-gait"
        gallery_mod.reset_gallery(gait_sess)
        for tid, t in tracks.items():
            pose_seq = t.get("pose_seq") or []
            sil_seq = t.get("sil_seq") or []
            if len(pose_seq) < settings.gait_min_frames:
                continue
            try:
                gvec = gait_mod.embed_track(pose_seq, sil_seq)
                if gvec is None:
                    continue
                gvec = np.asarray(gvec, dtype=np.float32).reshape(-1)
                if gait_dim is None:
                    gait_dim = int(gvec.shape[0])
                gres = gallery_mod.with_gallery_locked(
                    gait_sess, gait_dim,
                    lambda g: g.identify_or_enroll(gvec, None, auto_enroll=True),
                )
                identities[tid]["gait"] = {
                    "score": gres.get("score"),
                    "subject_id": gres.get("subject_id"),
                    "decision": gres.get("decision"),
                    "frames": len(pose_seq),
                }
            except Exception as exc:  # 步态认人失败不致命
                identities[tid].setdefault("gait_error", str(exc))

    # ---- 跨 track 三路合并：人脸库/人形库/步态库 任一路认出同一人 → 并成一个 subject ----
    if identities:
        _merge_tracks_cross_route(identities)

    # ---- A 汇聚：三路身份融合（人脸 + 人形 + 步态 按质量加权 → 统一身份置信度）----
    for tid in identities:
        fuse_identity(identities[tid])

    # ---- 流式分窗：把帧序列按"活动段 + 时长上限"切成事件窗 ----
    win_secs = settings.event_window_max_seconds if max_window_seconds is None else max_window_seconds
    max_window_frames = int(round(win_secs * fps)) if win_secs and win_secs > 0 else None
    windows = _split_windows(metas, quiet_frames, max_window_frames)
    if not windows:  # 全程无人/无活动 → 整段当一个窗，LLM 仍可描述场景
        windows = [list(range(len(metas)))]

    idx2frame = {i: fr for i, fr in enumerate(frames)}
    out_windows: list[dict] = []
    for w, win_idx in enumerate(windows):
        win_metas = [metas[i] for i in win_idx]
        sel = select_keyframes(win_metas, max_frames=max_keyframes)
        if not sel:
            sel = win_idx[: (max_keyframes or settings.keyframe_max)]

        # 该窗涉及的 track → 按 subject 合并后打包身份（避免同一人被列成多条、误导 LLM 计数）
        win_tracks = sorted({t for i in win_idx for t in metas[i].active_tracks})
        people = _group_people(win_tracks, tracks, identities, win_idx, img_w, img_h)
        identity_text = format_identity_context(people, img_w, img_h)

        kf = [{"image": idx2frame[i].local_path, "timestamp": idx2frame[i].timestamp} for i in sel]
        ts_range = [metas[win_idx[0]].timestamp, metas[win_idx[-1]].timestamp]

        window_out = {
            "window_index": w,
            "time_range": [seconds_to_timestamp(ts_range[0]), seconds_to_timestamp(ts_range[1])],
            "frame_count": len(win_idx),
            "keyframe_indices": sel,
            "keyframe_timestamps": [idx2frame[i].timestamp for i in sel],
            "events": sorted({e for i in win_idx for e in metas[i].events}),
            "people": people,
            "identity_context": identity_text,
        }
        if include_keyframe_images:
            window_out["keyframes"] = [
                {"timestamp": idx2frame[i].timestamp, "image": image_to_data_uri(idx2frame[i].local_path)}
                for i in sel
            ]
        if run_llm:
            window_out["event"] = understand_event(kf, identity_text, objective=objective)
        out_windows.append(window_out)

    # ---- 跨窗整段事件总结：所有窗理解完后，纯文本把多窗串成整段连贯故事（便宜；dry-run 跳过）----
    overall = None
    do_overall = settings.event_overall_summary if overall_summary is None else overall_summary
    if run_llm and do_overall and out_windows:
        try:
            overall = summarize_event_windows(out_windows) or None
        except Exception as exc:  # 总结失败不致命：逐窗结果仍在
            overall = {"error": str(exc)}

    return {
        "video": str(video_path),
        "fps": fps,
        "frames_total": len(frames),
        "img_size": [img_w, img_h],
        "session_id": session_id,
        "reid_backend": reid_mod.active_backend(),
        "reid_dim": dim,
        "with_face": with_face,
        "with_gait": gait_use,
        "gait_error": (gait_mod.load_error() if (with_gait and not gait_use) else gait_collect_error),
        "model": settings.event_llm_deployment or settings.azure_openai_deployment,
        "dry_run": not run_llm,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "tracks": {str(tid): identities[tid] for tid in identities},
        "windows": out_windows,
        "overall": overall,
    }


def _attach_faces(
    frames: list[Frame], tracks: dict[int, dict], identities: dict[int, dict],
    session_id: str,
) -> None:
    """对每条 track 的最佳帧跑一次人脸检测，关联到该 track；提 512 维人脸指纹查/建**人脸库**。

    人脸库(face gallery)和人形/步态同套路：清晰正脸入库且高置信命中 → 直接定身份；糊脸/侧脸
    质量不过关则**不入库**(避免污染)但仍记录存在、降权(身份退人形/步态)。这就是"越清晰越能拍板"。
    """
    face_sess = f"{session_id}-face"
    gallery_mod.reset_gallery(face_sess)
    by_frame: dict[int, list[int]] = {}
    for tid, t in tracks.items():
        by_frame.setdefault(t["best_idx"], []).append(tid)
    for fidx, tids in sorted(by_frame.items()):
        try:
            pil = Image.open(frames[fidx].local_path).convert("RGB")
            faces = face_mod.detect(pil)
        except Exception:
            continue
        person_dets = [{"box": tracks[tid]["best_box"], "track_id": tid, "label": "person"} for tid in tids]
        assoc = face_mod.associate_to_persons(faces, person_dets)
        for tid, fc in assoc.items():
            q = fc.get("quality", {}) or {}
            clear = bool(q.get("quality_ok"))
            rec = {
                "score": q.get("det_score"),
                "quality": "clear" if clear else "blurry",
                "matched": False,
                "face_subject_id": None,
                "match_score": None,
            }
            emb = fc.get("embedding")
            if emb is not None:
                try:
                    fvec = np.asarray(emb, dtype=np.float32).reshape(-1)
                    # 清晰脸才允许建档入库（auto_enroll）；糊脸只查不建（不污染人脸库）
                    fres = gallery_mod.with_gallery_locked(
                        face_sess, face_mod.FACE_DIM,
                        lambda g: g.identify_or_enroll(
                            fvec, None, auto_enroll=clear,
                            hit_thresh=settings.face_hit_thresh,
                            new_thresh=settings.face_new_thresh,
                        ),
                    )
                    rec["face_subject_id"] = fres.get("subject_id")
                    rec["match_score"] = fres.get("score")
                    rec["matched"] = fres.get("decision") == "hit"
                except Exception as exc:  # 人脸库比对失败不致命
                    rec["face_error"] = str(exc)
            identities[tid]["face"] = rec


def _person_record(tid: int, t: dict, ident: dict, win_idx: list[int], img_w: int, img_h: int) -> dict:
    """把一条 track 在某窗内的信息打包成 identity_context 能吃的 person dict。"""
    centers = [c for (i, c) in t["centers"] if i in set(win_idx)]
    box = t["boxes"].get(win_idx[-1]) or t.get("best_box") or []
    return {
        "track_id": tid,
        "box": box,
        "subject_id": ident.get("subject_id"),
        "decision": ident.get("decision"),
        "reused": ident.get("reused", False),
        "trajectory": [list(c) for c in centers],
        "reid": {"score": ident.get("score")} if ident.get("score") is not None else None,
        "face": ident.get("face"),
        "gait": ident.get("gait"),
        "fused": ident.get("fused"),
        "merge_routes": ident.get("merge_routes"),
        "merge_agree": ident.get("merge_agree"),
        "cross_track_merged": ident.get("cross_track_merged", False),
    }


def _group_people(
    win_tracks: list[int], tracks: dict[int, dict], identities: dict[int, dict],
    win_idx: list[int], img_w: int, img_h: int,
) -> list[dict]:
    """把窗内 track 按 subject 合并成 person 条目。

    同一 subject_id 的多条 track（ByteTrack 因遮挡/漂移把一个人断成多段）合并为**一个人**：
    轨迹跨 track 按帧序拼接、ReID 取最高分、box 取代表 track 的末位框，避免 LLM 把"一个人的
    若干轨迹"误数成多个人。subject_id 为空（没认出库内主体）的 track 各自独立成条。
    """
    win_set = set(win_idx)
    groups: dict[str, list[int]] = {}
    for tid in win_tracks:
        if tid not in tracks:
            continue
        sid = identities[tid].get("subject_id")
        key = f"subject:{sid}" if sid is not None else f"track:{tid}"
        groups.setdefault(key, []).append(tid)

    people: list[dict] = []
    for key, tids in groups.items():
        if len(tids) == 1:
            tid = tids[0]
            people.append(_person_record(tid, tracks[tid], identities[tid], win_idx, img_w, img_h))
            continue
        # 多 track → 同一人：选分最高者为代表，合并轨迹/取最高 ReID 分
        rep = max(tids, key=lambda t: identities[t].get("score") or 0.0)
        merged_centers = sorted(
            [(i, c) for t in tids for (i, c) in tracks[t]["centers"] if i in win_set],
            key=lambda x: x[0],
        )
        best_score = max((identities[t].get("score") or 0.0) for t in tids)
        rep_box = tracks[rep]["boxes"].get(win_idx[-1]) or tracks[rep].get("best_box") or []
        face = next((identities[t].get("face") for t in tids if identities[t].get("face")), None)
        gait = next((identities[t].get("gait") for t in tids if identities[t].get("gait")), None)
        # 融合：取置信度最高的那条 track 的融合结果作代表
        fused = max(
            (identities[t].get("fused") for t in tids if identities[t].get("fused")),
            key=lambda fz: (fz or {}).get("confidence", 0.0), default=None,
        )
        # 跨 track 合并用到了哪几路证据（人脸/人形/步态），并到代表里
        merge_routes = sorted({r for t in tids for r in (identities[t].get("merge_routes") or [])})
        route_cn = {"face": "人脸库", "body": "人形库", "gait": "步态库"}
        attrs = [f"由{len(tids)}条轨迹合并(同一人)"]
        if merge_routes:
            attrs.append("跨track印证：" + "+".join(route_cn.get(r, r) for r in merge_routes))
        people.append({
            "track_id": rep,
            "box": rep_box,
            "subject_id": identities[rep].get("subject_id"),
            "decision": "hit",  # 跨 track 复用即为命中已知主体
            "reused": True,
            "trajectory": [list(c) for (_, c) in merged_centers],
            "reid": {"score": round(best_score, 4)} if best_score > 0 else None,
            "face": face,
            "gait": gait,
            "fused": fused,
            "merge_routes": merge_routes or None,
            "merge_agree": len(merge_routes) or None,
            "attributes": attrs,
        })
    return people


def _stitch_orphans(
    tracks: dict[int, dict],
    identities: dict[int, dict],
    track_emb: dict[int, np.ndarray],
    thresh: float,
) -> None:
    """把灰区孤立 track（subject_id 为空）并进同视频内最相近的已建主体（就地改 identities）。

    做法：用各 track 的 ReID 向量，为每个已知主体算一个代表向量（成员均值，再归一化），
    按 track 出现时间顺序处理孤立 track；与某主体相似度 ≥ thresh 即并入该主体（标记 decision
    ="stitched"、reused=True），并把它的向量并进该主体代表里（让后续断片能接力缝合）。
    不达阈值则保持孤立（大概率是不同的人，宁缺毋滥）。
    """
    def _norm(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    # 各主体的成员向量（来自已分配 subject_id 的 track）
    members: dict[int, list[np.ndarray]] = {}
    for tid, idn in identities.items():
        sid = idn.get("subject_id")
        if sid is not None and tid in track_emb:
            members.setdefault(sid, []).append(track_emb[tid])
    if not members:
        return
    reps: dict[int, np.ndarray] = {sid: _norm(np.mean(vs, axis=0)) for sid, vs in members.items()}

    # 孤立 track：按首次出现时间顺序缝合
    orphans = [tid for tid, idn in identities.items()
               if idn.get("subject_id") is None and tid in track_emb]
    orphans.sort(key=lambda t: tracks[t]["first"])

    for tid in orphans:
        v = _norm(track_emb[tid])
        best_sid, best_sim = None, -1.0
        for sid, rep in reps.items():
            sim = float(np.dot(v, rep))
            if sim > best_sim:
                best_sid, best_sim = sid, sim
        if best_sid is not None and best_sim >= thresh:
            idn = identities[tid]
            idn["subject_id"] = best_sid
            idn["decision"] = "stitched"
            idn["reused"] = True
            idn["stitch_score"] = round(best_sim, 4)
            if idn.get("score") is None:
                idn["score"] = round(best_sim, 4)
            # 并入代表，便于后续断片接力
            members[best_sid].append(track_emb[tid])
            reps[best_sid] = _norm(np.mean(members[best_sid], axis=0))


def _merge_tracks_cross_route(identities: dict[int, dict]) -> None:
    """跨 track 三路合并：人脸库 / 人形库 / 步态库 **任一路**认出同一人 → 并成一个 subject。

    动机：人形缝合(_stitch_orphans)只用人形 ReID 一路；但同一个人在不同 track 里，可能人形
    糊了却**人脸命中同号**、或人脸糊了却**步态命中同号**。这里用并查集，把"任意一路库编号相同"
    的 track 并成同一人——多路同时印证则置信更高（写进每条 track 的 merge_routes/merge_agree）。

    实现：三路各自把 track 按各自库编号分组，同组内两两 union；并完后每个连通分量=一个人，
    统一改写 identities[tid]['subject_id'] 为该分量的规范主体号（优先沿用分量内已有人形主体号，
    取最小；若整分量都没有人形主体号则新铸一个），下游 _group_people 即可自然按统一 subject 归并。
    """
    tids = list(identities.keys())
    if len(tids) < 2:
        return

    parent = {t: t for t in tids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    def _route_id(idn: dict, route: str):
        if route == "body":
            return idn.get("subject_id")
        if route == "face":
            fc = idn.get("face") or {}
            # 仅清晰且命中库内主体才算"认出同一人"（糊脸只查不建，不足以做跨 track 锚点）
            if fc.get("matched") and fc.get("quality") == "clear":
                return fc.get("face_subject_id")
            return None
        if route == "gait":
            gt = idn.get("gait") or {}
            if gt.get("decision") == "hit":
                return gt.get("subject_id")
        return None

    # 三路分别按库编号分组 → 组内两两并；记录每条 track 触发合并用到了哪几路
    routes = ("body", "face", "gait")
    route_of_edge: dict[frozenset, set] = {}
    for route in routes:
        buckets: dict = {}
        for t in tids:
            gid = _route_id(identities[t], route)
            if gid is not None:
                buckets.setdefault(gid, []).append(t)
        for members in buckets.values():
            if len(members) < 2:
                continue
            base = members[0]
            for other in members[1:]:
                union(base, other)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    route_of_edge.setdefault(frozenset((members[i], members[j])), set()).add(route)

    # 连通分量 → 统一主体号
    comps: dict[int, list[int]] = {}
    for t in tids:
        comps.setdefault(find(t), []).append(t)

    existing_body = [idn.get("subject_id") for idn in identities.values()
                     if idn.get("subject_id") is not None]
    next_synth = (max(existing_body) + 1) if existing_body else 1

    for members in comps.values():
        if len(members) < 2:
            continue
        body_ids = [identities[t].get("subject_id") for t in members
                    if identities[t].get("subject_id") is not None]
        if body_ids:
            canonical = min(body_ids)
        else:
            canonical = next_synth
            next_synth += 1
        # 该分量里实际用到了哪几路证据（用于置信标注）
        comp_routes: set = set()
        mset = set(members)
        for edge, rs in route_of_edge.items():
            if edge <= mset:
                comp_routes |= rs
        agree = len(comp_routes)
        for t in members:
            idn = identities[t]
            prev = idn.get("subject_id")
            idn["subject_id"] = canonical
            idn["merge_routes"] = sorted(comp_routes)
            idn["merge_agree"] = agree
            if prev != canonical:
                idn["cross_track_merged"] = True
                idn["reused"] = True
                if idn.get("decision") not in ("hit", "stitched"):
                    idn["decision"] = "merged"


def _split_windows(
    metas: list[FrameMeta], quiet_frames: int, max_window_frames: int | None = None
) -> list[list[int]]:
    """把帧序列按"活动段 + 时长上限"切窗，返回每个窗的帧 index 列表。

    关窗条件二选一：
      (a) 活动结束：连续 ≥ quiet_frames 帧无人（允许桥接 < quiet_frames 的短暂无人）；
      (b) 时长封顶：窗内帧数达到 max_window_frames（= 时长上限 × fps）→ 冲刷并立刻开新窗。

    (b) 是为"长连续事件"准备的：否则一个人在画面里连续待很久会被压成单窗、只调一次 LLM、
    关键帧被严重欠采样。封顶后长事件会被切成多个窗、各调一次，细节不丢（跨窗的"谁"靠 ReID
    主体记忆保持一致）。
    """
    windows: list[list[int]] = []
    cur: list[int] = []
    gap = 0
    for m in metas:
        if m.active_tracks:
            cur.append(m.index)
            gap = 0
            if max_window_frames and len(cur) >= max_window_frames:
                windows.append(cur)          # 时长封顶 → 冲刷
                cur = []
                gap = 0
        elif cur:
            gap += 1
            if gap >= quiet_frames:
                windows.append(cur)          # 活动结束 → 冲刷
                cur = []
                gap = 0
            else:
                cur.append(m.index)          # 短暂无人，桥接进当前窗
    if cur:
        windows.append(cur)
    return windows


__all__ = ["analyze_event_stream"]
