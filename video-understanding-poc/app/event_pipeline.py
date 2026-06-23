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
from . import gallery as gallery_mod
from . import reid as reid_mod
from . import tracker as tracker_mod
from .core.config import settings
from .keyframe import FrameMeta, select_keyframes
from .services.event_understanding import understand_event
from .services.identity_context import format_identity_context
from .utils.image_utils import seconds_to_timestamp
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
    objective: str | None = None,
    quiet_seconds: float = 2.0,
    max_keyframes: int | None = None,
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
        objective: 可选关注点，写进事件理解 prompt（如"留意陌生人/包裹被取走"）。
        quiet_seconds: 连续无人多久算"活动结束"，用于切分事件窗。

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
    for tid, t in tracks.items():
        ident = {"track_id": tid, "subject_id": None, "decision": None,
                 "score": None, "reused": False, "face": None}
        crop = t["best_crop"]
        if crop is not None:
            try:
                vec = reid_mod.embed(crop)
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

    # 身份命中 → 在该 track 首帧补 identity_hit 语义事件
    for tid, ident in identities.items():
        if ident.get("reused") and ident.get("subject_id") is not None:
            fi = tracks[tid]["first"]
            metas[fi].events.append(f"identity_hit:主体#{ident['subject_id']}")

    # ---- 可选人脸分支：仅在每条 track 的最佳帧上稀疏跑（攻"人脸模糊"的同时不拖慢全片）----
    if with_face:
        _attach_faces(frames, tracks, identities)

    # ---- 流式分窗：把帧序列按"活动段"切成事件窗 ----
    windows = _split_windows(metas, quiet_frames)
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
        if run_llm:
            window_out["event"] = understand_event(kf, identity_text, objective=objective)
        out_windows.append(window_out)

    return {
        "video": str(video_path),
        "fps": fps,
        "frames_total": len(frames),
        "img_size": [img_w, img_h],
        "session_id": session_id,
        "reid_backend": reid_mod.active_backend(),
        "reid_dim": dim,
        "with_face": with_face,
        "model": settings.event_llm_deployment or settings.azure_openai_deployment,
        "dry_run": not run_llm,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "tracks": {str(tid): identities[tid] for tid in identities},
        "windows": out_windows,
    }


def _attach_faces(frames: list[Frame], tracks: dict[int, dict], identities: dict[int, dict]) -> None:
    """对每条 track 的最佳帧跑一次人脸检测，关联到该 track，写进 identity['face']。"""
    by_frame: dict[int, list[int]] = {}
    for tid, t in tracks.items():
        by_frame.setdefault(t["best_idx"], []).append(tid)
    for fidx, tids in by_frame.items():
        try:
            pil = Image.open(frames[fidx].local_path).convert("RGB")
            faces = face_mod.detect(pil)
        except Exception:
            continue
        person_dets = [{"box": tracks[tid]["best_box"], "track_id": tid, "label": "person"} for tid in tids]
        assoc = face_mod.associate_to_persons(faces, person_dets)
        for tid, fc in assoc.items():
            q = fc.get("quality", {}) or {}
            identities[tid]["face"] = {
                "score": q.get("det_score"),
                "quality": "clear" if q.get("quality_ok") else "blurry",
                "matched": False,  # 这里只提脸、未接人脸库比对（库比对留作后续）
            }


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
        "gait": None,
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
        people.append({
            "track_id": rep,
            "box": rep_box,
            "subject_id": identities[rep].get("subject_id"),
            "decision": "hit",  # 跨 track 复用即为命中已知主体
            "reused": True,
            "trajectory": [list(c) for (_, c) in merged_centers],
            "reid": {"score": round(best_score, 4)} if best_score > 0 else None,
            "face": face,
            "gait": None,
            "attributes": [f"由{len(tids)}条轨迹合并(同一人)"],
        })
    return people


def _split_windows(metas: list[FrameMeta], quiet_frames: int) -> list[list[int]]:
    """把帧序列按活动段切窗：连续有人为一窗，允许桥接 < quiet_frames 的短暂无人。"""
    windows: list[list[int]] = []
    cur: list[int] = []
    gap = 0
    for m in metas:
        if m.active_tracks:
            cur.append(m.index)
            gap = 0
        elif cur:
            gap += 1
            if gap >= quiet_frames:
                windows.append(cur)
                cur = []
                gap = 0
            else:
                cur.append(m.index)  # 短暂无人，桥接进当前窗
    if cur:
        windows.append(cur)
    return windows


__all__ = ["analyze_event_stream"]
