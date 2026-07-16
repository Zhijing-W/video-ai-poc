"""身份感知·多帧事件理解 端到端编排（Phase 4 · Step 24）—— 把叶子串成一条链路。

定位：这是 Phase 4 的**集成步**。前面写好的叶子模块都是"只做一件事"的纯函数式组件：
  - `video_processor.extract_frames`（选帧① 定时密采样）
  - `tracker.track_objects`（YOLO + ByteTrack，给每个人稳定 track_id）
  - `reid` + `gallery`（人形指纹 + 主体记忆库：认过一次就记住 → 身份）
  - `face`（可选：人脸指纹，清晰正脸时的最强身份信号）
  - `keyframe.select_keyframes`（选帧② 事件驱动地把几百帧砍到几十帧）
  - `identity.identity_context.format_identity_grounding`（把身份打包成 LLM 能读的文本）
  - `services.event_reporter.understand_event`（多帧 + 身份 → 跨帧事件叙述）

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

import base64
import io
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import face as face_mod
from . import gait as gait_mod
from . import ocr as ocr_mod
from . import body_reid as reid_mod
from . import tracker as tracker_mod
from .core.config import settings
from .identity import embedding_gallery as gallery_mod
from .identity.face_attachment import attach_faces
from .identity.identity_confidence import score_identity_confidence
from .identity.identity_context import format_identity_grounding
from .identity.resolution import (
    group_people,
    merge_tracks_cross_route,
    person_record,
    split_subject_time_conflicts,
    stitch_orphans,
)
from .keyframe import FrameMeta, select_keyframes
from .pipeline.object_context import build_object_context, format_object_context
from .pipeline.session import EventAnalysisSession
from .pipeline.spatial_context import (
    build_spatial_grounding,
    center_from_box,
    direction,
    format_spatial_grounding,
    norm_box,
)
from .pipeline.windowing import split_windows
from .services.event_reporter import summarize_event_windows, understand_event
from .utils.image_utils import image_to_data_uri, seconds_to_timestamp
from .video_processor import Frame, extract_frames


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

def _pil_to_thumb_uri(img: Image.Image | None, max_h: int = 128) -> str | None:
    """把一张 PIL 裁图缩成小缩略图并转 data URI（前端身份画廊头像用）。失败返回 None。"""
    if img is None:
        return None
    try:
        im = img.convert("RGB")
        w, h = im.size
        if h > max_h:
            im = im.resize((max(1, int(w * max_h / h)), max_h))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None

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

# ---------------- 兼容导出 / shim ----------------
_norm_box = norm_box
_center_from_box = center_from_box
_direction = direction
_build_spatial_grounding = build_spatial_grounding
_format_spatial_grounding = format_spatial_grounding
_build_object_context = build_object_context
_format_object_context = format_object_context
_attach_faces = attach_faces
_person_record = person_record
_group_people = group_people
_stitch_orphans = stitch_orphans
_split_subject_time_conflicts = split_subject_time_conflicts
_merge_tracks_cross_route = merge_tracks_cross_route
_split_windows = split_windows


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
    with_ocr: bool = False,
    with_objects: bool = False,
    objective: str | None = None,
    quiet_seconds: float = 2.0,
    max_keyframes: int | None = None,
    include_keyframe_images: bool = False,
    max_window_seconds: float | None = None,
    stitch_thresh: float | None = None,
    overall_summary: bool | None = None,
) -> dict:
    """对一段视频做"身份感知·多帧事件理解"的完整端到端处理。"""
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    t_start = time.time()

    stage_timings: dict[str, float] = {}
    _cursor = [time.time()]

    def _lap(name: str) -> None:
        now = time.time()
        stage_timings[name] = round(now - _cursor[0], 2)
        _cursor[0] = now

    frames = extract_frames(video_path, frames_dir, max_frames=max_frames, fps=fps)
    _lap("extract_frames")

    session = EventAnalysisSession(
        video_path=video_path,
        out_dir=out_dir,
        session_id=session_id,
        finish_fn=lambda buffered_frames: _finish_session(
            buffered_frames,
            video_path=video_path,
            out_dir=out_dir,
            fps=fps,
            session_id=session_id,
            run_llm=run_llm,
            with_face=with_face,
            with_gait=with_gait,
            with_ocr=with_ocr,
            with_objects=with_objects,
            objective=objective,
            quiet_seconds=quiet_seconds,
            max_keyframes=max_keyframes,
            include_keyframe_images=include_keyframe_images,
            max_window_seconds=max_window_seconds,
            stitch_thresh=stitch_thresh,
            overall_summary=overall_summary,
            stage_timings=stage_timings,
            t_start=t_start,
            cursor_start=_cursor[0],
        ),
    )
    for frame in frames:
        session.process_frame(frame)
    return session.finish()


def _finish_session(
    frames: list[Frame],
    *,
    video_path: str | Path,
    out_dir: str | Path,
    fps: float,
    session_id: str,
    run_llm: bool,
    with_face: bool,
    with_gait: bool,
    with_ocr: bool,
    with_objects: bool,
    objective: str | None,
    quiet_seconds: float,
    max_keyframes: int | None,
    include_keyframe_images: bool,
    max_window_seconds: float | None,
    stitch_thresh: float | None,
    overall_summary: bool | None,
    stage_timings: dict[str, float],
    t_start: float,
    cursor_start: float,
) -> dict:
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    step = 1.0 / float(fps)
    _cursor = [cursor_start]

    def _lap(name: str) -> None:
        now = time.time()
        stage_timings[name] = round(now - _cursor[0], 2)
        _cursor[0] = now
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
    # 物体轨迹（LANE D）：非 person 目标的跨帧记录（label/boxes/centers/first/last），结构仿 tracks
    object_tracks: dict[int, dict] = {}
    prev_person_count = 0
    # 步态可用性（OpenGait+权重就绪才采集，避免每帧白跑 pose/seg）
    gait_use = bool(with_gait and settings.gait_enabled and gait_mod.available())
    gait_collect_error = None
    # 场景文字 OCR 可用性（LANE D）：显式开关 with_ocr 或全局 settings.ocr_enabled
    ocr_use = bool(with_ocr or settings.ocr_enabled)
    # 物体/包裹检测可用性（LANE D）：显式开关 with_objects 或全局 settings.object_detect
    obj_use = bool(with_objects or settings.object_detect)
    obj_classes = settings.object_class_set() if obj_use else set()

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

        # ---- 物体/包裹检测（LANE D）：把非 person 的关注类目标捡回来，逐帧累积轨迹 ----
        if obj_use:
            for d in res["detections"]:
                lbl = d.get("label")
                otid = d.get("track_id")
                if (lbl == "person" or lbl not in obj_classes or otid is None
                        or float(d.get("confidence") or 0.0) < settings.object_min_conf):
                    continue
                otid = int(otid)
                box = d["box"]
                ot = object_tracks.get(otid)
                if ot is None:
                    ot = {"label": lbl, "first": i, "last": i, "centers": {}, "boxes": {},
                          "max_conf": 0.0}
                    object_tracks[otid] = ot
                    events.append(f"object_new:{lbl}#{otid}")
                ot["last"] = i
                ot["label"] = lbl  # 以最近一次标签为准（同一 track 类别一般稳定）
                ot["boxes"][i] = box
                ot["max_conf"] = max(ot["max_conf"], float(d.get("confidence") or 0.0))
                ocx = (box[0] + box[2]) / 2.0 / max(1, img_w)
                ocy = (box[1] + box[3]) / 2.0 / max(1, img_h)
                ot["centers"][i] = (round(ocx, 3), round(ocy, 3))

        # ---- 步态采集已移到"分窗之后"的第二遍，只在活动窗帧里跑（见下方 pass-2），避免全片逐帧白跑 ----

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

    # ---- 流式分窗（提前到重活之前）：先用便宜的 YOLO 占用信息切事件窗，
    #      后面最重的步态采集只在"活动窗内的帧"跑，避免全片逐帧白跑 pose/seg。----
    win_secs = settings.event_window_max_seconds if max_window_seconds is None else max_window_seconds
    max_window_frames = int(round(win_secs * fps)) if win_secs and win_secs > 0 else None
    windows = _split_windows(metas, quiet_frames, max_window_frames)
    if not windows:  # 全程无人/无活动 → 整段当一个窗，LLM 仍可描述场景
        windows = [list(range(len(metas)))]
    windowed_frames = sorted({i for win in windows for i in win})

    # ---- Track 级门控：筛掉"太短/太低质"的 track，整条不做身份提取（省 reid/face/步态 + 防污染库）----
    def _track_worth_identity(t: dict) -> bool:
        n_frames = len(t.get("boxes", {}))
        if settings.track_min_frames and n_frames < settings.track_min_frames:
            return False
        if settings.track_min_quality and float(t.get("best_q", 0.0)) < settings.track_min_quality:
            return False
        return True

    skip_tracks = {tid for tid, t in tracks.items() if not _track_worth_identity(t)}
    _lap("detect_track")

    # ---- 步态采集（第二遍·仅活动窗帧，且跳过被门控掉的 track）：把最重的逐帧 YOLO-Pose+Seg 只在需要处跑 ----
    if gait_use:
        for i in windowed_frames:
            present = [(tid, t["boxes"][i]) for tid, t in tracks.items()
                       if i in t.get("boxes", {}) and tid not in skip_tracks]
            if not present:
                continue
            try:
                bgr = np.asarray(Image.open(frames[i].local_path).convert("RGB"))[:, :, ::-1]
                gp = gait_mod.extract_persons(bgr)
                for tid, pb in present:
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

    # ---- 认人：每条 track 用最佳 crop 提指纹、查/登记主体记忆库 → 身份 ----
    if gait_use:
        _lap("gait_collect")
    identities: dict[int, dict] = {}
    track_emb: dict[int, np.ndarray] = {}   # track_id -> ReID 向量（缝合用）
    for tid, t in tracks.items():
        ident = {"track_id": tid, "subject_id": None, "decision": None,
                 "score": None, "reused": False, "face": None}
        if tid in skip_tracks:
            # 门控：太短/太低质的 track 整条跳过身份提取（仍出现在事件里，身份留空、不入库）
            ident["skipped"] = f"low_track(frames={len(t.get('boxes', {}))},q={float(t.get('best_q', 0.0)):.1f})"
            identities[tid] = ident
            continue
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
                ident["quality_ok"] = res.get("quality_ok")
                ident["quality_reason"] = res.get("quality_reason")
                ident["enrolled"] = res.get("enrolled")
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
    _lap("reid_identify")
    if with_face:
        _attach_faces(frames, tracks, identities, session_id)
        _lap("face")

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
    if gait_use:
        _lap("gait_embed")
    if identities:
        _merge_tracks_cross_route(identities)
        _split_subject_time_conflicts(tracks, identities)

    # ---- A 汇聚：多路线身份置信度（人脸 + 人形 + 步态按质量加权）----
    for tid in identities:
        score_identity_confidence(identities[tid])

    # ---- 身份画廊头像：每条 track 的最佳人物 crop 缩略图（仅 Web 端要图时才生成）----
    if include_keyframe_images:
        for tid, t in tracks.items():
            if tid in identities and t.get("best_crop") is not None:
                thumb = _pil_to_thumb_uri(t["best_crop"])
                if thumb:
                    identities[tid]["thumb"] = thumb

    # 事件窗已在前面（重活之前）切好；此处直接逐窗选帧②并做多帧事件理解
    _lap("merge_fusion_thumb")
    idx2frame = {i: fr for i, fr in enumerate(frames)}
    ocr_cache: dict[int, list[dict]] = {}  # 帧 index → OCR 结果，避免重复 OCR
    out_windows: list[dict] = []
    for w, win_idx in enumerate(windows):
        win_metas = [metas[i] for i in win_idx]
        sel = select_keyframes(win_metas, max_frames=max_keyframes)
        if not sel:
            sel = win_idx[: (max_keyframes or settings.keyframe_max)]

        # 该窗涉及的 track → 按 subject 合并后打包身份（避免同一人被列成多条、误导 LLM 计数）
        win_tracks = sorted({t for i in win_idx for t in metas[i].active_tracks})
        kf = [{"image": idx2frame[i].local_path, "timestamp": idx2frame[i].timestamp} for i in sel]
        people = _group_people(win_tracks, tracks, identities, win_idx, img_w, img_h)
        spatial_grounding = _build_spatial_grounding(sel, people, tracks, identities, metas, img_w, img_h)
        identity_text = format_identity_grounding(people, img_w, img_h)
        grounding_text = _format_spatial_grounding(spatial_grounding)
        if grounding_text:
            identity_text = (identity_text + "\n\n" + grounding_text) if identity_text else grounding_text
        ts_range = [metas[win_idx[0]].timestamp, metas[win_idx[-1]].timestamp]

        # ---- LANE D：场景文字 OCR —— 在该窗关键帧上读时间戳/车牌/单号，汇成 scene_context ----
        scene_context = ""
        if ocr_use:
            per_frame = []
            for i in sel:
                if i not in ocr_cache:
                    ocr_cache[i] = ocr_mod.read_frame(idx2frame[i].local_path)
                # frame_index + FrameMeta.timestamp：与空间 grounding 用同一套 frame#N @ ts 锚点，
                # 让 LLM 能逐关键帧对齐"谁在哪(grounding) + 画面时钟读数(OCR)"。
                per_frame.append({
                    "frame_index": i,
                    "timestamp": metas[i].timestamp if 0 <= i < len(metas) else None,
                    "texts": ocr_cache[i],
                })
            scene_context = ocr_mod.format_scene_context(per_frame)

        # ---- LANE D：物体/包裹 —— 汇总窗内非人物体轨迹 → object_context（场景级，含 logo 提示）----
        object_list: list[dict] = []
        object_context = ""
        if obj_use:
            object_list = _build_object_context(object_tracks, win_idx, metas, img_w, img_h)
            object_context = _format_object_context(object_list)

        window_out = {
            "window_index": w,
            "time_range": [seconds_to_timestamp(ts_range[0]), seconds_to_timestamp(ts_range[1])],
            "frame_count": len(win_idx),
            "keyframe_indices": sel,
            "keyframe_timestamps": [idx2frame[i].timestamp for i in sel],
            "events": sorted({e for i in win_idx for e in metas[i].events}),
            "people": people,
            "spatial_grounding": spatial_grounding,
            "identity_context": identity_text,
        }
        if scene_context:
            window_out["scene_context"] = scene_context
        if object_context:
            window_out["object_context"] = object_context
            window_out["objects"] = object_list
        if include_keyframe_images:
            window_out["keyframes"] = [
                {"timestamp": idx2frame[i].timestamp, "image": image_to_data_uri(idx2frame[i].local_path)}
                for i in sel
            ]
        if run_llm:
            window_out["event"] = understand_event(
                kf, identity_text, objective=objective,
                scene_context=scene_context or None, object_context=object_context or None,
            )
        out_windows.append(window_out)

    # ---- 跨窗整段事件总结：所有窗理解完后，纯文本把多窗串成整段连贯故事（便宜；dry-run 跳过）----
    _lap("windows_llm" if run_llm else "windows_select")
    overall = None
    do_overall = settings.event_overall_summary if overall_summary is None else overall_summary
    if run_llm and do_overall and out_windows:
        try:
            overall = summarize_event_windows(out_windows) or None
        except Exception as exc:  # 总结失败不致命：逐窗结果仍在
            overall = {"error": str(exc)}
    if overall is not None:
        _lap("overall_summary")

    return {
        "video": str(video_path),
        "fps": fps,
        "frames_total": len(frames),
        "img_size": [img_w, img_h],
        "session_id": session_id,
        "tracker_backend": tracker_mod.active_backend(),
        "reid_backend": reid_mod.active_backend(),
        "reid_dim": dim,
        "with_face": with_face,
        "with_gait": gait_use,
        "with_ocr": ocr_use,
        "with_objects": obj_use,
        "gait_error": (gait_mod.load_error() if (with_gait and not gait_use) else gait_collect_error),
        "ocr_backend": (ocr_mod.active_backend() if ocr_use else None),
        "ocr_error": (ocr_mod.load_error() if ocr_use else None),
        "object_classes": (sorted(obj_classes) if obj_use else None),
        "model": settings.event_llm_deployment or settings.azure_openai_deployment,
        "dry_run": not run_llm,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "stage_timings": stage_timings,
        "tracks": {str(tid): identities[tid] for tid in identities},
        "windows": out_windows,
        "overall": overall,
    }

__all__ = ["analyze_event_stream"]
