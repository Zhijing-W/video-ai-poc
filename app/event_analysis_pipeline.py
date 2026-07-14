"""身份感知·多帧事件理解 端到端编排（Phase 4 · Step 24）—— 把叶子串成一条链路。

定位：这是 Phase 4 的**集成步**。前面写好的叶子模块都是"只做一件事"的纯函数式组件：
  - `video_processor.extract_frames`（选帧① 定时密采样）
  - `tracker.track_objects`（YOLO + ByteTrack，给每个人稳定 track_id）
  - `reid` + `gallery`（人形指纹 + 主体记忆库：认过一次就记住 → 身份）
  - `face`（可选：人脸指纹，清晰正脸时的最强身份信号）
  - `keyframe.select_keyframes`（选帧② 事件驱动地把几百帧砍到几十帧）
  - `identity_grounding.format_identity_grounding`（把身份打包成 LLM 能读的文本）
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

import base64
import io
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import face as face_mod
from . import gait as gait_mod
from . import body_gallery as gallery_mod
from . import ocr as ocr_mod
from . import body_reid as reid_mod
from . import tracker as tracker_mod
from .core.config import settings
from .keyframe import FrameMeta, select_keyframes
from .services.event_understanding import summarize_event_windows, understand_event
from .services.identity_grounding import format_identity_grounding
from .services.multimodal_identity_fusion import fuse_multimodal_identity
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
        with_ocr: 是否启用场景文字 OCR（LANE D）。读关键帧里的时间戳/车牌/单号等**场景级**文字，
            汇成 scene_context 与人物身份**并列**喂 LLM；**不进** subject_id/gallery/融合。
        with_objects: 是否启用物体/包裹检测（LANE D）。把非 person 目标(包裹/行李/车辆)从 YOLO 结果
            捡回来、记跨帧轨迹，汇成 object_context 与身份**并列**喂 LLM；同样**不进** subject_id。
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

    # ---- 阶段计时：记录每个重活阶段耗时（秒），放进 payload 供前端进度/性能观测 ----
    stage_timings: dict[str, float] = {}
    _cursor = [time.time()]

    def _lap(name: str) -> None:
        now = time.time()
        stage_timings[name] = round(now - _cursor[0], 2)
        _cursor[0] = now

    # ---- 选帧①：定时密采样（当作"流"的来源）----
    step = 1.0 / float(fps)
    frames = extract_frames(video_path, frames_dir, max_frames=max_frames, fps=fps)
    _lap("extract_frames")

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

    # ---- A 汇聚：三路身份融合（人脸 + 人形 + 步态 按质量加权 → 统一身份置信度）----
    for tid in identities:
        fuse_multimodal_identity(identities[tid])

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


def _norm_box(box: list[float], img_w: int, img_h: int) -> list[float]:
    if not box or img_w <= 0 or img_h <= 0:
        return []
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    vals = [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]
    return [round(max(0.0, min(1.0, v)), 4) for v in vals]


def _center_from_box(box: list[float], img_w: int, img_h: int) -> list[float]:
    if not box or img_w <= 0 or img_h <= 0:
        return []
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return [round(((x1 + x2) / 2.0) / img_w, 4), round(((y1 + y2) / 2.0) / img_h, 4)]


def _direction(points: list[list[float]]) -> str:
    if len(points) < 2:
        return "unknown"
    sx, sy = points[0]
    ex, ey = points[-1]
    dx, dy = ex - sx, ey - sy
    parts: list[str] = []
    if abs(dx) >= 0.05:
        parts.append("right" if dx > 0 else "left")
    if abs(dy) >= 0.05:
        parts.append("down" if dy > 0 else "up")
    return "+".join(parts) if parts else "mostly_static"


def _build_spatial_grounding(
    keyframe_indices: list[int],
    people: list[dict],
    tracks: dict[int, dict],
    identities: dict[int, dict],
    metas: list[FrameMeta],
    img_w: int,
    img_h: int,
) -> dict:
    """Build object-centric spatial evidence for LLM grounding: per-keyframe boxes + trajectory summary."""
    tid_to_subject = {tid: identities.get(tid, {}).get("subject_id") for tid in tracks}
    frames: list[dict] = []
    for idx in keyframe_indices:
        objects: list[dict] = []
        active = metas[idx].active_tracks if 0 <= idx < len(metas) else []
        for tid in active:
            t = tracks.get(tid)
            if not t:
                continue
            box = t["boxes"].get(idx)
            if not box:
                continue
            sid = tid_to_subject.get(tid)
            objects.append({
                "track_id": tid,
                "subject_id": sid,
                "label": f"subject#{sid}" if sid is not None else f"track#{tid}",
                "bbox": [round(float(v), 1) for v in box[:4]],
                "bbox_norm": _norm_box(box, img_w, img_h),
                "center_norm": _center_from_box(box, img_w, img_h),
                "decision": identities.get(tid, {}).get("decision"),
            })
        frames.append({
            "frame_index": idx,
            "timestamp": metas[idx].timestamp if 0 <= idx < len(metas) else None,
            "objects": objects,
        })

    trajectories: list[dict] = []
    for p in people:
        pts = p.get("trajectory") or []
        if not pts:
            continue
        show = [pts[0], pts[len(pts) // 2], pts[-1]] if len(pts) >= 3 else pts
        trajectories.append({
            "subject_id": p.get("subject_id"),
            "track_id": p.get("track_id"),
            "track_ids": p.get("source_track_ids") or [p.get("track_id")],
            "label": f"subject#{p.get('subject_id')}" if p.get("subject_id") is not None else f"track#{p.get('track_id')}",
            "path_sample": [[round(float(x), 4) for x in pt] for pt in show],
            "direction": _direction([[float(x) for x in pt] for pt in pts]),
            "points": len(pts),
        })
    return {
        "image_size": [img_w, img_h],
        "coord": "bbox=[x1,y1,x2,y2] pixels; bbox_norm/center_norm normalized to 0..1 from top-left",
        "frames": frames,
        "trajectories": trajectories,
    }


def _format_spatial_grounding(grounding: dict) -> str:
    """Compact text block appended to identity_context so the LLM can bind subjects to frame coordinates."""
    frames = grounding.get("frames") or []
    trajectories = grounding.get("trajectories") or []
    if not frames and not trajectories:
        return ""
    lines = [
        "【关键帧空间 grounding（坐标辅助理解，不要求模型重新检测）】",
        f"坐标约定：{grounding.get('coord', '')}",
    ]
    for fr in frames:
        objs = fr.get("objects") or []
        lines.append(f"- frame#{fr.get('frame_index')} @ {fr.get('timestamp')}：{len(objs)} objects")
        for obj in objs[:24]:
            lines.append(
                "  "
                f"{obj.get('label')} track={obj.get('track_id')} "
                f"bbox_norm={obj.get('bbox_norm')} center={obj.get('center_norm')}"
            )
        if len(objs) > 24:
            lines.append(f"  ... {len(objs) - 24} more objects omitted")
    if trajectories:
        lines.append("【跨帧轨迹摘要】")
        for tr in trajectories[:40]:
            lines.append(
                f"- {tr.get('label')} tracks={tr.get('track_ids')} "
                f"direction={tr.get('direction')} path={tr.get('path_sample')}"
            )
        if len(trajectories) > 40:
            lines.append(f"... {len(trajectories) - 40} more trajectories omitted")
    return "\n".join(lines)


# COCO 标签 → 中文（仅常见的 LANE D 物体类，便于 LLM/人读；缺省回退英文原名）
_OBJ_LABEL_CN = {
    "backpack": "背包", "handbag": "手提包", "suitcase": "行李箱/箱子",
    "car": "汽车", "truck": "卡车", "bus": "公交车",
    "motorcycle": "摩托车", "bicycle": "自行车",
}


def _build_object_context(
    object_tracks: dict[int, dict], win_idx: list[int], metas: list[FrameMeta],
    img_w: int, img_h: int,
) -> list[dict]:
    """汇总落在本窗内的非人物体轨迹：label、起止 frame#@ts、运动方向（场景级，非身份）。"""
    win_set = set(win_idx)
    objs: list[dict] = []
    for otid, ot in object_tracks.items():
        idxs = sorted(i for i in ot["boxes"] if i in win_set)
        if len(idxs) < settings.object_min_frames:  # 出现帧数太少 → 多为误检/ID 跳变，丢弃
            continue
        pts = [list(ot["centers"][i]) for i in idxs if i in ot["centers"]]
        first_i, last_i = idxs[0], idxs[-1]
        objs.append({
            "track_id": otid,
            "label": ot["label"],
            "label_cn": _OBJ_LABEL_CN.get(ot["label"], ot["label"]),
            "first_ts": metas[first_i].timestamp if 0 <= first_i < len(metas) else None,
            "last_ts": metas[last_i].timestamp if 0 <= last_i < len(metas) else None,
            "first_frame": first_i,
            "last_frame": last_i,
            "direction": _direction(pts) if len(pts) >= 2 else "unknown",
            "frames_present": len(idxs),
            "conf": round(float(ot.get("max_conf", 0.0)), 3),
        })
    objs.sort(key=lambda o: (-o["frames_present"], o["first_frame"]))
    return objs


def _format_object_context(objs: list[dict]) -> str:
    """把窗内物体摘要格式化成注入 LLM 的 object_context（场景级；含包裹品牌/logo 提示）。"""
    if not objs:
        return ""
    lines = ["【画面中的物体（YOLO 检测，场景级，非人物身份）】"]
    for o in objs[:20]:
        direction = o["direction"]
        motion = ("基本静止" if direction in ("mostly_static", "unknown")
                  else f"移动({direction})，疑似被搬动/取走/放下")
        lines.append(
            f"- {o['label_cn']}({o['label']}) track#{o['track_id']}："
            f"frame#{o['first_frame']}@{o['first_ts']} → frame#{o['last_frame']}@{o['last_ts']}"
            f"（共{o['frames_present']}帧）；{motion}"
        )
    if len(objs) > 20:
        lines.append(f"  ... 其余 {len(objs) - 20} 个物体省略")
    lines.append(
        "说明：以上为画面中的**物体**（包裹/行李/车辆等场景线索，frame# 与人物 grounding 同坐标系），"
        "用于理解放下、取走、搬运、到达、离开等事件；**若疑似快递/包裹，请结合关键帧画面识别其品牌或 "
        "logo（如 Amazon / UPS / FedEx，OCR 读文字、你看图认 logo）**。请勿把物体当作人物身份。"
    )
    return "\n".join(lines)



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
        if identities.get(tid, {}).get("skipped"):
            continue  # 门控掉的 track（太短/太低质）不跑人脸
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
            category = q.get("category", "poor")
            clear = category == "clear"  # 只有 clear 才建档/当锚点（marginal/poor 只查不建）
            rec = {
                "score": q.get("det_score"),
                "quality": category,
                "quality_score": q.get("quality"),  # 连续质量分(0~1)→ 融合软性加权用
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
        "source_track_ids": [tid],
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
        "subject_conflict_split": ident.get("subject_conflict_split", False),
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
        local_subject = any(identities[t].get("local_subject") for t in tids)
        conflict_split = any(identities[t].get("subject_conflict_split") for t in tids)
        kind = "本视频本地subject" if local_subject else ("时间冲突拆分subject" if conflict_split else "同一人")
        attrs = [f"由{len(tids)}条轨迹合并({kind})"]
        if merge_routes:
            attrs.append("跨track印证：" + "+".join(route_cn.get(r, r) for r in merge_routes))
        people.append({
            "track_id": rep,
            "source_track_ids": sorted(tids),
            "box": rep_box,
            "subject_id": identities[rep].get("subject_id"),
            "decision": "local_stitched" if local_subject else ("conflict_split" if conflict_split else "hit"),
            "reused": False if (local_subject or conflict_split) else True,
            "trajectory": [list(c) for (_, c) in merged_centers],
            "reid": {"score": round(best_score, 4)} if best_score > 0 else None,
            "face": face,
            "gait": gait,
            "fused": fused,
            "merge_routes": merge_routes or None,
            "merge_agree": len(merge_routes) or None,
            "local_subject": local_subject or None,
            "subject_conflict_split": conflict_split or None,
            "attributes": attrs,
        })
    return people


def _stitch_orphans(
    tracks: dict[int, dict],
    identities: dict[int, dict],
    track_emb: dict[int, np.ndarray],
    thresh: float,
) -> None:
    """把灰区/低质孤立 track（subject_id 为空）并成同视频内本地主体（就地改 identities）。

    做法：
      1. 若已有 gallery subject：孤立 track 与最相近主体相似度 ≥ thresh 且时间不重叠，才并入。
      2. 剩余孤立 track 彼此之间用更保守的阈值做本地聚类；同一簇内 track 也不能时间重叠。

    注意：这里不写 gallery，只给事件报告/LLM 一个稳定本地称呼。这样低质 crop 不会污染长期库，
    但也不会在报告里显示成一堆 "track 17"。
    """
    def _norm(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def _overlap(a: int, b: int) -> bool:
        return not (tracks[a]["last"] < tracks[b]["first"] or tracks[b]["last"] < tracks[a]["first"])

    # 各主体的成员向量（来自已分配 subject_id 的 track）
    members: dict[int, list[np.ndarray]] = {}
    member_tids: dict[int, list[int]] = {}
    for tid, idn in identities.items():
        sid = idn.get("subject_id")
        if sid is not None and tid in track_emb:
            members.setdefault(sid, []).append(track_emb[tid])
            member_tids.setdefault(sid, []).append(tid)
    reps: dict[int, np.ndarray] = {sid: _norm(np.mean(vs, axis=0)) for sid, vs in members.items()}

    # 孤立 track：按首次出现时间顺序缝合
    orphans = [tid for tid, idn in identities.items()
               if idn.get("subject_id") is None and tid in track_emb]
    orphans.sort(key=lambda t: tracks[t]["first"])

    for tid in orphans:
        v = _norm(track_emb[tid])
        best_sid, best_sim = None, -1.0
        hit_thresh = thresh if identities[tid].get("quality_ok") else max(thresh, settings.event_local_stitch_thresh)
        for sid, rep in reps.items():
            if any(_overlap(tid, mt) for mt in member_tids.get(sid, [])):
                continue
            sim = float(np.dot(v, rep))
            if sim > best_sim:
                best_sid, best_sim = sid, sim
        if best_sid is not None and best_sim >= hit_thresh:
            idn = identities[tid]
            idn["subject_id"] = best_sid
            idn["decision"] = "stitched"
            idn["reused"] = True
            idn["stitch_score"] = round(best_sim, 4)
            if idn.get("score") is None:
                idn["score"] = round(best_sim, 4)
            # 并入代表，便于后续断片接力
            members[best_sid].append(track_emb[tid])
            member_tids.setdefault(best_sid, []).append(tid)
            reps[best_sid] = _norm(np.mean(members[best_sid], axis=0))

    # 仍无 subject 的低质/灰区 track：只在本视频内保守聚类，铸本地 subject_id，不污染 gallery。
    remaining = [tid for tid in orphans if identities[tid].get("subject_id") is None]
    if not remaining:
        return

    local_thresh = max(thresh, settings.event_local_stitch_thresh)
    clusters: list[dict] = []
    for tid in remaining:
        v = _norm(track_emb[tid])
        best_cluster, best_sim = None, -1.0
        for cluster in clusters:
            if any(_overlap(tid, mt) for mt in cluster["tids"]):
                continue
            sim = float(np.dot(v, cluster["rep"]))
            if sim > best_sim:
                best_cluster, best_sim = cluster, sim
        if best_cluster is not None and best_sim >= local_thresh:
            best_cluster["tids"].append(tid)
            best_cluster["vecs"].append(track_emb[tid])
            best_cluster["scores"].append(best_sim)
            best_cluster["rep"] = _norm(np.mean(best_cluster["vecs"], axis=0))
        else:
            clusters.append({"tids": [tid], "vecs": [track_emb[tid]], "scores": [], "rep": v})

    existing = [idn.get("subject_id") for idn in identities.values() if idn.get("subject_id") is not None]
    next_sid = (max(existing) + 1) if existing else 1
    for cluster in sorted(clusters, key=lambda c: tracks[min(c["tids"])]["first"]):
        tids = cluster["tids"]
        sid = next_sid
        next_sid += 1
        best_sim = round(max(cluster["scores"]), 4) if cluster["scores"] else None
        for tid in tids:
            idn = identities[tid]
            idn["subject_id"] = sid
            idn["local_subject"] = True
            idn["reused"] = False
            idn["cross_track_merged"] = len(tids) > 1
            idn["decision"] = "local_stitched" if len(tids) > 1 else "local"
            if best_sim is not None:
                idn["stitch_score"] = best_sim
                if idn.get("score") is None:
                    idn["score"] = best_sim


def _split_subject_time_conflicts(tracks: dict[int, dict], identities: dict[int, dict]) -> None:
    """拆开不可能属于同一人的 subject：同一 subject 下的 track 时间重叠则必须分成不同主体。

    ReID gallery 在远景小人/多人场景里可能把很多相似低质 crop 都 hit 到同一个 subject。
    但如果两条 track 的时间区间重叠，它们在同一时刻同时出现在画面里，就不可能是同一个人。
    这里用这个物理约束做兜底拆分，避免报告出现"主体#1 · 34条轨迹"。
    """
    def _overlap(a: int, b: int) -> bool:
        return not (tracks[a]["last"] < tracks[b]["first"] or tracks[b]["last"] < tracks[a]["first"])

    by_subject: dict[int, list[int]] = {}
    for tid, ident in identities.items():
        sid = ident.get("subject_id")
        if sid is not None and tid in tracks:
            by_subject.setdefault(int(sid), []).append(tid)

    existing = [int(sid) for sid in by_subject]
    next_sid = (max(existing) + 1) if existing else 1
    for sid, tids in list(by_subject.items()):
        if len(tids) < 2:
            continue
        tids.sort(key=lambda t: (tracks[t]["first"], tracks[t]["last"]))
        clusters: list[list[int]] = []
        for tid in tids:
            placed = False
            for cluster in clusters:
                if not any(_overlap(tid, other) for other in cluster):
                    cluster.append(tid)
                    placed = True
                    break
            if not placed:
                clusters.append([tid])
        if len(clusters) <= 1:
            continue

        for idx, cluster in enumerate(clusters):
            target_sid = sid if idx == 0 else next_sid
            if idx > 0:
                next_sid += 1
            for tid in cluster:
                ident = identities[tid]
                ident["subject_id"] = target_sid
                ident["subject_conflict_split"] = True
                ident["reused"] = False
                if ident.get("decision") == "hit":
                    ident["decision"] = "conflict_split"


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
