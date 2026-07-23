"""集中读取环境变量并暴露运行时路径配置。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR") or BASE_DIR / "data").expanduser()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR") or BASE_DIR / "out").expanduser()
GALLERY_DIR = Path(os.getenv("GALLERY_DIR") or BASE_DIR / "gallery").expanduser()
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _shared_asset(name: str) -> str:
    candidate = BASE_DIR / "models" / name
    return str(candidate) if candidate.exists() else name


def _get(name: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(
            f"缺少必需的环境变量 {name}。请复制 .env.example 为 .env 并填写。"
        )
    return value


@dataclass
class Settings:
    azure_openai_endpoint: str | None = _get("AZURE_OPENAI_ENDPOINT")
    azure_openai_api_key: str | None = _get("AZURE_OPENAI_API_KEY")
    azure_openai_deployment: str | None = _get("AZURE_OPENAI_DEPLOYMENT")
    azure_openai_api_version: str = _get("AZURE_OPENAI_API_VERSION", "2024-10-21")

    frame_interval_seconds: int = int(_get("FRAME_INTERVAL_SECONDS", "5"))
    max_frames: int = int(_get("MAX_FRAMES", "8"))
    frame_width: int = int(_get("FRAME_WIDTH", "768"))

    storage_account_name: str | None = _get("AZURE_STORAGE_ACCOUNT_NAME")
    storage_container_name: str = _get(
        "AZURE_STORAGE_CONTAINER_NAME", "video-understanding-poc"
    )
    storage_connection_string: str | None = _get("AZURE_STORAGE_CONNECTION_STRING")

    ffmpeg_path: str = _get("FFMPEG_PATH") or "ffmpeg"

    yolo_model: str = _get("YOLO_MODEL", _shared_asset("yolov8m.pt"))
    yolo_conf: float = float(_get("YOLO_CONF", "0.4"))

    # 多目标跟踪 MOT（Phase 3 · Step 11 / Phase 4 升级）：可切换 ByteTrack / BoT-SORT / BoT-SORT+ReID。
    # 给每个目标分配跨帧稳定的 track_id，使"识别一次、整条轨迹复用"成为可能。
    track_backend: str = _get("TRACK_BACKEND", "botsort_reid").strip().lower()  # bytetrack | botsort | botsort_reid
    track_conf: float = float(_get("TRACK_CONF", "0.1"))            # 喂给跟踪器的低检测阈值（让 ByteTrack 用低分框做二次关联）
    track_buffer: int = int(_get("TRACK_BUFFER", "30"))            # 轨迹丢失后保留的帧数（越大越抗短遮挡，但更易 ID 漂移）
    track_high_thresh: float = float(_get("TRACK_HIGH_THRESH", "0.25"))   # 一段匹配高分阈值
    track_low_thresh: float = float(_get("TRACK_LOW_THRESH", "0.1"))      # 二段匹配低分阈值
    new_track_thresh: float = float(_get("NEW_TRACK_THRESH", "0.25"))     # 高于此分且无匹配才新建轨迹
    track_match_thresh: float = float(_get("TRACK_MATCH_THRESH", "0.8"))  # 关联相似度（IoU/cost）阈值
    track_fuse_score: bool = _get("TRACK_FUSE_SCORE", "true").strip().lower() in {"1", "true", "yes", "on"}
    track_gmc_method: str = _get("TRACK_GMC_METHOD", "sparseOptFlow")  # BoT-SORT 全局运动补偿：sparseOptFlow|orb|sift|ecc|none
    track_proximity_thresh: float = float(_get("TRACK_PROXIMITY_THRESH", "0.5"))  # BoT-SORT ReID 先验 IoU 门
    track_appearance_thresh: float = float(_get("TRACK_APPEARANCE_THRESH", "0.8"))  # BoT-SORT ReID 外观相似门

    # 细粒度感知（Phase 3 · Step 13）：YOLO-Pose 派生躯干区取色，修 Phase 2 颜色误判。
    # 仅在画面有人时跑；不可用/几何反常自动回落到写死比例 torso（不劣于原行为）。
    pose_color: bool = _get("POSE_COLOR", "true").strip().lower() in {"1", "true", "yes", "on"}
    pose_model: str = _get("POSE_MODEL", _shared_asset("yolov8n-pose.pt"))   # 与检测的 yolov8m 独立的姿态模型
    pose_conf: float = float(_get("POSE_CONF", "0.3"))         # Pose 人体检测置信度
    pose_kpt_conf: float = float(_get("POSE_KPT_CONF", "0.3")) # 单个关键点的可信阈值（低于则视为不可见）

    # 主体记忆 / ReID 向量库（Phase 3 · Step 14）：认过一次就记住、命中即复用、不调 LLM。
    # backend: auto 自动择优（osnet→resnet50→coarse）；也可固定为某一档。
    reid_backend: str = _get("REID_BACKEND", "auto")
    reid_osnet_weights: str = _get("REID_OSNET_WEIGHTS", "osnet_ain_x1_0_msmt17.pt")  # boxmot OSNet 域泛化权重
    reid_device: str = _get("REID_DEVICE", "auto")  # auto/cuda/cpu：ReID 推理设备（auto=有 GPU 用 GPU）
    # 余弦判定阈值（注意：不同 backend 的相似度分布不同，换 backend 需重调）。
    reid_hit_thresh: float = float(_get("REID_HIT_THRESH", "0.6"))     # ≥ 此分 → 认出已知主体
    reid_new_thresh: float = float(_get("REID_NEW_THRESH", "0.4"))     # < 此分 → 判为新主体（开放集登记）
    reid_low_quality_hit_thresh: float = float(_get("REID_LOW_QUALITY_HIT_THRESH", "0.88"))  # 低质 crop 复用已有主体的更高门槛
    reid_decision_top_k: int = int(_get("REID_DECISION_TOP_K", "30"))  # 检索候选数，用于 top-k 一致性判断
    reid_consistency_enabled: bool = _get("REID_CONSISTENCY_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    reid_vote_score_thresh: float = float(_get("REID_VOTE_SCORE_THRESH", "0.55"))  # 高相似候选才参与投票
    reid_consistency_ratio: float = float(_get("REID_CONSISTENCY_RATIO", "0.5"))   # top subject 票占比下限
    reid_top1_margin: float = float(_get("REID_TOP1_MARGIN", "0.03"))              # top1 与 runner-up 间隔过小则 grey
    reid_max_shots: int = int(_get("REID_MAX_SHOTS", "8"))            # 每主体最多保留的 shot 数（multi-shot）
    # 质量门控：糊/太小/长宽比异常的 crop 不入库（防止污染向量库）。
    reid_min_area: int = int(_get("REID_MIN_AREA", "1600"))          # 最小像素面积（约 40×40）
    reid_min_blur_var: float = float(_get("REID_MIN_BLUR_VAR", "20.0"))  # 拉普拉斯方差下限（越小越糊）
    reid_min_aspect: float = float(_get("REID_MIN_ASPECT", "0.8"))   # 高/宽 下限
    reid_max_aspect: float = float(_get("REID_MAX_ASPECT", "4.0"))   # 高/宽 上限
    # 负缓存：记住"查过、确认不在库"的查询向量，相似查询直接短路。
    reid_neg_cache_thresh: float = float(_get("REID_NEG_CACHE_THRESH", "0.92"))
    reid_neg_cache_size: int = int(_get("REID_NEG_CACHE_SIZE", "256"))

    # 多线索融合 + 最佳帧投票（Phase 3 · Step 15 / 3.5）：一条 track 攒多帧证据再定身份。
    fusion_buffer_size: int = int(_get("FUSION_BUFFER_SIZE", "12"))   # 每 track 保留的观测帧数
    fusion_ref_area: int = int(_get("FUSION_REF_AREA", "20000"))      # 面积归一化基准（最佳帧"大小"项）
    fusion_resolve_thresh: float = float(_get("FUSION_RESOLVE_THRESH", "0.55"))  # 融合分≥此→采信身份
    fusion_continuity_bonus: float = float(_get("FUSION_CONTINUITY_BONUS", "0.15"))  # 时序黏滞先验（防抖）
    fusion_color_penalty: float = float(_get("FUSION_COLOR_PENALTY", "0.5"))     # 颜色不一致帧的票权折扣
    fusion_motion_sigma: float = float(_get("FUSION_MOTION_SIGMA", "1.5"))       # 运动连续性高斯宽度（×框对角线）
    # 多线索权重（投票/ReID/颜色/运动/人脸；人脸为 Step 17 占位，默认 0）
    fusion_w_vote: float = float(_get("FUSION_W_VOTE", "0.45"))
    fusion_w_reid: float = float(_get("FUSION_W_REID", "0.30"))
    fusion_w_color: float = float(_get("FUSION_W_COLOR", "0.10"))
    fusion_w_motion: float = float(_get("FUSION_W_MOTION", "0.15"))
    fusion_w_face: float = float(_get("FUSION_W_FACE", "0.0"))

    # 人脸识别分支（Phase 4 · Step 20）：InsightFace（SCRFD 检测 + ArcFace 识别）。
    # 只在每条 track 的最佳帧稀疏调用；糊脸/侧脸/小脸由质量门控降权或拒用（攻"人脸模糊"）。
    face_backend: str = _get("FACE_BACKEND", "insightface")
    face_device: str = _get("FACE_DEVICE", "auto")  # auto/cuda/cpu：人脸(InsightFace/AdaFace)推理设备
    face_model: str = _get("FACE_MODEL", "buffalo_l")            # InsightFace 模型包
    face_det_size: int = int(_get("FACE_DET_SIZE", "640"))       # 检测输入边长（小→快、精度略降）
    face_min_det_score: float = float(_get("FACE_MIN_DET_SCORE", "0.5"))   # 低于此检测分不可信
    face_min_size: int = int(_get("FACE_MIN_SIZE", "28"))        # 人脸框最小边（像素），太小不入库
    face_min_frontalness: float = float(_get("FACE_MIN_FRONTALNESS", "0.45"))  # 正脸度下限（侧脸降权）
    face_min_blur_var: float = float(_get("FACE_MIN_BLUR_VAR", "15.0"))    # 清晰度下限（拉普拉斯方差）
    face_ref_area: int = int(_get("FACE_REF_AREA", "10000"))     # 面积归一化基准（约 100×100）
    face_assoc_min_contain: float = float(_get("FACE_ASSOC_MIN_CONTAIN", "0.6"))  # 人脸被人体框包含度阈值
    face_assoc_max_head_y_ratio: float = float(_get("FACE_ASSOC_MAX_HEAD_Y_RATIO", "0.48"))
    face_assoc_ambiguity_margin: float = float(_get("FACE_ASSOC_AMBIGUITY_MARGIN", "0.08"))
    # 人脸库比对阈值（ArcFace 余弦分布与人形 OSNet 不同，单独配）：≥hit 认人，<new 判新主体。
    face_hit_thresh: float = float(_get("FACE_HIT_THRESH", "0.45"))   # 同人 ArcFace 余弦通常 >0.4
    face_new_thresh: float = float(_get("FACE_NEW_THRESH", "0.30"))   # 陌生人通常 <0.3

    # 人脸质量分级（对齐客户「人脸过滤」逻辑：主看**模糊 + 角度**；分 clear / marginal / poor）。
    # 角度：偏航 yaw 与俯仰 pitch。监控摄像头俯拍→低头(pitch<0)比抬头(pitch>0)更不容忍（眉眼被遮）。
    face_yaw_clear: float = float(_get("FACE_YAW_CLEAR", "25"))       # |yaw|≤ 此值算清晰正脸（度）
    face_yaw_max: float = float(_get("FACE_YAW_MAX", "80"))           # |yaw|≥ 此值判不合格（大侧脸）
    face_pitch_clear: float = float(_get("FACE_PITCH_CLEAR", "20"))   # |pitch|≤ 此值算清晰
    face_pitch_down_max: float = float(_get("FACE_PITCH_DOWN_MAX", "35"))  # 低头超此判不合格（严）
    face_pitch_up_max: float = float(_get("FACE_PITCH_UP_MAX", "50"))      # 抬头超此判不合格（宽）
    face_blur_clear_var: float = float(_get("FACE_BLUR_CLEAR_VAR", "60"))  # 拉普拉斯方差≥ 此值算清晰
    # CR-FIQA仅作诊断；产品质量分桶、路由和融合权重由可解释规则指标决定。
    # 官方CR-FIQA代码为CC BY-NC 4.0，商业使用前需完成许可确认。
    face_fiqa_backend: str = _get("FACE_FIQA_BACKEND", "off").strip().lower()
    face_fiqa_root: str = _get(
        "FACE_FIQA_ROOT",
        str(BASE_DIR / "models" / "CR-FIQA" / "source"),
    )
    face_fiqa_weights: str = _get(
        "FACE_FIQA_WEIGHTS",
        str(BASE_DIR / "models" / "CR-FIQA" / "32572backbone.pth"),
    )
    face_fiqa_arch: str = _get("FACE_FIQA_ARCH", "iresnet50").strip().lower()
    face_fiqa_device: str = _get("FACE_FIQA_DEVICE", "auto").strip().lower()
    # CR-FIQA回归头输出不是通用概率；以下阈值只生成诊断，需独立校准后才能升级为门控。
    face_fiqa_poor_thresh: float = float(_get("FACE_FIQA_POOR_THRESH", "0.3"))
    face_fiqa_clear_thresh: float = float(_get("FACE_FIQA_CLEAR_THRESH", "0.6"))

    # 攻"人脸模糊"的可插拔进阶武器（Phase 4 · §3.8 / Step 27b）。默认全开；测试对比时可逐个关。
    # ① 3D-68 几何 cue：打开 buffalo_l 自带的 1k3d68 landmark，用 3D 面部几何（颧骨/鼻梁/下巴
    #    等结构）做额外身份线索——纹理糊但几何还在，对姿态+中度模糊鲁棒。
    face_3d_cue: bool = _get("FACE_3D_CUE", "true").strip().lower() in {"1", "true", "yes", "on"}
    # ② 人脸超分：off 或任意已注册后端名称；内置 gfpgan，其他算法通过注册表接入。
    face_superres: str = _get("FACE_SUPERRES", "gfpgan").strip().lower()
    face_recoverable_min_size: int = int(_get("FACE_RECOVERABLE_MIN_SIZE", "20"))
    face_superres_max_size: int = int(
        _get("FACE_SUPERRES_MAX_SIZE", _get("FACE_SUPERRES_MIN_SIZE", "90"))
    )
    face_superres_min_size: int = face_superres_max_size  # deprecated compatibility alias
    face_candidate_top_k: int = int(_get("FACE_CANDIDATE_TOP_K", "3"))
    face_candidate_min_gap_frames: int = int(_get("FACE_CANDIDATE_MIN_GAP_FRAMES", "2"))
    face_track_consistency_thresh: float = float(_get("FACE_TRACK_CONSISTENCY_THRESH", "0.82"))
    face_gfpgan_weights: str = _get("FACE_GFPGAN_WEIGHTS", "")               # 留空自动下载/默认路径
    face_codeformer_weights: str = _get("FACE_CODEFORMER_WEIGHTS", "")
    face_codeformer_fidelity: float = float(_get("FACE_CODEFORMER_FIDELITY", "1.0"))
    face_realesrgan_x2plus_weights: str = _get("FACE_REALESRGAN_X2PLUS_WEIGHTS", "")
    # ③ AdaFace：质量自适应人脸识别后端（低清脸更强）。arcface / adaface（默认 adaface，最强）。
    face_rec_backend: str = _get("FACE_REC_BACKEND", "adaface").strip().lower()
    face_adaface_root: str = _get("FACE_ADAFACE_ROOT", str(BASE_DIR / "models" / "AdaFace"))
    face_adaface_arch: str = _get("FACE_ADAFACE_ARCH", "ir_101")
    face_adaface_weights: str = _get(
        "FACE_ADAFACE_WEIGHTS",
        str(BASE_DIR / "models" / "AdaFace" / "pretrained" / "pretrained_model" / "model.pt"),
    )

    # 多帧事件理解（Phase 4 · Step 23 / 3.4，本阶段灵魂）：多帧关键帧 + 身份上下文 → 跨帧事件叙述。
    # 模型名可配置：默认用现有 AZURE_OPENAI_DEPLOYMENT；以后指向 gpt-4.1/更强只改这一项。
    event_llm_deployment: str | None = _get("EVENT_LLM_DEPLOYMENT")   # 留空则回退主部署
    event_llm_max_tokens: int = int(_get("EVENT_LLM_MAX_TOKENS", "1500"))
    event_llm_max_retries: int = int(_get("EVENT_LLM_MAX_RETRIES", "5"))  # 429 限流时退避重试次数
    event_frame_detail: str = _get("EVENT_FRAME_DETAIL", "low")       # low 省 token / high 看细节

    # 选帧②：事件驱动关键帧选择（Phase 4 · Step 25 / 3.3）——喂 LLM 前按"事件"砍图片数。
    keyframe_max: int = int(_get("KEYFRAME_MAX", "24"))            # 喂 LLM 的关键帧上限
    keyframe_context: int = int(_get("KEYFRAME_CONTEXT", "1"))     # 事件前后各留几帧上下文
    keyframe_dedup_diff: float = float(_get("KEYFRAME_DEDUP_DIFF", "0.06"))  # 低于此签名差异视为"太像"去重

    # 流式事件分窗（Phase 4 · Step 24）：窗 = 一次 LLM 调用。窗按"活动段 + 时长上限"切。
    # 时长上限是给"长连续事件"准备的：超过则冲刷开新窗，否则长事件被压成单窗、关键帧严重欠采样。
    event_window_max_seconds: float = float(_get("EVENT_WINDOW_MAX_SECONDS", "30"))

    # Track 级门控（认人前先筛掉不值得认的 track，整条省下 reid/face/步态，并防垃圾 track 污染库）：
    # 存活帧数 < 此值 视为检测抖动/昙花一现的假 track；最佳质量 < 此值 视为全程太糊/太小。
    # 二者任一不达标 → 整条 track 跳过身份提取（仍保留在事件里，只是身份留空、不入库）。设 0 关闭对应门。
    track_min_frames: int = int(_get("TRACK_MIN_FRAMES", "3"))       # 至少出现几帧才认人
    track_min_quality: float = float(_get("TRACK_MIN_QUALITY", "0.0"))  # 最佳帧质量下限（0=不按质量筛）

    # 跨窗整段事件总结（Phase 4 · Step E）：所有窗逐窗理解完后，再纯文本把多窗串成一段连贯故事。
    # 便宜（仅文本一次调用）；dry-run 自动跳过。设 0/false 关闭。
    event_overall_summary: bool = _get("EVENT_OVERALL_SUMMARY", "1") not in ("0", "false", "False", "")

    # 同视频内"轨迹缝合"（Phase 4 · Step 27）：把灰区孤立 track 并进最相近的已建主体。
    # 同一段视频里 tracker 把一个连续的人断成几段，先验强，可比 gallery 跨摄像头阈值更大胆地并。
    # 设 0 关闭缝合。阈值越低越敢并（省"一人两条"），但过低会误并不同人。
    event_stitch_thresh: float = float(_get("EVENT_STITCH_THRESH", "0.45"))
    # 低质 track 无法入长期 gallery 时，只铸"本视频本地 subject"；人群远景里外观相似，阈值必须更保守。
    event_local_stitch_thresh: float = float(_get("EVENT_LOCAL_STITCH_THRESH", "0.82"))

    # 三路身份融合（Phase 4 · A 汇聚）：人脸 + 人形 ReID + 步态 按质量加权 → 一个统一身份置信度。
    # 质量自适应：清晰脸权重高、糊脸降权退人形/步态；多路一致再加成。设为各路的相对权重。
    identity_w_face: float = float(_get("IDENTITY_W_FACE", "0.5"))    # 人脸（清晰时最强）
    identity_w_body: float = float(_get("IDENTITY_W_BODY", "0.3"))    # 人形 ReID
    identity_w_gait: float = float(_get("IDENTITY_W_GAIT", "0.2"))    # 步态（无脸/背身兜底）
    identity_face_blurry_factor: float = float(_get("IDENTITY_FACE_BLURRY_FACTOR", "0.35"))  # 糊脸权重折扣（仅回退用）
    # 软性连续加权（文献最优，SER-FIQ/CR-FIQA 风格）：人脸有效权重 = w_face×(floor + (1-floor)×质量分)。
    # 用连续质量分平滑降权，中等/微糊脸不再被一刀切压死；floor 给 poor 脸保底贡献，不完全归零。
    identity_face_quality_floor: float = float(_get("IDENTITY_FACE_QUALITY_FLOOR", "0.3"))
    identity_agree_bonus: float = float(_get("IDENTITY_AGREE_BONUS", "0.15"))  # 多路一致命中的加成
    identity_resolve_thresh: float = float(_get("IDENTITY_RESOLVE_THRESH", "0.5"))  # ≥此置信视为"已确认"

    # 步态识别分支（Phase 4 · Step 27）：SkeletonGait++（OpenGait，GREW 权重）。本机纯 CPU 跑（慢，
    # 效果与 GPU 相同）；上云换 device='cuda'。OpenGait 仓库与 726MB 权重在 git 仓库外，路径可配。
    gait_enabled: bool = _get("GAIT_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    gait_opengait_root: str = _get("GAIT_OPENGAIT_ROOT", str(BASE_DIR / "models" / "OpenGait"))
    gait_ckpt: str = _get(
        "GAIT_CKPT",
        str(
            BASE_DIR
            / "models"
            / "OpenGait"
            / "checkpoints"
            / "GREW"
            / "SkeletonGaitPP"
            / "SkeletonGaitPP"
            / "checkpoints"
            / "SkeletonGaitPP-180000.pt"
        ),
    )
    gait_seg_model: str = _get("GAIT_SEG_MODEL", _shared_asset("yolov8m-seg.pt"))   # 剪影分割（ultralytics 实例分割）
    gait_min_frames: int = int(_get("GAIT_MIN_FRAMES", "10"))        # 一条 track 至少几帧才算步态（帧太少不可靠）
    gait_device: str = _get("GAIT_DEVICE", "cpu")                    # 本地 cpu；上云改 cuda

    # 场景文字 OCR（Phase 4 · Step 29，LANE D 子能力）：读画面时间戳/车牌/包裹单号等**场景级**文字。
    # 输出 scene_context 只在 LLM 阶段与人物身份并列注入，**不进** gallery/subject_id/三路融合。
    # 默认引擎 RapidOCR（onnxruntime，内置 PP-OCRv4，CPU 友好）；可切 paddleocr（上云追更高精度）。
    ocr_enabled: bool = _get("OCR_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    ocr_backend: str = _get("OCR_BACKEND", "rapidocr").strip().lower()  # rapidocr | paddleocr
    ocr_lang: str = _get("OCR_LANG", "ch")                  # paddleocr 用；ch=中英混合
    ocr_min_conf: float = float(_get("OCR_MIN_CONF", "0.5"))   # 低于此识别置信度的文字丢弃
    # 归一化 ROI [x1,y1,x2,y2]（0~1），只在画面该区域跑 OCR（如时间戳常在角落）。环境变量
    # OCR_ROI 形如 "0.6,0.0,1.0,0.12"；留空=全图。
    ocr_roi: list[float] | None = (
        [float(x) for x in _get("OCR_ROI", "").split(",")]
        if len(_get("OCR_ROI", "").split(",")) == 4 and _get("OCR_ROI", "")
        else None
    )

    # 物体/包裹检测（Phase 4 · Step 30，LANE D 子能力）：把非 person 目标(包裹/行李/车辆)从 YOLO
    # 结果里"捡回来"（管线本只留 person），记录跨帧轨迹→object_context 与身份**并列**喂 LLM。
    # **场景级**，同 OCR：不进 subject_id/gallery/三路融合。COCO 无"快递箱"类，用 bag/suitcase 近似；
    # 真正"这是快递包裹 + 什么品牌"靠多模态 LLM 看关键帧识别（OCR 读文字、LLM 认 logo）。
    object_detect: bool = _get("OBJECT_DETECT", "false").strip().lower() in {"1", "true", "yes", "on"}
    object_classes: str = _get(
        "OBJECT_CLASSES",
        "backpack,handbag,suitcase,car,truck,bus,motorcycle,bicycle",
    )
    object_min_conf: float = float(_get("OBJECT_MIN_CONF", "0.35"))  # 低于此检测分的物体不计入
    object_min_frames: int = int(_get("OBJECT_MIN_FRAMES", "2"))  # 窗内出现帧数 < 此值的物体丢弃（抗 1 帧误检/ID 跳变）

    def __post_init__(self) -> None:
        if not (
            0 < self.face_recoverable_min_size
            <= self.face_min_size
            <= self.face_superres_max_size
        ):
            raise ValueError(
                "人脸尺寸配置必须满足 0 < FACE_RECOVERABLE_MIN_SIZE "
                "<= FACE_MIN_SIZE <= FACE_SUPERRES_MAX_SIZE"
            )
        if self.face_candidate_top_k < 1:
            raise ValueError("FACE_CANDIDATE_TOP_K 必须至少为 1")
        if self.face_candidate_min_gap_frames < 1:
            raise ValueError("FACE_CANDIDATE_MIN_GAP_FRAMES 必须至少为 1")
        if not 0.0 <= self.face_codeformer_fidelity <= 1.0:
            raise ValueError("FACE_CODEFORMER_FIDELITY 必须在 [0, 1] 范围内")

    def object_class_set(self) -> set[str]:
        return {c.strip() for c in self.object_classes.split(",") if c.strip()}

    # 智能抽帧（Phase 2 · Step 7）：场景突变 OR 定时兜底
    smart_frames: bool = _get("SMART_FRAMES", "true").strip().lower() in {"1", "true", "yes", "on"}
    scene_threshold: float = float(_get("SCENE_THRESHOLD", "0.4"))
    fallback_interval_seconds: int = int(_get("FALLBACK_INTERVAL_SECONDS", "30"))

    gate_key_classes: str = _get(
        "GATE_KEY_CLASSES",
        "person,car,truck,bus,motorcycle,bicycle,dog,cat,backpack,handbag,suitcase,knife,cell phone",
    )
    gate_cooldown_ms: int = int(_get("GATE_COOLDOWN_MS", "3000"))
    gate_heartbeat_ms: int = int(_get("GATE_HEARTBEAT_MS", "30000"))

    def gate_key_class_set(self) -> set[str]:
        return {item.strip() for item in self.gate_key_classes.split(",") if item.strip()}

    def override(self, **kwargs):
        """临时覆盖若干配置项（请求期内生效，退出即恢复）。供"本次请求覆盖"的轻量设置面板用。

        只接受 Settings 已有的属性；None 值忽略（表示前端没传、用默认）。返回一个上下文管理器。
        """
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            saved = {}
            try:
                for k, v in kwargs.items():
                    if v is None or not hasattr(self, k):
                        continue
                    saved[k] = getattr(self, k)
                    setattr(self, k, v)
                yield self
            finally:
                for k, v in saved.items():
                    setattr(self, k, v)

        return _ctx()

    def require_openai(self) -> None:
        missing = [
            name
            for name, value in {
                "AZURE_OPENAI_ENDPOINT": self.azure_openai_endpoint,
                "AZURE_OPENAI_API_KEY": self.azure_openai_api_key,
                "AZURE_OPENAI_DEPLOYMENT": self.azure_openai_deployment,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Azure OpenAI 配置不完整，缺少："
                + ", ".join(missing)
                + "。请在 .env 中填写。"
            )


settings = Settings()

__all__ = [
    "ALLOWED_VIDEO_SUFFIXES",
    "BASE_DIR",
    "DATA_DIR",
    "GALLERY_DIR",
    "OUTPUT_DIR",
    "STATIC_DIR",
    "Settings",
    "TEMPLATES_DIR",
    "settings",
]
