"""集中读取环境变量并暴露运行时路径配置。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
MONITOR_DIR = DATA_DIR / "monitor_sessions"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


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

    yolo_model: str = _get("YOLO_MODEL", "yolov8m.pt")
    yolo_conf: float = float(_get("YOLO_CONF", "0.4"))

    # 多目标跟踪 MOT（Phase 3 · Step 11）：ultralytics 内置 ByteTrack。
    # 给每个目标分配跨帧稳定的 track_id，使"识别一次、整条轨迹复用"成为可能。
    track_conf: float = float(_get("TRACK_CONF", "0.1"))            # 喂给跟踪器的低检测阈值（让 ByteTrack 用低分框做二次关联）
    track_buffer: int = int(_get("TRACK_BUFFER", "30"))            # 轨迹丢失后保留的帧数（越大越抗短遮挡，但更易 ID 漂移）
    track_high_thresh: float = float(_get("TRACK_HIGH_THRESH", "0.25"))   # 一段匹配高分阈值
    track_low_thresh: float = float(_get("TRACK_LOW_THRESH", "0.1"))      # 二段匹配低分阈值
    new_track_thresh: float = float(_get("NEW_TRACK_THRESH", "0.25"))     # 高于此分且无匹配才新建轨迹
    track_match_thresh: float = float(_get("TRACK_MATCH_THRESH", "0.8"))  # 关联相似度（IoU/cost）阈值
    track_fuse_score: bool = _get("TRACK_FUSE_SCORE", "true").strip().lower() in {"1", "true", "yes", "on"}

    # 细粒度感知（Phase 3 · Step 13）：YOLO-Pose 派生躯干区取色，修 Phase 2 颜色误判。
    # 仅在画面有人时跑；不可用/几何反常自动回落到写死比例 torso（不劣于原行为）。
    pose_color: bool = _get("POSE_COLOR", "true").strip().lower() in {"1", "true", "yes", "on"}
    pose_model: str = _get("POSE_MODEL", "yolov8n-pose.pt")   # 与检测的 yolov8m 独立的姿态模型
    pose_conf: float = float(_get("POSE_CONF", "0.3"))         # Pose 人体检测置信度
    pose_kpt_conf: float = float(_get("POSE_KPT_CONF", "0.3")) # 单个关键点的可信阈值（低于则视为不可见）

    # 主体记忆 / ReID 向量库（Phase 3 · Step 14）：认过一次就记住、命中即复用、不调 LLM。
    # backend: auto 自动择优（osnet→resnet50→coarse）；也可固定为某一档。
    reid_backend: str = _get("REID_BACKEND", "auto")
    reid_osnet_weights: str = _get("REID_OSNET_WEIGHTS", "osnet_ain_x1_0_msmt17.pt")  # boxmot OSNet 域泛化权重
    # 余弦判定阈值（注意：不同 backend 的相似度分布不同，换 backend 需重调）。
    reid_hit_thresh: float = float(_get("REID_HIT_THRESH", "0.6"))     # ≥ 此分 → 认出已知主体
    reid_new_thresh: float = float(_get("REID_NEW_THRESH", "0.4"))     # < 此分 → 判为新主体（开放集登记）
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
    face_model: str = _get("FACE_MODEL", "buffalo_l")            # InsightFace 模型包
    face_det_size: int = int(_get("FACE_DET_SIZE", "640"))       # 检测输入边长（小→快、精度略降）
    face_min_det_score: float = float(_get("FACE_MIN_DET_SCORE", "0.5"))   # 低于此检测分不可信
    face_min_size: int = int(_get("FACE_MIN_SIZE", "28"))        # 人脸框最小边（像素），太小不入库
    face_min_frontalness: float = float(_get("FACE_MIN_FRONTALNESS", "0.45"))  # 正脸度下限（侧脸降权）
    face_min_blur_var: float = float(_get("FACE_MIN_BLUR_VAR", "15.0"))    # 清晰度下限（拉普拉斯方差）
    face_ref_area: int = int(_get("FACE_REF_AREA", "10000"))     # 面积归一化基准（约 100×100）
    face_assoc_min_contain: float = float(_get("FACE_ASSOC_MIN_CONTAIN", "0.6"))  # 人脸被人体框包含度阈值

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

    # 同视频内"轨迹缝合"（Phase 4 · Step 27）：把灰区孤立 track 并进最相近的已建主体。
    # 同一段视频里 ByteTrack 把一个连续的人断成几段，先验强，可比 gallery 跨摄像头阈值更大胆地并。
    # 设 0 关闭缝合。阈值越低越敢并（省"一人两条"），但过低会误并不同人。
    event_stitch_thresh: float = float(_get("EVENT_STITCH_THRESH", "0.45"))

    # 跨窗整段事件总结（Phase 4 · E）：所有事件窗理解完后，把各窗叙述 + 身份名册做一次**纯文本**整合，
    # 串成整段视频的连贯事件故事（靠 ReID 身份跨窗关联同一人）。纯文本调用、便宜；dry-run 跳过。
    event_overall_summary: bool = _get("EVENT_OVERALL_SUMMARY", "true").strip().lower() in {"1", "true", "yes", "on"}

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
    "JOBS_DIR",
    "MONITOR_DIR",
    "STATIC_DIR",
    "Settings",
    "TEMPLATES_DIR",
    "settings",
]
