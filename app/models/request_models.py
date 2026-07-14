"""请求体模型。"""
from __future__ import annotations

from pydantic import BaseModel, Field

from .response_models import TargetPlan


class AnalyzeFrameRequest(BaseModel):
    image: str
    target: str | None = None
    reference_image: str | None = None
    gate_enabled: bool = True
    prev_counts: dict[str, int] | None = None
    since_last_llm_ms: int | None = None
    comparing: bool = False
    plan: TargetPlan | None = None
    last_llm_signature: str | None = None
    # Phase 3 · Step 12「三时钟解耦」：track 门控（按 track_id 复用结论，新主体才调 LLM）
    track_enabled: bool = False
    session_id: str | None = None


class DetectRequest(BaseModel):
    image: str
    conf: float | None = None


class TrackRequest(BaseModel):
    """`/track` 请求：有状态的逐轨迹检测（Phase 3 · Step 11）。"""

    image: str
    conf: float | None = None
    session_id: str = "default"  # 不同视频/摄像头用不同 id，跟踪状态互不串味
    reset: bool = False          # True 则在本帧之前先清空该 session 的轨迹状态


class TrackResetRequest(BaseModel):
    """`/track/reset` 请求：换视频 / 重新开始监控时清空跟踪状态。"""

    session_id: str = "default"


class CompileTargetRequest(BaseModel):
    target: str
    reference_image: str | None = None


class CruiseRequest(BaseModel):
    image: str
    plan: TargetPlan


class MonitorEntryRequest(BaseModel):
    seq: int
    ts: str | None = None
    level: str | None = None
    msg: str | None = None
    is_match: bool | None = None
    image: str | None = None
    result: dict | None = None


class MonitorSessionRequest(BaseModel):
    started_at: str | None = None
    ended_at: str | None = None
    target: str | None = None
    mode: str | None = None
    stats: dict | None = None
    summary: dict | None = None
    entries: list[MonitorEntryRequest] = Field(default_factory=list)


class SummarizeRequest(BaseModel):
    """`/summarize` 请求：把实时整段分析累积的逐帧事件归纳成末尾总结。"""

    events: list[dict] = Field(default_factory=list)


class IdentifyDetection(BaseModel):
    """`/identify` 里的单个待认目标（通常来自 `/track` 的检测结果）。"""

    box: list[float]
    track_id: int | None = None
    label: str | None = None
    attributes: list[str] | None = None


class IdentifyRequest(BaseModel):
    """`/identify` 请求：主体记忆向量库查库认人 / 开放集登记（Phase 3 · Step 14）。

    传整帧 image + 若干目标框，逐个提 ReID 指纹查库：命中即复用档案、未命中按需登记。
    既可传 `detections` 批量，也可只给单个 `box` 便捷认一个目标。
    """

    image: str
    session_id: str = "default"
    detections: list[IdentifyDetection] = Field(default_factory=list)
    box: list[float] | None = None       # 便捷：只认单个目标时直接给 box
    track_id: int | None = None
    label: str | None = None
    auto_enroll: bool = True             # 未命中的新主体是否自动建档
    reset: bool = False                  # True 则在本次之前先清空该 session 的记忆库


class GalleryResetRequest(BaseModel):
    """`/gallery/reset` 请求：换视频 / 重新开始时清空主体记忆。"""

    session_id: str = "default"


class FusionObservation(BaseModel):
    """`/fusion/observe` 里"某 track 在某帧的一次识别观测"（Phase 3 · Step 15 / 3.5）。"""

    track_id: int
    frame_idx: int
    box: list[float] | None = None
    quality: dict | None = None        # reid.assess_quality 输出（清晰度/面积/长宽比）
    reid_subject: int | None = None    # gallery 这一帧判出的 subject_id
    reid_decision: str | None = None   # hit / new / grey
    reid_score: float = 0.0            # gallery 余弦分
    color: str | None = None           # 主色（颜色一致线索）


class FusionObserveRequest(BaseModel):
    """`/fusion/observe` 请求：累积多帧观测做最佳帧投票 + 多线索融合。"""

    session_id: str = "default"
    observations: list[FusionObservation] = Field(default_factory=list)
    reset: bool = False                # True 则在本批之前先清空该 session 的融合状态


class FusionResetRequest(BaseModel):
    """`/fusion/reset` 请求：换视频 / 重新开始时清空融合缓冲。"""

    session_id: str = "default"
