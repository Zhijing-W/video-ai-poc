"""响应体与服务层共享模型。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TargetAttribute(BaseModel):
    type: str | None = None
    value: str | None = None
    region: str | None = None


class TargetPlan(BaseModel):
    yolo_class: str | None = None
    attribute: TargetAttribute | None = None
    can_yolo_handle: bool = False
    summary: str | None = None


class Detection(BaseModel):
    label: str
    confidence: float
    box: list[float]
    track_id: int | None = None
    color: str | None = None
    color_zh: str | None = None
    color_source: str | None = None  # Phase 3 · Step 13：取色区域来源（pose_full/pose_shoulders/fallback_torso）


class YoloResult(BaseModel):
    model: str
    infer_ms: float
    img_w: int
    img_h: int
    detections: list[Detection] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)


class GateDecision(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    passed: bool = Field(alias="pass")
    reason: str
    priority: str
    signals: dict[str, Any] = Field(default_factory=dict)


class CruiseDecision(BaseModel):
    is_match: bool
    reason: str
    matched_boxes: list[list[float]] = Field(default_factory=list)


class MatchDecision(BaseModel):
    is_match: bool | None = None
    confidence: str | None = None
    target: str = ""
    reason: str = ""


class SubjectProfile(BaseModel):
    ref: str | None = None
    label: str | None = None
    box: list[float] = Field(default_factory=list)
    appearance: str | None = None
    attributes: list[str] = Field(default_factory=list)


class AnalyzeResult(BaseModel):
    scene: str = ""
    detected_objects: list[str] = Field(default_factory=list)
    subjects: list[SubjectProfile] = Field(default_factory=list)
    match: MatchDecision = Field(default_factory=MatchDecision)
    alert_level: str = "normal"
    notification: str = ""
