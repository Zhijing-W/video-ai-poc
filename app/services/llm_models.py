"""Backend-owned model aliases and deterministic routing for the PoC."""
from __future__ import annotations

from dataclasses import dataclass

from ..core.config import settings


@dataclass(frozen=True)
class ModelOption:
    alias: str
    label: str
    model: str
    deployment: str | None
    description: str

    def public_dict(self) -> dict:
        return {
            "alias": self.alias,
            "label": self.label,
            "model": self.model,
            "description": self.description,
            "available": bool(self.deployment),
        }


@dataclass(frozen=True)
class ModelSelection:
    task: str
    requested: str
    selected: str
    model: str
    deployment: str
    reason: str

    def public_dict(self) -> dict:
        return {
            "task": self.task,
            "requested": self.requested,
            "selected": self.selected,
            "model": self.model,
            "reason": self.reason,
        }


def _options() -> dict[str, ModelOption]:
    return {
        "gpt-4.1": ModelOption(
            alias="gpt-4.1",
            label="GPT-4.1（质量优先）",
            model="gpt-4.1",
            deployment=settings.foundry_analysis_deployment,
            description="多帧视觉分析和复杂证据推理。",
        ),
        "gpt-4.1-mini": ModelOption(
            alias="gpt-4.1-mini",
            label="GPT-4.1 mini（低延迟）",
            model="gpt-4.1-mini",
            deployment=settings.foundry_chat_deployment,
            description="低成本问答，也可用于快速分析。",
        ),
    }


def model_catalog() -> dict:
    options = [option.public_dict() for option in _options().values()]
    return {
        "auth": (
            "api_key"
            if settings.azure_openai_auth == "api_key"
            or (
                settings.azure_openai_auth == "auto"
                and settings.azure_openai_api_key
            )
            else "managed_identity"
        ),
        "defaults": {
            "analysis": settings.event_analysis_model,
            "chat": settings.event_chat_model,
        },
        "analysis": [
            {
                "alias": "auto",
                "label": "Auto（质量优先）",
                "model": None,
                "description": "当前自动选择 GPT-4.1，后续可扩展评测驱动路由。",
                "available": bool(settings.foundry_analysis_deployment),
            },
            *options,
        ],
        "chat": [
            {
                "alias": "auto",
                "label": "Auto（按问题复杂度）",
                "model": None,
                "description": "简单问题走 mini，复杂证据推理走 GPT-4.1。",
                "available": bool(
                    settings.foundry_analysis_deployment
                    and settings.foundry_chat_deployment
                ),
            },
            *options,
        ],
    }


_COMPLEX_CHAT_CUES = (
    "为什么",
    "依据",
    "证据",
    "比较",
    "推理",
    "异常",
    "全过程",
    "详细",
    "why",
    "evidence",
    "compare",
    "explain",
    "reason",
)


def resolve_model(
    task: str,
    requested: str | None = None,
    *,
    prompt: str | None = None,
    require_deployment: bool = True,
) -> ModelSelection:
    if task not in {"analysis", "chat"}:
        raise ValueError(f"未知大模型任务：{task}")
    default = (
        settings.event_analysis_model
        if task == "analysis"
        else settings.event_chat_model
    )
    alias = (requested or default or "auto").strip().lower()
    options = _options()

    if alias == "auto":
        if task == "analysis":
            selected = "gpt-4.1"
            reason = "accuracy-first video analysis"
        else:
            text = (prompt or "").lower()
            complex_question = len(text) > 180 or any(
                cue in text for cue in _COMPLEX_CHAT_CUES
            )
            selected = "gpt-4.1" if complex_question else "gpt-4.1-mini"
            reason = (
                "complex evidence reasoning"
                if complex_question
                else "latency-sensitive follow-up"
            )
    else:
        selected = alias
        reason = "explicit user selection"

    option = options.get(selected)
    if option is None:
        allowed = ", ".join(["auto", *options])
        raise ValueError(f"未知{task}模型：{alias}；可选：{allowed}")
    if require_deployment and not option.deployment:
        raise RuntimeError(f"模型 {option.label} 尚未配置部署")
    return ModelSelection(
        task=task,
        requested=alias,
        selected=selected,
        model=option.model,
        deployment=option.deployment or "",
        reason=reason,
    )


__all__ = ["ModelSelection", "model_catalog", "resolve_model"]
