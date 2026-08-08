"""Backend-owned routing over server-discovered callable Foundry targets."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from ..core.config import settings
from .llm_catalog import event_llm_catalog


@dataclass(frozen=True)
class ModelSelection:
    task: str
    requested: str
    selected: str
    model: str
    deployment: str
    reason: str
    source: str = "configured"

    def public_dict(self) -> dict:
        return {
            "task": self.task,
            "requested": self.requested,
            "selected": self.selected,
            "model": self.model,
            "reason": self.reason,
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


def _is_english(locale: str | None) -> bool:
    return (locale or "").lower().startswith("en")


def _text(locale: str | None, english: str, chinese: str) -> str:
    return english if _is_english(locale) else chinese


_MODEL_DESCRIPTIONS_ZH = {
    "gpt-4o": "均衡的多模态分析。",
    "gpt-4.1": "适合细节和证据推理。",
    "gpt-4.1-mini": "快速、高效的追问聊天。",
    "gpt-5.4": "适合复杂视觉证据的最高质量分析。",
    "gpt-5.4-mini": "高效的多模态分析与聊天。",
    "gpt-5.6-luna": "均衡的多模态推理。",
    "gpt-5.6-terra": "最高质量的结构化聊天推理。",
    "phi-4-reasoning": "专注推理的文本和结构化聊天。",
    "deepseek-v4-flash": "快速结构化聊天推理。",
    "kimi-k2.6": "预览版长上下文结构化聊天。",
}

_REQUEST_FAMILY = {
    "gpt-5.4": "gpt5",
    "gpt-5.4-mini": "gpt5",
    "gpt-5.6-luna": "gpt5",
    "gpt-5.6-terra": "gpt5",
}


def _resolved_auth_mode() -> str:
    if settings.azure_openai_auth == "auto":
        return "api_key" if settings.azure_openai_api_key else "managed_identity"
    return settings.azure_openai_auth


def _targets(task: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    catalog = event_llm_catalog()
    capability = "image_input" if task == "analysis" else "chat"
    targets = [
        target
        for target in catalog.get("callable_targets", [])
        if target.get("deployment")
        and (target.get("capabilities") or {}).get(capability) is True
        and (target.get("capabilities") or {}).get("json_output") is True
        and (task != "analysis" or (target.get("capabilities") or {}).get("chat") is True)
    ]
    return targets, catalog


def _model_rank(model: str) -> tuple[int, str]:
    """A deterministic quality ordering with a stable lexical tie-breaker."""
    value = (model or "").lower()
    match = re.search(r"gpt-(\d+)(?:\.(\d+))?", value)
    if not match:
        return (0, value)
    major = int(match.group(1))
    minor = int(match.group(2) or 0)
    # "mini" is deliberately a capacity qualifier, not a deployment name.
    # This keeps the ordering extensible while preferring full models on ties.
    return (major * 100 + minor * 10 - (1 if "mini" in value else 0), value)


def _best_target(targets: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        targets,
        key=lambda target: (
            -_model_rank(str(target.get("model") or ""))[0],
            str(target.get("deployment") or "").lower(),
        ),
    )[0]


def _configured_target(
    targets: list[dict[str, Any]], deployment: str | None
) -> dict[str, Any] | None:
    return next(
        (target for target in targets if target["deployment"] == deployment), None
    )


def model_catalog(*, locale: str | None = None) -> dict:
    """Public UI data. Catalog model metadata is never a callable selection."""
    analysis, raw = _targets("analysis")
    chat, _ = _targets("chat")

    def option(target: dict[str, Any]) -> dict[str, Any]:
        return {
            "alias": target["deployment"],
            "label": target["label"],
            "model": target["model"],
            "provider": target.get("provider", "openai"),
            "group": target.get("group", "Versatile"),
            "context": target.get("context"),
            "performance": target.get("performance"),
            "preview": bool(target.get("preview")),
            "description": _text(
                locale,
                str(target.get("description") or "Callable Foundry deployment."),
                _MODEL_DESCRIPTIONS_ZH.get(
                    str(target.get("model") or "").lower(),
                    "服务端发现并验证可调用的 Foundry 部署。",
                ),
            ),
            "available": True,
        }

    auto_analysis = {
        "alias": "auto",
        "label": _text(locale, "Auto (best compatible deployment)", "自动（最佳兼容部署）"),
        "model": None,
        "description": _text(
            locale,
            "Deterministically selects the highest-ranked image-capable deployment.",
            "确定性选择排名最高的图像输入兼容部署。",
        ),
        "available": bool(analysis),
    }
    auto_chat = {
        "alias": "auto",
        "label": _text(locale, "Auto (task-aware deployment)", "自动（按任务选择部署）"),
        "model": None,
        "description": _text(
            locale,
            "Uses the configured chat deployment for short follow-ups and the highest-ranked compatible deployment for evidence-heavy questions.",
            "简短追问使用已配置的聊天部署，证据推理问题使用排名最高的兼容部署。",
        ),
        "available": bool(chat),
    }
    return {
        "auth": _resolved_auth_mode(),
        "defaults": {"analysis": "auto", "chat": "auto"},
        "source": raw.get("source"),
        "warning": raw.get("warning"),
        "catalog_models": raw.get("catalog_models", []),
        "callable_targets": raw.get("callable_targets", []),
        "analysis": [auto_analysis, *map(option, analysis)],
        "chat": [auto_chat, *map(option, chat)],
    }


def chat_completion_options(
    model: str, *, max_tokens: int, temperature: float | None
) -> dict[str, Any]:
    """Return the supported output control for the selected reviewed model."""
    if _REQUEST_FAMILY.get(model.lower()) == "gpt5":
        return {"max_completion_tokens": max_tokens}
    options: dict[str, Any] = {"max_tokens": max_tokens}
    if temperature is not None:
        options["temperature"] = temperature
    return options


def resolve_model(
    task: str,
    requested: str | None = None,
    *,
    prompt: str | None = None,
    require_deployment: bool = True,
    locale: str | None = None,
) -> ModelSelection:
    if task not in {"analysis", "chat"}:
        raise ValueError(_text(locale, f"Unknown LLM task: {task}", f"未知大模型任务：{task}"))
    alias = (requested or "auto").strip()
    targets, catalog = _targets(task)
    allowed = {target["deployment"]: target for target in targets}

    if alias.lower() == "auto":
        if not targets:
            if require_deployment:
                capability = "image-capable" if task == "analysis" else "chat-compatible"
                raise RuntimeError(
                    _text(
                        locale,
                        f"No callable {capability} Microsoft Foundry deployment is available.",
                        f"没有可调用且{('支持图像输入' if task == 'analysis' else '支持聊天')}的 Microsoft Foundry 部署。",
                    )
                )
            return ModelSelection(
                task=task,
                requested="auto",
                selected="auto",
                model="",
                deployment="",
                reason=_text(locale, "No deployment is required for dry-run.", "dry-run 不需要部署。"),
                source=str(catalog.get("source") or "configured"),
            )
        if task == "analysis":
            selected = _best_target(targets)
            reason = _text(
                locale,
                "Auto selected the highest-ranked image-capable deployment.",
                "自动选择排名最高的图像输入兼容部署。",
            )
        else:
            text = (prompt or "").lower()
            complex_question = len(text) > 180 or any(
                cue in text for cue in _COMPLEX_CHAT_CUES
            )
            if complex_question:
                selected = _best_target(targets)
                reason = _text(
                    locale,
                    "Auto selected the highest-ranked compatible deployment for evidence reasoning.",
                    "自动为证据推理选择排名最高的兼容部署。",
                )
            else:
                selected = _configured_target(
                    targets, settings.foundry_chat_deployment
                ) or _best_target(targets)
                reason = _text(
                    locale,
                    "Auto selected the configured chat deployment for a short follow-up.",
                    "自动为简短追问选择已配置的聊天部署。",
                )
        requested_value = "auto"
    else:
        selected = allowed.get(alias)
        if selected is None:
            raise ValueError(
                _text(
                    locale,
                    "Unknown or unavailable callable deployment.",
                    "未知或不可用的可调用部署。",
                )
            )
        requested_value = alias
        reason = _text(locale, "Explicit callable deployment selection.", "显式选择可调用部署。")

    return ModelSelection(
        task=task,
        requested=requested_value,
        selected=selected["deployment"],
        model=selected["model"],
        deployment=selected["deployment"],
        reason=reason,
        source=str(catalog.get("source") or "configured"),
    )


__all__ = [
    "ModelSelection",
    "chat_completion_options",
    "model_catalog",
    "resolve_model",
]
