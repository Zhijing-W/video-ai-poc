from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from app.event_monitor_i18n import build_page_bundle

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


def _run_module_script(script: str) -> dict:
    if NODE is None:
        pytest.skip("Node.js is not installed")
    completed = subprocess.run(
        [
            NODE,
            "--input-type=module",
            "--eval",
            script,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _bundle(locale: str, *message_keys: str) -> str:
    bundle = build_page_bundle(locale)
    messages = (
        {key: bundle["messages"][key] for key in message_keys}
        if message_keys
        else bundle["messages"]
    )
    return json.dumps(
        {
            "locale": bundle["locale"],
            "reportLanguage": bundle["reportLanguage"],
            "messages": messages,
        }
    )


def test_reid_diagnostics_render_tracking_zero_and_failed_calls() -> None:
    render_url = json.dumps(
        (ROOT / "static" / "js" / "event-monitor" / "render.js").as_uri()
    )
    bundle = _bundle("zh-CN", "results")
    result = _run_module_script(
        f"""
globalThis.__EVENT_MONITOR_I18N__ = {bundle};
const elements = {{
  reidDiagnostics: {{ hidden: true, innerHTML: "" }},
  empty: {{ style: {{}}, textContent: "" }},
}};
globalThis.document = {{
  getElementById: (id) => elements[id],
}};
const {{ renderReidDiagnostics, showRunFailure }} = await import({render_url});
const timing = (calls, failures) => ({{
  backend: "differ",
  device: "cuda:0",
  call_count: calls,
  total_ms: calls ? 20 : 0,
  mean_ms: calls ? 10 : null,
  p95_ms: calls ? 11 : null,
  failed_call_count: failures,
  by_purpose: calls
    ? {{ tracking: {{ call_count: calls, total_ms: 20 }} }}
    : {{}},
}});
const capture = () => ({{
  hidden: elements.reidDiagnostics.hidden,
  html: elements.reidDiagnostics.innerHTML,
}});
renderReidDiagnostics({{
  config_used: {{ with_body: false }},
  body_reid_timing: timing(2, 0),
}});
const trackingOnly = capture();
renderReidDiagnostics({{
  config_used: {{ with_body: true }},
  body_reid_timing: timing(2, 1),
}});
const failed = capture();
renderReidDiagnostics({{
  config_used: {{ with_body: true }},
  body_reid_timing: timing(0, 0),
}});
const identityZero = capture();
renderReidDiagnostics({{
  config_used: {{ with_body: false }},
  body_reid_timing: timing(0, 0),
}});
const disabledZero = capture();
showRunFailure("fatal", {{
  config_used: {{ with_body: false }},
  body_reid_timing: timing(1, 1),
}});
const fatal = capture();
console.log(JSON.stringify({{
  trackingOnly,
  failed,
  identityZero,
  disabledZero,
  fatal,
  failureText: elements.empty.textContent,
}}));
"""
    )

    assert result["trackingOnly"]["hidden"] is False
    assert "跟踪 2/20.0ms" in result["trackingOnly"]["html"]
    assert "失败 <b>0</b>" in result["trackingOnly"]["html"]
    assert "has-failures" not in result["trackingOnly"]["html"]
    assert result["failed"]["hidden"] is False
    assert "has-failures" in result["failed"]["html"]
    assert "失败 <b>1</b>" in result["failed"]["html"]
    assert result["identityZero"]["hidden"] is False
    assert "0 次调用" in result["identityZero"]["html"]
    assert result["disabledZero"] == {"hidden": True, "html": ""}
    assert result["fatal"]["hidden"] is False
    assert "has-failures" in result["fatal"]["html"]
    assert result["failureText"] == "处理失败：fatal"


def test_api_preserves_structured_error_detail_for_ui() -> None:
    api_url = json.dumps(
        (ROOT / "static" / "js" / "event-monitor" / "api.js").as_uri()
    )
    result = _run_module_script(
        f"""
globalThis.fetch = async () => ({{
  ok: false,
  status: 500,
  json: async () => ({{
    detail: {{
      message: "事件理解失败：embedding exploded",
      body_reid_timing: {{ call_count: 1, failed_call_count: 1 }},
    }},
  }}),
}});
const {{ runAnalysis }} = await import({api_url});
try {{
  await runAnalysis(null);
}} catch (error) {{
  console.log(JSON.stringify({{
    message: error.message,
    detail: error.detail,
  }}));
}}
"""
    )

    assert result["message"] == "事件理解失败：embedding exploded"
    assert result["detail"]["body_reid_timing"] == {
        "call_count": 1,
        "failed_call_count": 1,
    }


def test_progress_omits_virtual_percentage_explanation() -> None:
    progress_url = json.dumps(
        (ROOT / "static" / "js" / "event-monitor" / "progress.js").as_uri()
    )
    bundle = _bundle("zh-CN", "progress")
    result = _run_module_script(
        f"""
globalThis.__EVENT_MONITOR_I18N__ = {bundle};
const elements = {{
  progress: {{ hidden: true }},
  progressBar: {{
    classList: {{ add: () => {{}}, remove: () => {{}} }},
    style: {{ width: "50%" }},
  }},
  progressStage: {{ textContent: "" }},
  progressSteps: {{ innerHTML: "old" }},
  progressTimer: {{ textContent: "" }},
}};
globalThis.document = {{
  getElementById: (id) => elements[id],
  querySelector: () => null,
}};
globalThis.setInterval = () => 1;
globalThis.clearInterval = () => {{}};
const {{ startProgress }} = await import({progress_url});
startProgress(true);
console.log(JSON.stringify({{
  hidden: elements.progress.hidden,
  steps: elements.progressSteps.innerHTML,
  stage: elements.progressStage.textContent,
}}));
"""
    )

    assert result["hidden"] is False
    assert result["steps"] == ""
    assert result["stage"] == "⏳ 服务端分析中…"
    assert "不显示估算阶段" not in result["stage"]
    assert "实测阶段耗时" not in result["stage"]


def test_dynamic_translation_and_gallery_markup_follow_selected_language() -> None:
    render_url = json.dumps(
        (ROOT / "static" / "js" / "event-monitor" / "render.js").as_uri()
    )
    gallery_url = json.dumps(
        (ROOT / "static" / "js" / "event-monitor" / "identity-gallery.js").as_uri()
    )
    bundle = _bundle("en", "samples", "service", "gallery", "common")
    result = _run_module_script(
        f"""
globalThis.__EVENT_MONITOR_I18N__ = {bundle};
const backendStatus = {{
  className: "",
  lastChild: {{ textContent: "" }},
}};
const elements = {{
  sampleSelect: {{
    innerHTML: "",
    options: [],
    replaceChildren() {{ this.options = []; }},
    appendChild(option) {{ this.options.push(option); }},
  }},
  sampleCount: {{ textContent: "" }},
  backendStatus,
}};
globalThis.document = {{
  createElement: () => ({{
    value: "",
    textContent: "",
  }}),
  getElementById: (id) => elements[id],
}};
const {{ renderSamples, setBackendIndicator }} = await import({render_url});
const {{ renderSubjectGallery }} = await import({gallery_url});
renderSamples({{ samples: [{{ name: "demo.mp4", size_mb: 1.2 }}] }});
setBackendIndicator(true);
const html = renderSubjectGallery({{
  tracks: {{
    "1": {{
      subject_id: 7,
      thumb: "thumb",
      score: 0.91,
      reused: true,
      local_subject: true,
      subject_conflict_split: false,
      face: {{
        observed: true,
        eligibility: "usable",
        match_ready: true,
        match_source: "superres",
        quality: 0.8,
      }},
      fused: {{
        confidence: 0.88,
        primary: "body",
        resolved: true,
        agreed: true,
      }},
    }},
  }},
}});
console.log(JSON.stringify({{
  sampleCount: elements.sampleCount.textContent,
  backendText: backendStatus.lastChild.textContent,
  html,
}}));
"""
    )

    assert result["sampleCount"] == "samples: 1"
    assert result["backendText"] == "Service online"
    assert "Identity gallery (subjects: 1)" in result["html"]
    assert "class=\"em-gallery-panel\"" in result["html"]
    assert "Subject #7" in result["html"]
    assert "tracks: 1" in result["html"]


def test_product_reid_selector_shows_one_default_and_four_validated_backends() -> None:
    settings_url = json.dumps(
        (ROOT / "static" / "js" / "event-monitor" / "settings.js").as_uri()
    )
    bundle = _bundle("en", "settings")
    result = _run_module_script(
        f"""
globalThis.__EVENT_MONITOR_I18N__ = {bundle};
globalThis.Option = class {{
  constructor(text, value) {{
    this.text = text;
    this.value = value;
  }}
}};
const select = {{
  dataset: {{}},
  disabled: true,
  options: [],
  value: null,
  replaceChildren() {{ this.options = []; }},
  add(option) {{ this.options.push({{ text: option.text, value: option.value }}); }},
}};
globalThis.document = {{
  getElementById: (id) => id === "reidBackend" ? select : null,
}};
const {{ renderReidBackends }} = await import({settings_url});
renderReidBackends({{
  default: "differ",
  backends: [
    "auto",
    "clipreid",
    "coarse",
    "differ",
    "osnet",
    "resnet50",
    "siglip2",
  ],
  metadata: {{}},
}});
console.log(JSON.stringify({{
  options: select.options,
  defaultBackend: select.dataset.defaultBackend,
  disabled: select.disabled,
}}));
"""
    )

    assert result["defaultBackend"] == "differ"
    assert result["disabled"] is False
    assert result["options"] == [
        {
            "text": "DIFFER EVA02-L (default · accuracy first)",
            "value": "",
        },
        {"text": "CLIP-ReID ViT-B/16", "value": "clipreid"},
        {"text": "SigLIP2 Person ReID", "value": "siglip2"},
        {"text": "OSNet-AIN MSMT17", "value": "osnet"},
    ]


def test_superres_selector_does_not_duplicate_the_default_backend() -> None:
    settings_url = json.dumps(
        (ROOT / "static" / "js" / "event-monitor" / "settings.js").as_uri()
    )
    bundle = _bundle("zh-CN", "settings")
    result = _run_module_script(
        f"""
globalThis.__EVENT_MONITOR_I18N__ = {bundle};
globalThis.Option = class {{
  constructor(text, value) {{
    this.text = text;
    this.value = value;
  }}
}};
const select = {{
  dataset: {{}},
  options: [],
  value: "",
  replaceChildren() {{ this.options = []; }},
  add(option) {{ this.options.push({{ text: option.text, value: option.value }}); }},
}};
const fidelityField = {{ hidden: true }};
globalThis.document = {{
  getElementById: (id) => ({{
    faceSuperres: select,
    faceCodeformerFidelityField: fidelityField,
    faceCodeformerFidelity: {{ value: "" }},
  }})[id] || null,
}};
const {{ renderSuperresBackends }} = await import({settings_url});
renderSuperresBackends({{
  default: "off",
  backends: ["off", "gfpgan", "codeformer", "realesrgan_x2plus"],
  metadata: {{}},
}});
console.log(JSON.stringify({{ options: select.options }}));
"""
    )

    assert result["options"] == [
        {"text": "关闭（默认）", "value": ""},
        {"text": "GFP-GAN", "value": "gfpgan"},
        {"text": "CodeFormer", "value": "codeformer"},
        {"text": "Real-ESRGAN x2plus", "value": "realesrgan_x2plus"},
    ]


def test_llm_model_selector_renders_allowlisted_aliases() -> None:
    settings_url = json.dumps(
        (ROOT / "static" / "js" / "event-monitor" / "settings.js").as_uri()
    )
    bundle = _bundle("en", "panel")
    result = _run_module_script(
        f"""
globalThis.__EVENT_MONITOR_I18N__ = {bundle};
globalThis.Option = class {{
  constructor(text, value) {{
    this.text = text;
    this.value = value;
  }}
}};
const analysis = {{
  disabled: true,
  options: [],
  value: null,
  replaceChildren() {{ this.options = []; }},
  add(option) {{ this.options.push({{ text: option.text, value: option.value }}); }},
}};
const chat = {{ ...analysis, options: [] }};
const hint = {{ textContent: "" }};
const badge = {{ textContent: "", className: "" }};
globalThis.document = {{
  getElementById: (id) => ({{
    analysisModel: analysis,
    chatModel: chat,
    llmModelHint: hint,
    llmAuthBadge: badge,
  }})[id] || null,
}};
const {{ renderLlmModels }} = await import({settings_url});
renderLlmModels({{
  auth: "managed_identity",
  defaults: {{ analysis: "auto", chat: "gpt-4.1-mini" }},
  analysis: [
    {{ alias: "auto", label: "Auto (quality first)", available: true }},
    {{ alias: "gpt-4.1", label: "GPT-4.1", available: true }},
  ],
  chat: [
    {{ alias: "auto", label: "Auto (question complexity)", available: true }},
    {{ alias: "gpt-4.1-mini", label: "GPT-4.1 mini", available: true }},
  ],
}});
console.log(JSON.stringify({{
  analysis: analysis.options,
  chat: chat.options,
  analysisValue: analysis.value,
  chatValue: chat.value,
  badge: badge.textContent,
  disabled: analysis.disabled || chat.disabled,
}}));
"""
    )

    assert result == {
        "analysis": [
            {"text": "Auto (quality first)", "value": "auto"},
            {"text": "GPT-4.1", "value": "gpt-4.1"},
        ],
        "chat": [
            {"text": "Auto (question complexity)", "value": "auto"},
            {"text": "GPT-4.1 mini", "value": "gpt-4.1-mini"},
        ],
        "analysisValue": "auto",
        "chatValue": "gpt-4.1-mini",
        "badge": "VM managed identity",
        "disabled": False,
    }
