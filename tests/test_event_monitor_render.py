from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from app.event_monitor_i18n import build_page_bundle

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


def test_gait_summary_uses_effective_pipeline_state() -> None:
    source = (
        ROOT / "static" / "js" / "event-monitor" / "render.js"
    ).read_text(encoding="utf-8")

    assert "const withGait = data.with_gait ?? configUsed.with_gait;" in source
    assert 't("results.config_gait", { state: on(withGait) })' in source


def test_post_analysis_ai_actions_are_separate_and_run_scoped() -> None:
    template = (ROOT / "templates" / "event-monitor.html").read_text(encoding="utf-8")
    results_css = (
        ROOT / "static" / "css" / "event-monitor" / "results.css"
    ).read_text(encoding="utf-8")
    polish_css = (
        ROOT / "static" / "css" / "event-monitor" / "polish.css"
    ).read_text(encoding="utf-8")
    render_source = (
        ROOT / "static" / "js" / "event-monitor" / "render.js"
    ).read_text(encoding="utf-8")

    assert 'id="resultTools" class="em-result-actions"' in template
    assert 'id="dryRunAction" class="em-ai-action em-ai-action-reuse" hidden' in template
    assert 'id="btnSendLlm" class="em-ai-action-button" type="button"' in template
    assert 'class="em-result-utilities" aria-labelledby="resultUtilitiesTitle"' in template
    assert '<section id="chatPanel" class="em-chat em-evidence-chat"' in template
    assert template.index('class="em-result-utilities"') < template.index('id="dryRunAction"')
    assert template.index('id="dryRunAction"') < template.index('id="chatPanel"')
    utilities_start = template.index('class="em-result-utilities"')
    assert utilities_start < template.index('id="promptView"') < template.index('id="dryRunAction"')
    assert "em-ai-action-icon" not in template
    assert "em-chat-icon" not in template
    assert 'id="chatModelStatus" class="em-chat-model-status" role="status" aria-live="polite"' in template
    assert 'aria-describedby="chatEvidenceMeta"' in template
    assert "$(\"dryRunAction\").hidden = !data.dry_run;" in render_source
    assert "chatPanel.dataset.runId = data.run_id;" in render_source
    assert ".em-ai-action[hidden]" in results_css
    assert ".em-ai-action-icon" not in results_css
    assert "grid-template-columns: minmax(0, 1fr) auto;" in results_css
    assert ".em-ai-action-button { justify-self: start; }" in results_css
    assert "@media (max-width: 640px)" in results_css
    assert ".em-ai-action-button:focus-visible" in polish_css
    assert ".em-chat-icon" not in polish_css
    assert ".em-chat-compose .em-btn { align-self: flex-end; }" in polish_css

    for locale in ("en", "zh-CN"):
        messages = build_page_bundle(locale)["messages"]
        assert messages["results"]["dry_run_action_title"]
        assert messages["results"]["utilities_title"]
        assert messages["chat"]["kind"]
        assert messages["chat"]["evidence_scope"]


def test_prompt_artifact_controls_are_localized_and_label_tsv_accurately() -> None:
    template = (ROOT / "templates" / "event-monitor.html").read_text(encoding="utf-8")
    actions_source = (
        ROOT / "static" / "js" / "event-monitor" / "actions.js"
    ).read_text(encoding="utf-8")
    english = build_page_bundle("en")["messages"]["results"]
    chinese = build_page_bundle("zh-CN")["messages"]["results"]

    assert '<option value="tsv" selected>' in template
    assert template.index('id="promptView"') < template.index('id="dryRunAction"')
    assert 'const DEFAULT_PROMPT_FORMAT = "tsv";' in actions_source
    assert 'return $("promptFormat").value === "json" ? "json" : DEFAULT_PROMPT_FORMAT;' in actions_source
    assert 'const format = selectedPromptFormat();' in actions_source
    assert "anchor.download = `event-monitor_${runId}_prompt.${format}`;" in actions_source
    assert english["download_prompt_button"] == "⬇ Download prompt"
    assert english["view_prompt_show"] == "View prompt"
    assert english["prompt_format_tsv"] == "Compact table (TSV/CSV-style)"
    assert chinese["download_prompt_button"] == "⬇ 下载提示词"
    assert chinese["view_prompt_show"] == "查看提示词"
    assert "TSV/CSV" in chinese["prompt_format_tsv"]
    assert "default" in english["prompt_export_hint"]
    assert "默认" in chinese["prompt_export_hint"]


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


def test_prompt_api_uses_only_run_id_and_selected_format() -> None:
    api_url = json.dumps(
        (ROOT / "static" / "js" / "event-monitor" / "api.js").as_uri()
    )
    result = _run_module_script(
        f"""
const calls = [];
globalThis.fetch = async (url, options) => {{
  calls.push({{ url, options: options || null }});
  return {{ ok: true, text: async () => "EM-EVIDENCE-TSV/1" }};
}};
const {{ getRunPrompt, runPromptUrl }} = await import({api_url});
const content = await getRunPrompt("abcdef123456", "tsv");
console.log(JSON.stringify({{
  url: runPromptUrl("abcdef123456", "json"),
  call: calls[0],
  content,
}}));
"""
    )

    assert result == {
        "url": "/api/event-monitor/runs/abcdef123456/prompt?format=json",
        "call": {
            "url": "/api/event-monitor/runs/abcdef123456/prompt?format=tsv",
            "options": None,
        },
        "content": "EM-EVIDENCE-TSV/1",
    }


def test_prompt_download_defaults_to_tsv_when_the_control_has_no_value() -> None:
    actions_url = json.dumps(
        (ROOT / "static" / "js" / "event-monitor" / "actions.js").as_uri()
    )
    state_url = json.dumps(
        (ROOT / "static" / "js" / "event-monitor" / "state.js").as_uri()
    )
    result = _run_module_script(
        f"""
const elements = {{
  promptFormat: {{ value: "" }},
}};
let download = null;
globalThis.document = {{
  getElementById: (id) => elements[id],
  createElement: () => ({{
    click() {{
      download = {{ href: this.href, filename: this.download }};
    }},
  }}),
}};
const {{ setLastPayload }} = await import({state_url});
const {{ downloadPrompt }} = await import({actions_url});
setLastPayload({{ run_id: "abcdef123456" }});
downloadPrompt();
console.log(JSON.stringify(download));
"""
    )

    assert result == {
        "href": "/api/event-monitor/runs/abcdef123456/prompt?format=tsv",
        "filename": "event-monitor_abcdef123456_prompt.tsv",
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


@pytest.mark.parametrize(
    ("locale", "expand_text", "collapse_text", "unavailable_text"),
    [
        ("en", "Show details", "Hide details", "unavailable"),
        ("zh-CN", "展开详情", "收起详情", "不可用"),
    ],
)
def test_measured_timings_are_collapsed_and_keep_truthful_widths(
    locale: str,
    expand_text: str,
    collapse_text: str,
    unavailable_text: str,
) -> None:
    render_url = json.dumps(
        (ROOT / "static" / "js" / "event-monitor" / "render.js").as_uri()
    )
    bundle = _bundle(locale, "results", "timings")
    result = _run_module_script(
        f"""
globalThis.__EVENT_MONITOR_I18N__ = {bundle};
const elements = {{
  timings: {{ hidden: true, innerHTML: "" }},
}};
globalThis.document = {{
  getElementById: (id) => elements[id],
}};
const {{ renderTimings }} = await import({render_url});
renderTimings({{
  elapsed_seconds: "10",
  stage_timings: {{
    longest: 3.09,
    other_overhead: 6.9,
    tiny: 0.01,
    zero: 0,
    invalid: "not-a-number",
    negative: -1,
  }},
}});
console.log(JSON.stringify(elements.timings));
"""
    )

    html = result["innerHTML"]
    assert result["hidden"] is False
    assert '<details class="em-timings-details">' in html
    assert '<details class="em-timings-details" open' not in html
    assert expand_text in html
    assert collapse_text in html
    assert "not live progress" in html or "不是实时进度" in html
    assert 'style="width:30.900%"' in html
    assert 'style="width:69.000%"' in html
    assert 'style="width:0.100%"' in html
    assert 'style="width:100.000%"' not in html
    assert "30.9%" in html
    assert "69.0%" in html
    assert "0.1%" in html
    widths = [
        float(width)
        for width in re.findall(r'style="width:([0-9.]+)%"', html)
    ]
    assert sum(widths) == pytest.approx(100.0)
    assert "Measured stages" in html or "实测阶段" in html
    assert "longest measured stage" not in html
    assert "最长的实测阶段" not in html
    assert 'class="em-tbar-marker"' in html
    assert "0ms · 0%" in html
    assert html.count(unavailable_text) >= 2
    assert "NaN" not in html
    assert "Infinity" not in html
    timing_css = (ROOT / "static" / "css" / "event-monitor" / "polish.css").read_text(
        encoding="utf-8"
    )
    assert ".em-tbar-fill {\n  display: block;" in timing_css
    assert ".em-tbar-marker {" in timing_css


@pytest.mark.parametrize(
    ("locale", "gallery_head", "subject_title", "details_title", "track_label"),
    [
        ("en", "Identity gallery (subjects: 1)", "Subject #7", "Identity evidence", "Track IDs"),
        ("zh-CN", "身份画廊（本段共 1 个主体）", "主体 #7", "身份依据", "轨迹编号"),
    ],
)
def test_dynamic_translation_and_gallery_markup_follow_selected_language(
    locale: str,
    gallery_head: str,
    subject_title: str,
    details_title: str,
    track_label: str,
) -> None:
    render_url = json.dumps(
        (ROOT / "static" / "js" / "event-monitor" / "render.js").as_uri()
    )
    gallery_url = json.dumps(
        (ROOT / "static" / "js" / "event-monitor" / "identity-gallery.js").as_uri()
    )
    bundle = _bundle(locale, "samples", "service", "gallery", "common")
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
      subject_conflict_split: true,
      decision: "conflict_split",
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

    assert result["sampleCount"]
    assert result["backendText"]
    assert gallery_head in result["html"]
    assert "class=\"em-gallery-panel\"" in result["html"]
    assert subject_title in result["html"]
    assert '<details class="em-subcard"' in result["html"]
    assert 'class="em-subcard-summary"' in result["html"]
    assert details_title in result["html"]
    assert track_label in result["html"]
    assert "Best body score" in result["html"] or "最佳人形分数" in result["html"]
    assert "Face evidence" in result["html"] or "人脸依据" in result["html"]
    assert "Gait evidence" in result["html"] or "步态依据" in result["html"]


def test_gallery_missing_scores_render_as_unavailable() -> None:
    gallery_url = json.dumps(
        (ROOT / "static" / "js" / "event-monitor" / "identity-gallery.js").as_uri()
    )
    bundle = _bundle("en", "gallery", "common")
    result = _run_module_script(
        f"""
globalThis.__EVENT_MONITOR_I18N__ = {bundle};
const {{ renderSubjectGallery }} = await import({gallery_url});
console.log(JSON.stringify({{
  html: renderSubjectGallery({{
    tracks: {{
      "1": {{
        subject_id: 7,
        score: null,
        fused: {{ confidence: null }},
      }},
    }},
  }}),
}}));
"""
    )

    assert "Best body score" in result["html"]
    assert "unavailable" in result["html"].lower()
    assert "0.00" not in result["html"]
    assert "0%" not in result["html"]


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
    {{ alias: "auto", label: "Auto", available: true }},
    {{ alias: "gpt-4.1", label: "GPT-4.1", available: true }},
  ],
  chat: [
    {{ alias: "auto", label: "Auto", available: true }},
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
            {"text": "Auto", "value": "auto"},
            {"text": "GPT-4.1", "value": "gpt-4.1"},
        ],
        "chat": [
            {"text": "Auto", "value": "auto"},
            {"text": "GPT-4.1 mini", "value": "gpt-4.1-mini"},
        ],
        "analysisValue": "auto",
        "chatValue": "gpt-4.1-mini",
        "badge": "VM managed identity",
        "disabled": False,
    }


def test_model_picker_uses_accessible_popover_instead_of_model_selects() -> None:
    template = (ROOT / "templates" / "event-monitor.html").read_text(encoding="utf-8")
    source = (
        ROOT / "static" / "js" / "event-monitor" / "settings.js"
    ).read_text(encoding="utf-8")
    css = (ROOT / "static" / "css" / "event-monitor" / "polish.css").read_text(
        encoding="utf-8"
    )

    assert '<select id="analysisModel"' not in template
    assert '<select id="chatModel"' not in template
    assert 'role="listbox"' in template
    assert 'class="em-model-search"' in template
    assert 'modelPickerDocumentEventsWired' in source
    assert 'event.key === "Escape"' in source
    assert 'event.key === "ArrowDown"' in source
    assert 'providerMark(option.provider)' in source
    assert 'class="em-provider-slot"' in source
    assert 'brand-assets/${mark.asset}' in source
    assert ".em-model-popover" in css
    assert ".em-provider-slot" in css
    assert ".em-model-copy { min-width: 0; }" in css
    assert "overflow-wrap: anywhere;" in css

    assets = ROOT / "static" / "vendor" / "brand-assets"
    assert (assets / "auto-routing.svg").is_file()
    assert (assets / "openai.svg").is_file()
    assert (assets / "microsoft.svg").is_file()
    assert (assets / "deepseek.svg").is_file()
    assert "CC0-1.0" in (assets / "README.md").read_text(encoding="utf-8")
