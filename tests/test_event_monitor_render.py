from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


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


def test_reid_diagnostics_render_tracking_zero_and_failed_calls() -> None:
    render_url = json.dumps(
        (ROOT / "static" / "js" / "event-monitor" / "render.js").as_uri()
    )
    result = _run_module_script(
        f"""
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
    assert "跟踪 2 次" in result["trackingOnly"]["html"]
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
