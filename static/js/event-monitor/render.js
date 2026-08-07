import { renderSubjectGallery } from "./identity-gallery.js";
import { clearLastPayload, resetKeyframeRegistry, setLastPayload } from "./state.js";
import { renderTimeline } from "./timeline.js";
import { $, baseName, esc } from "./utils.js";

const STAGE_CN = {
  extract_frames: "视频解码与采样",
  pipeline_setup: "模型与会话初始化",
  object_detection: "YOLO 目标检测推理",
  multi_object_tracking: "多目标轨迹关联（含 Tracker 外观 ReID）",
  frame_preprocess: "帧读取、候选生成与事件分窗",
  gait_collect: "步态样本采集（Pose + Seg）",
  body_identity: "人形身份特征提取与检索",
  face_identity: "人脸检测、对齐与身份检索",
  gait_identity: "步态特征提取与检索",
  identity_fusion: "身份融合与证据整理",
  event_preparation: "事件上下文与关键帧准备",
  event_understanding: "gpt-4o 多帧事件理解",
  overall_summary: "跨事件窗整段总结",
  other_overhead: "其他编排开销",
  detect_track: "逐帧检测、跟踪与候选生成（旧版）",
  reid_identify: "人形身份特征与检索（旧版）",
  face: "人脸身份分支（旧版）",
  gait_embed: "步态身份分支（旧版）",
  merge_fusion_thumb: "身份融合与证据整理（旧版）",
  windows_select: "事件上下文与关键帧准备（旧版）",
  windows_llm: "事件准备与 gpt-4o 理解（旧版）",
};

export function setStatus(message, isError = false) {
  const element = $("status");
  element.textContent = message;
  element.style.color = isError ? "var(--alert)" : "var(--muted)";
}

export function renderSamples(data, failed = false) {
  const select = $("sampleSelect");
  const count = $("sampleCount");

  if (failed) {
    select.innerHTML = '<option value="">加载样片失败</option>';
    if (count) count.textContent = "";
    return;
  }

  const samples = data.samples || [];
  select.innerHTML = "";
  samples.forEach((sample) => {
    const option = document.createElement("option");
    option.value = sample.name;
    option.textContent = `${sample.name} (${sample.size_mb} MB)`;
    select.appendChild(option);
  });

  if (count) count.textContent = samples.length ? `${samples.length} 个` : "";
  if (!samples.length) select.innerHTML = '<option value="">（data/samples 下没有样片）</option>';
}

export function setBackendIndicator(online) {
  const element = $("backendStatus");
  if (!element) return;
  element.className = `em-svc ${online ? "online" : "offline"}`;
  element.lastChild.textContent = online ? "服务在线" : "服务离线";
}

export function prepareForRun() {
  clearLastPayload();
  resetKeyframeRegistry();
  $("empty").style.display = "none";
  $("overall").hidden = true;
  $("cfgSummary").hidden = true;
  $("reidDiagnostics").hidden = true;
  $("timings").hidden = true;
  $("resultTools").hidden = true;
  $("timeline").innerHTML = "";
  $("tracks").innerHTML = "";
  $("meta").innerHTML = "";
  $("jsonView").hidden = true;
  $("jsonView").textContent = "";
}

export function showRunFailure(message, detail = null) {
  $("empty").style.display = "block";
  $("empty").textContent = "处理失败：" + message;
  if (detail?.body_reid_timing) {
    renderReidDiagnostics({
      body_reid_timing: detail.body_reid_timing,
      config_used: detail.config_used || {},
    });
  }
}

function renderMeta(data) {
  const configUsed = data.config_used || {};
  const withBody = configUsed.with_body ?? data.with_body ?? true;
  const reidText = withBody
    ? `人形 ReID <b>${esc(data.reid_backend || configUsed.reid_backend || "unknown")}</b>` +
      `${data.reid_dim ? `(${data.reid_dim}d)` : ""}`
    : "人形 ReID <b>off</b>";
  $("meta").innerHTML =
    `视频 <b>${esc(baseName(data.video))}</b> · ${data.frames_total} 帧 @ ${data.fps}fps · ` +
    `${(data.windows || []).length} 个事件窗 · Tracker <b>${esc(data.tracker_backend || configUsed.track_backend || "botsort_reid")}</b> · ` +
    `${reidText} · ` +
    `模型 <b>${esc(data.model)}</b>${data.dry_run ? " · <b>dry-run</b>" : ""} · ${data.elapsed_seconds}s`;
}

function renderConfigSummary(data) {
  const configUsed = data.config_used || {};
  const chips = [];
  const on = (enabled) => (enabled ? "on" : "off");
  const withBody = configUsed.with_body ?? data.with_body ?? true;

  chips.push(
    `人形 <b>${on(withBody)}</b>` +
      (withBody
        ? `（${esc(data.reid_backend || configUsed.reid_backend || "auto")}` +
          `${configUsed.reid_consistency_enabled ? `，top-${configUsed.reid_decision_top_k} 一致性` : "，top-1"}）`
        : "")
  );
  chips.push(
    `人脸 <b>${on(configUsed.with_face)}</b>` +
      (configUsed.with_face
        ? `（${esc(configUsed.face_rec_backend || "arcface")}` +
          `${configUsed.face_superres && configUsed.face_superres !== "off" ? "+超分" : ""}` +
          `${configUsed.face_3d_cue ? "+3D" : ""}）`
        : "")
  );
  chips.push(`步态 <b>${on(configUsed.with_gait)}</b>`);
  chips.push(`OCR <b>${configUsed.with_ocr ? esc(data.ocr_backend || "on") : "off"}</b>`);
  chips.push(`物体 <b>${on(configUsed.with_objects)}</b>`);
  if (data.gait_error) chips.push(`<span class="warn">步态告警: ${esc(data.gait_error)}</span>`);
  if (data.ocr_error) chips.push(`<span class="warn">OCR告警: ${esc(data.ocr_error)}</span>`);

  $("cfgSummary").hidden = false;
  $("cfgSummary").innerHTML = `本次生效： ${chips.map((chip) => `<span class="em-cfgchip">${chip}</span>`).join(" ")}`;
}

function renderTimings(data) {
  const stageTimings = data.stage_timings || {};
  const rows = Object.entries(stageTimings)
    .map(([key, value]) => ({ key, value: +value }))
    .filter((row) => Number.isFinite(row.value) && row.value > 0)
    .sort((left, right) => right.value - left.value);

  if (!rows.length) {
    $("timings").hidden = true;
    return;
  }

  const max = Math.max(...rows.map((row) => row.value), 0.01);
  const total = rows.reduce((sum, row) => sum + row.value, 0);
  const formatDuration = (seconds) => {
    if (seconds < 0.001) return `${(seconds * 1000000).toFixed(0)}μs`;
    if (seconds < 1) return `${(seconds * 1000).toFixed(seconds < 0.01 ? 1 : 0)}ms`;
    return `${seconds.toFixed(2)}s`;
  };
  const bars = rows
    .map((row, index) => {
      const pct = (row.value / max) * 100;
      const share = total > 0 ? ((row.value / total) * 100).toFixed(0) : "0";
      return (
        `<div class="em-tbar ${index === 0 ? "top" : ""}">` +
        `<span class="em-tbar-label">${esc(STAGE_CN[row.key] || row.key)}</span>` +
        `<span class="em-tbar-track"><span class="em-tbar-fill" style="width:${pct.toFixed(1)}%"></span></span>` +
        `<span class="em-tbar-val">${formatDuration(row.value)} · ${share}%</span></div>`
      );
    })
    .join("");

  $("timings").hidden = false;
  $("timings").innerHTML =
    `<div class="em-timings-head">⏱ 服务端实测耗时（按耗时降序）` +
    `<span class="tot">总 ${esc(data.elapsed_seconds)}s</span></div>${bars}`;
}

export function renderReidDiagnostics(data) {
  const element = $("reidDiagnostics");
  const timing = data.body_reid_timing;
  const withBody = data.config_used?.with_body ?? data.with_body ?? true;
  const calls = Number(timing?.call_count) || 0;
  const failedCalls = Number(timing?.failed_call_count) || 0;
  if (!timing || (!withBody && calls === 0 && failedCalls === 0)) {
    element.hidden = true;
    element.innerHTML = "";
    return;
  }

  const formatMs = (value) =>
    Number.isFinite(Number(value)) ? `${Number(value).toFixed(1)}ms` : "—";
  const purposeLabels = {
    tracking: "跟踪",
    identity_gallery: "最终身份/画廊",
    face_consistency: "人脸候选一致性",
    unspecified: "其他",
  };
  const breakdown = Object.entries(timing.by_purpose || {})
    .filter(([, value]) => Number(value?.call_count) > 0)
    .map(
      ([purpose, value]) =>
        `${purposeLabels[purpose] || purpose} ${value.call_count} 次/${formatMs(value.total_ms)}`
    )
    .join(" · ");
  const backend = timing.backend || data.reid_backend || "unknown";
  const device = timing.device || data.runtime?.reid_device || "unknown";
  const values = calls
    ? `调用 <b>${calls}</b> 次 · 总计 <b>${formatMs(timing.total_ms)}</b> · ` +
      `均值 <b>${formatMs(timing.mean_ms)}</b> · P95 <b>${formatMs(timing.p95_ms)}</b>`
    : "本次没有执行 Body ReID crop embedding（0 次调用）";
  const failures =
    `<span class="em-reid-fail ${failedCalls ? "has-failures" : ""}">` +
    `失败 <b>${failedCalls}</b></span>`;

  element.hidden = false;
  element.innerHTML =
    `<div class="em-reid-head">🧥 单 crop Body ReID 耗时 <span>（不是视频 FPS）</span></div>` +
    `<div class="em-reid-values"><b>${esc(backend)}</b> · ${esc(device)} · ${values} · ${failures}</div>` +
    (breakdown ? `<div class="em-reid-breakdown">${esc(breakdown)}</div>` : "");
}

function renderOverall(overall) {
  const element = $("overall");
  if (!overall || overall.error) {
    element.hidden = true;
    element.innerHTML = "";
    return;
  }

  const level = overall.overall_alert_level || "normal";
  const story = (overall.story || [])
    .map(
      (item) =>
        `<div class="em-event"><span class="et">${esc(item.time)}</span>` +
        `<span class="es">${esc(item.subject)}</span><span class="ea">${esc(item.action)}</span></div>`
    )
    .join("");
  const subjects = (overall.subjects || []).map((subject) => `<li>${esc(subject)}</li>`).join("");

  element.hidden = false;
  element.className = `em-overall ${level}`;
  element.innerHTML =
    `<div class="em-window-head"><span class="em-otitle">📋 整段事件总结（跨窗整合）</span>` +
    `<span class="em-badge ${esc(level)}">${esc(level)}</span></div>` +
    `<div class="em-summary">${esc(overall.overall_summary)}</div>` +
    (overall.notification ? `<div class="em-notify">🔔 ${esc(overall.notification)}</div>` : "") +
    (story ? `<div class="em-events">${story}</div>` : "") +
    (subjects ? `<ul class="em-subjects">${subjects}</ul>` : "");
}

export function renderResult(data) {
  setLastPayload(data);
  $("resultTools").hidden = false;
  $("jsonView").hidden = true;
  $("btnToggleJson").textContent = "查看原始 JSON";
  $("btnSendLlm").hidden = !data.dry_run;

  renderOverall(data.overall);
  renderMeta(data);
  renderConfigSummary(data);
  renderReidDiagnostics(data);
  renderTimings(data);
  $("tracks").innerHTML = renderSubjectGallery(data);

  resetKeyframeRegistry();
  $("timeline").innerHTML = renderTimeline(data);
}
