import { renderSubjectGallery } from "./identity-gallery.js";
import { clearLastPayload, resetKeyframeRegistry, setLastPayload } from "./state.js";
import { renderTimeline } from "./timeline.js";
import { $, baseName, esc } from "./utils.js";

const STAGE_CN = {
  extract_frames: "① 抽帧",
  detect_track: "② YOLO 检测 + 跟踪",
  gait_collect: "· 步态采集（Pose+Seg 逐帧）",
  reid_identify: "③ 人形 ReID 认人",
  face: "· 人脸分支（检测+对齐+AdaFace）",
  gait_embed: "· 步态向量提取",
  merge_fusion_thumb: "⑥ 三路融合 + 头像",
  windows_select: "④⑤ 选帧 / 分窗",
  windows_llm: "⑦ gpt-4o 事件理解",
  overall_summary: "· 整段总结",
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
  $("timings").hidden = true;
  $("resultTools").hidden = true;
  $("timeline").innerHTML = "";
  $("tracks").innerHTML = "";
  $("meta").innerHTML = "";
  $("jsonView").hidden = true;
  $("jsonView").textContent = "";
}

export function showRunFailure(message) {
  $("empty").style.display = "block";
  $("empty").textContent = "处理失败：" + message;
}

function renderMeta(data) {
  const configUsed = data.config_used || {};
  $("meta").innerHTML =
    `视频 <b>${esc(baseName(data.video))}</b> · ${data.frames_total} 帧 @ ${data.fps}fps · ` +
    `${(data.windows || []).length} 个事件窗 · Tracker <b>${esc(data.tracker_backend || configUsed.track_backend || "botsort_reid")}</b> · ` +
    `ReID <b>${esc(data.reid_backend)}</b>(${data.reid_dim}d) · ` +
    `模型 <b>${esc(data.model)}</b>${data.dry_run ? " · <b>dry-run</b>" : ""} · ${data.elapsed_seconds}s`;
}

function renderConfigSummary(data) {
  const configUsed = data.config_used || {};
  const chips = [];
  const on = (enabled) => (enabled ? "on" : "off");

  chips.push(
    `人脸 <b>${on(configUsed.with_face)}</b>` +
      (configUsed.with_face
        ? `（${esc(configUsed.face_rec_backend || "adaface")}` +
          `${configUsed.face_superres && configUsed.face_superres !== "off" ? "+超分" : ""}` +
          `${configUsed.face_3d_cue ? "+3D" : ""}）`
        : "")
  );
  chips.push(`步态 <b>${on(configUsed.with_gait)}</b>`);
  chips.push(`OCR <b>${configUsed.with_ocr ? esc(data.ocr_backend || "on") : "off"}</b>`);
  chips.push(`物体 <b>${on(configUsed.with_objects)}</b>`);
  chips.push(
    `ReID <b>${esc(configUsed.reid_backend || "auto")}</b>` +
      `${configUsed.reid_consistency_enabled ? `（top-${configUsed.reid_decision_top_k} 一致性）` : "（top-1）"}`
  );
  if (data.gait_error) chips.push(`<span class="warn">步态告警: ${esc(data.gait_error)}</span>`);
  if (data.ocr_error) chips.push(`<span class="warn">OCR告警: ${esc(data.ocr_error)}</span>`);

  $("cfgSummary").hidden = false;
  $("cfgSummary").innerHTML = `本次生效： ${chips.map((chip) => `<span class="em-cfgchip">${chip}</span>`).join(" ")}`;
}

function renderTimings(data) {
  const stageTimings = data.stage_timings || {};
  const rows = Object.entries(stageTimings)
    .map(([key, value]) => ({ key, value: +value }))
    .filter((row) => Number.isFinite(row.value) && row.value >= 0)
    .sort((left, right) => right.value - left.value);

  if (!rows.length) {
    $("timings").hidden = true;
    return;
  }

  const max = Math.max(...rows.map((row) => row.value), 0.01);
  const total = rows.reduce((sum, row) => sum + row.value, 0);
  const bars = rows
    .map((row, index) => {
      const pct = (row.value / max) * 100;
      const share = total > 0 ? ((row.value / total) * 100).toFixed(0) : "0";
      return (
        `<div class="em-tbar ${index === 0 ? "top" : ""}">` +
        `<span class="em-tbar-label">${esc(STAGE_CN[row.key] || row.key)}</span>` +
        `<span class="em-tbar-track"><span class="em-tbar-fill" style="width:${pct.toFixed(1)}%"></span></span>` +
        `<span class="em-tbar-val">${row.value.toFixed(1)}s · ${share}%</span></div>`
      );
    })
    .join("");

  $("timings").hidden = false;
  $("timings").innerHTML =
    `<div class="em-timings-head">⏱ 本次各阶段实测耗时` +
    `<span class="tot">总 ${esc(data.elapsed_seconds)}s · 本地 CPU</span></div>${bars}`;
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
  renderTimings(data);
  $("tracks").innerHTML = renderSubjectGallery(data);

  resetKeyframeRegistry();
  $("timeline").innerHTML = renderTimeline(data);
}
