import { renderSubjectGallery } from "./identity-gallery.js";
import { common, labelLevel, t } from "./i18n.js";
import {
  clearLastPayload,
  clearPromptArtifact,
  resetKeyframeRegistry,
  setLastPayload,
} from "./state.js";
import { renderTimeline } from "./timeline.js";
import { $, baseName, esc } from "./utils.js";

const stageLabel = (key) => t(`timings.stages.${key}`);

export function setStatus(message, isError = false) {
  const element = $("status");
  element.textContent = message;
  element.style.color = isError ? "var(--alert)" : "var(--muted)";
}

export function renderSamples(data, failed = false) {
  const select = $("sampleSelect");
  const count = $("sampleCount");

  if (failed) {
    select.innerHTML = `<option value="">${esc(t("samples.load_failed"))}</option>`;
    if (count) count.textContent = "";
    return;
  }

  const samples = data.samples || [];
  select.innerHTML = "";
  samples.forEach((sample) => {
    const option = document.createElement("option");
    option.value = sample.name;
    const title = sample.provenance?.title || sample.name;
    option.textContent = `${title} (${sample.size_mb} MB)`;
    select.appendChild(option);
  });

  if (count) count.textContent = samples.length ? t("samples.count", { count: samples.length }) : "";
  if (!samples.length) select.innerHTML = `<option value="">${esc(t("samples.empty"))}</option>`;
}

export function setBackendIndicator(online) {
  const element = $("backendStatus");
  if (!element) return;
  element.className = `em-svc ${online ? "online" : "offline"}`;
  element.lastChild.textContent = online ? t("service.online") : t("service.offline");
}

export function prepareForRun() {
  clearLastPayload();
  clearPromptArtifact();
  resetKeyframeRegistry();
  $("empty").style.display = "none";
  $("overall").hidden = true;
  $("cfgSummary").hidden = true;
  $("reidDiagnostics").hidden = true;
  $("timings").hidden = true;
  $("resultTools").hidden = true;
  $("chatPanel").hidden = true;
  $("chatMessages").innerHTML = "";
  $("chatPanel").dataset.runId = "";
  $("timeline").innerHTML = "";
  $("tracks").innerHTML = "";
  $("meta").innerHTML = "";
  $("promptView").hidden = true;
  $("promptView").textContent = "";
}

export function showRunFailure(message, detail = null) {
  $("empty").style.display = "block";
  $("empty").textContent = t("results.failure_prefix", { message });
  if (detail?.body_reid_timing) {
    renderReidDiagnostics({
      body_reid_timing: detail.body_reid_timing,
      config_used: detail.config_used || {},
    });
  }
}

function renderMeta(data) {
  const configUsed = data.config_used || {};
  const trackingFps = data.tracking_fps || data.fps;
  const keyframeLimit = data.max_keyframes || 8;
  const withBody = configUsed.with_body ?? data.with_body ?? true;
  const dimText = data.reid_dim ? esc(t("results.meta_dim", { dim: data.reid_dim })) : "";
  const reidText = withBody
    ? t("results.meta_body_reid", {
      backend: esc(data.reid_backend || configUsed.reid_backend || t("results.meta_unknown")),
      dim: dimText,
    })
    : t("results.meta_body_reid_off");
  $("meta").innerHTML =
    `${t("results.meta_video")} <b>${esc(baseName(data.video))}</b> · ${t("results.meta_rates", {
      frames: data.frames_total,
      tracking_fps: trackingFps,
      semantic_frames: data.semantic_frames_total || data.frames_total,
      semantic_fps: data.fps,
      keyframes: keyframeLimit,
    })} · ` +
    `${t("results.meta_windows", { count: (data.windows || []).length })} · ${t("results.meta_tracker", {
      tracker: esc(data.tracker_backend || configUsed.track_backend || "botsort_reid"),
    })} · ` +
    `${reidText} · ` +
    `${t("results.meta_model", { model: esc(data.model) })}${data.dry_run ? ` · <b>${t("results.meta_dry_run")}</b>` : ""} · ${t("results.meta_elapsed", { seconds: data.elapsed_seconds })}`;
}

function renderConfigSummary(data) {
  const configUsed = data.config_used || {};
  const chips = [];
  const on = (enabled) => (enabled ? common("on") : common("off"));
  const withBody = configUsed.with_body ?? data.with_body ?? true;
  const withGait = data.with_gait ?? configUsed.with_gait;
  const bodyMode = configUsed.reid_consistency_enabled
    ? t("results.config_body_mode_topk", { topk: configUsed.reid_decision_top_k })
    : t("results.config_body_mode_top1");
  const faceDetail = configUsed.with_face
    ? t("results.config_face_detail", {
      backend: esc(configUsed.face_rec_backend || "arcface"),
      superres: configUsed.face_superres && configUsed.face_superres !== "off" ? t("results.config_face_superres") : "",
      cue3d: configUsed.face_3d_cue ? t("results.config_face_3d") : "",
    })
    : "";

  chips.push(
    t("results.config_body", {
      state: on(withBody),
      detail: withBody
        ? t("results.config_body_detail", {
          backend: esc(data.reid_backend || configUsed.reid_backend || "auto"),
          mode: bodyMode,
        })
        : "",
    })
  );
  chips.push(
    t("results.config_face", {
      state: on(configUsed.with_face),
      detail: configUsed.with_face ? faceDetail : "",
    })
  );
  chips.push(t("results.config_gait", { state: on(withGait) }));
  chips.push(t("results.config_ocr", { state: configUsed.with_ocr ? esc(data.ocr_backend || common("on")) : common("off") }));
  chips.push(t("results.config_objects", { state: on(configUsed.with_objects) }));
  chips.push(t("results.config_ai", {
    model: esc(data.model || configUsed.llm_model || common("unknown")),
  }));
  if (data.gait_error) chips.push(`<span class="warn">${esc(t("results.config_warn_gait", { message: data.gait_error }))}</span>`);
  if (data.ocr_error) chips.push(`<span class="warn">${esc(t("results.config_warn_ocr", { message: data.ocr_error }))}</span>`);

  $("cfgSummary").hidden = false;
  $("cfgSummary").innerHTML = `${t("results.config_title")} ${chips.map((chip) => `<span class="em-cfgchip">${chip}</span>`).join(" ")}`;
}

function parseTimingValue(value) {
  if (typeof value !== "number" && typeof value !== "string") return null;
  const normalized = typeof value === "string" ? value.trim() : value;
  if (normalized === "") return null;
  const seconds = Number(normalized);
  return Number.isFinite(seconds) && seconds >= 0 ? seconds : null;
}

function formatTimingDuration(seconds) {
  if (seconds === 0) return "0ms";
  if (seconds < 0.001) return `${(seconds * 1000000).toFixed(0)}μs`;
  if (seconds < 1) return `${(seconds * 1000).toFixed(seconds < 0.01 ? 1 : 0)}ms`;
  return `${seconds.toFixed(2)}s`;
}

export function renderTimings(data) {
  const stageTimings = data.stage_timings;
  if (!stageTimings || typeof stageTimings !== "object" || Array.isArray(stageTimings)) {
    $("timings").hidden = true;
    return;
  }

  const rows = Object.entries(stageTimings)
    .map(([key, rawValue]) => ({
      key,
      value: parseTimingValue(rawValue),
    }))
    .sort((left, right) => {
      if (left.value === null) return right.value === null ? 0 : 1;
      if (right.value === null) return -1;
      return right.value - left.value;
    });

  if (!rows.length) {
    $("timings").hidden = true;
    return;
  }

  const positiveRows = rows.filter((row) => row.value !== null && row.value > 0);
  const measuredTotal = positiveRows.reduce((sum, row) => sum + row.value, 0);
  const bars = rows
    .map((row, index) => {
      const stageText = stageLabel(row.key) || row.key;
      const stage = esc(stageText);
      if (row.value === null) {
        return (
          `<div class="em-tbar is-unavailable">` +
          `<span class="em-tbar-label">${stage}</span>` +
          `<span class="em-tbar-track" role="img" aria-label="${esc(t("results.timings_invalid_bar_label", { stage: stageText }))}"></span>` +
          `<span class="em-tbar-val">${esc(t("results.timings_invalid"))}</span></div>`
        );
      }

      if (row.value === 0) {
        return (
          `<div class="em-tbar is-zero">` +
          `<span class="em-tbar-label">${stage}</span>` +
          `<span class="em-tbar-track" role="img" aria-label="${esc(t("results.timings_zero_bar_label", { stage: stageText }))}"></span>` +
          `<span class="em-tbar-val">${esc(t("results.timings_zero"))}</span></div>`
        );
      }

      const share = (row.value / measuredTotal) * 100;
      const duration = formatTimingDuration(row.value);
      const tiny = share < 1;
      const fill = `<span class="em-tbar-fill" aria-hidden="true" style="width:${share.toFixed(3)}%"></span>`;
      const marker = tiny
        ? `<span class="em-tbar-marker" aria-hidden="true" title="${esc(t("results.timings_tiny_marker"))}"></span>`
        : "";
      return (
        `<div class="em-tbar ${index === 0 ? "top" : ""}">` +
        `<span class="em-tbar-label">${stage}</span>` +
        `<span class="em-tbar-track${tiny ? " has-tiny-marker" : ""}" role="img" aria-label="${esc(t("results.timings_bar_label", {
          stage: stageText,
          duration,
          share: share.toFixed(1),
        }))}">${fill}${marker}</span>` +
        `<span class="em-tbar-val">${duration} · ${share.toFixed(1)}%</span></div>`
      );
    })
    .join("");

  const elapsedSeconds = parseTimingValue(data.elapsed_seconds);
  const elapsedLabel = elapsedSeconds === null
    ? esc(t("results.timings_invalid"))
    : esc(t("results.timings_total", { seconds: elapsedSeconds }));
  const measuredTotalLabel = esc(t("results.timings_measured_total", {
    seconds: formatTimingDuration(measuredTotal),
  }));
  $("timings").hidden = false;
  $("timings").innerHTML =
    `<details class="em-timings-details">` +
    `<summary aria-label="${esc(t("results.timings_toggle_label"))}" aria-describedby="timingsMeasuredNote">` +
    `<span class="em-timings-title">${esc(t("results.timings_title"))}</span>` +
    `<span class="em-timings-action em-timings-expand" aria-hidden="true">${esc(t("results.timings_expand"))}</span>` +
    `<span class="em-timings-action em-timings-collapse" aria-hidden="true">${esc(t("results.timings_collapse"))}</span></summary>` +
    `<div class="em-timings-content">` +
    `<div id="timingsMeasuredNote" class="em-timings-note">${esc(t("results.timings_measured_note"))}</div>` +
    `<div class="em-timings-head"><span>${esc(t("results.timings_relative_hint"))}</span>` +
    `<span class="tot">${measuredTotalLabel} · ${elapsedLabel}</span></div>${bars}</div></details>`;
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
    tracking: t("results.reid_purpose_tracking"),
    identity_gallery: t("results.reid_purpose_identity_gallery"),
    face_consistency: t("results.reid_purpose_face_consistency"),
    unspecified: t("results.reid_purpose_unspecified"),
  };
  const breakdown = Object.entries(timing.by_purpose || {})
    .filter(([, value]) => Number(value?.call_count) > 0)
    .map(
      ([purpose, value]) =>
        `${purposeLabels[purpose] || purpose} ${value.call_count}/${formatMs(value.total_ms)}`
    )
    .join(" · ");
  const backend = timing.backend || data.reid_backend || common("unknown");
  const device = timing.device || data.runtime?.reid_device || common("unknown");
  const values = calls
    ? t("results.reid_calls", {
      calls,
      total: formatMs(timing.total_ms),
      mean: formatMs(timing.mean_ms),
      p95: formatMs(timing.p95_ms),
    })
    : t("results.reid_zero");
  const failures =
    `<span class="em-reid-fail ${failedCalls ? "has-failures" : ""}">` +
    `${t("results.reid_failures", { count: failedCalls })}</span>`;

  element.hidden = false;
  element.innerHTML =
    `<div class="em-reid-head">${esc(t("results.reid_title"))} <span>${esc(t("results.reid_subtitle"))}</span></div>` +
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
    `<div class="em-window-head"><span class="em-otitle">${esc(t("results.overall_title"))}</span>` +
    `<span class="em-badge ${esc(level)}">${esc(labelLevel(level))}</span></div>` +
    `<div class="em-summary">${esc(overall.overall_summary)}</div>` +
    (overall.notification ? `<div class="em-notify">${esc(t("results.overall_notification_prefix"))} ${esc(overall.notification)}</div>` : "") +
    (story ? `<div class="em-events">${story}</div>` : "") +
    (subjects ? `<ul class="em-subjects">${subjects}</ul>` : "");
}

export function renderResult(data) {
  clearPromptArtifact();
  setLastPayload(data);
  $("resultTools").hidden = false;
  $("promptView").hidden = true;
  $("promptView").textContent = "";
  $("btnViewPrompt").textContent = t("results.view_prompt_show");
  $("btnSendLlm").hidden = !data.dry_run;
  $("dryRunAction").hidden = !data.dry_run;
  const chatPanel = $("chatPanel");
  if (data.run_id) {
    if (chatPanel.dataset.runId !== data.run_id) $("chatMessages").innerHTML = "";
    chatPanel.dataset.runId = data.run_id;
    chatPanel.hidden = false;
    const selection = data.llm_selection || {};
    $("chatModelStatus").textContent =
      `${selection.requested || "auto"} → ${selection.model || data.model || t("chat.waiting")}`;
  } else {
    chatPanel.hidden = true;
  }

  renderOverall(data.overall);
  renderMeta(data);
  renderConfigSummary(data);
  renderReidDiagnostics(data);
  renderTimings(data);
  $("tracks").innerHTML = renderSubjectGallery(data);

  resetKeyframeRegistry();
  $("timeline").innerHTML = renderTimeline(data);
}
