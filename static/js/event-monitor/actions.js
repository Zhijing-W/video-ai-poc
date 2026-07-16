import { completeDryRun } from "./api.js";
import { startProgress, finishProgress } from "./progress.js";
import { renderResult, setStatus } from "./render.js";
import { getObjectiveValue } from "./settings.js";
import { getKeyframe, getLastPayload } from "./state.js";
import { boxesHtml } from "./timeline.js";
import { $, baseName } from "./utils.js";

export function cleanedPayload() {
  const payload = getLastPayload();
  if (!payload) return null;

  const copy = JSON.parse(JSON.stringify(payload));
  (copy.windows || []).forEach((windowData) => {
    (windowData.keyframes || []).forEach((keyframe) => {
      if (keyframe.image) keyframe.image = "<data-uri omitted>";
    });
  });
  Object.values(copy.tracks || {}).forEach((track) => {
    if (track && track.thumb) track.thumb = "<data-uri omitted>";
  });
  return copy;
}

export function downloadJson() {
  const payload = cleanedPayload();
  if (!payload) return;

  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  const stem = baseName(payload.video).replace(/\.[^.]+$/, "") || "result";
  anchor.download = `event-monitor_${stem}${payload.dry_run ? "_dryrun" : ""}.json`;
  anchor.click();
  URL.revokeObjectURL(anchor.href);
}

export function toggleJson() {
  const element = $("jsonView");
  if (element.hidden) {
    element.textContent = JSON.stringify(cleanedPayload(), null, 2);
    element.hidden = false;
    $("btnToggleJson").textContent = "收起 JSON";
    return;
  }

  element.hidden = true;
  $("btnToggleJson").textContent = "查看原始 JSON";
}

export function openLightbox(index) {
  const keyframe = getKeyframe(index);
  if (!keyframe) return;

  $("lightboxStage").innerHTML =
    `<img src="${keyframe.image}"/><div class="em-boxes">${boxesHtml(keyframe.boxes)}</div>`;
  $("lightboxCap").textContent = keyframe.caption || "";
  $("lightbox").hidden = false;
}

export function closeLightbox() {
  $("lightbox").hidden = true;
  $("lightboxStage").innerHTML = "";
}

export async function sendDryRunToLlm() {
  const payload = getLastPayload();
  if (!payload || !payload.dry_run) return;

  $("btnSendLlm").disabled = true;
  const startedAt = Date.now();
  startProgress(false);
  setStatus("⏳ 正在复用 dry-run 的关键帧和身份上下文调用大模型…");

  try {
    const data = await completeDryRun(payload, getObjectiveValue());
    finishProgress(true);
    renderResult(data);
    setStatus(`✓ 大模型事件理解完成，用时 ${((Date.now() - startedAt) / 1000).toFixed(1)}s`);
  } catch (error) {
    finishProgress(false);
    setStatus("✗ 调用大模型失败：" + error.message, true);
  } finally {
    $("btnSendLlm").disabled = false;
  }
}
