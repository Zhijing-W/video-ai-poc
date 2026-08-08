import { completeDryRun, getRunPrompt, runPromptUrl } from "./api.js";
import { reportLanguage, t } from "./i18n.js";
import { startProgress, finishProgress } from "./progress.js";
import { renderResult, setStatus } from "./render.js";
import {
  getObjectiveValue,
  getAnalysisModelValue,
} from "./settings.js?v=20260808-foundry-routing";
import {
  clearPromptArtifact,
  getKeyframe,
  getLastPayload,
  getPromptArtifact,
  setPromptArtifact,
} from "./state.js";
import { boxesHtml } from "./timeline.js";
import { $ } from "./utils.js";

function selectedPromptFormat() {
  return $("promptFormat").value;
}

function promptRunId() {
  return getLastPayload()?.run_id || null;
}

export function downloadPrompt() {
  const runId = promptRunId();
  if (!runId) return;
  const anchor = document.createElement("a");
  const format = selectedPromptFormat();
  anchor.href = runPromptUrl(runId, format);
  anchor.download = `event-monitor_${runId}_prompt.${format}`;
  anchor.click();
}

export async function viewPrompt() {
  const element = $("promptView");
  if (!element.hidden) {
    element.hidden = true;
    $("btnViewPrompt").textContent = t("results.view_prompt_show");
    return;
  }

  const runId = promptRunId();
  if (!runId) return;
  const format = selectedPromptFormat();
  const cached = getPromptArtifact();
  $("btnViewPrompt").disabled = true;
  try {
    const content = cached?.format === format
      ? cached.content
      : await getRunPrompt(runId, format);
    setPromptArtifact(format, content);
    element.textContent = content;
    element.hidden = false;
    $("btnViewPrompt").textContent = t("results.view_prompt_hide");
  } finally {
    $("btnViewPrompt").disabled = false;
  }
}

export function resetPromptView() {
  clearPromptArtifact();
  $("promptView").hidden = true;
  $("promptView").textContent = "";
  $("btnViewPrompt").textContent = t("results.view_prompt_show");
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
  setStatus(t("status.reusing_dry_run"));

  try {
    const data = await completeDryRun(
      payload,
      getObjectiveValue(),
      reportLanguage(),
      getAnalysisModelValue()
    );
    finishProgress(true);
    renderResult(data);
    setStatus(t("status.llm_completed", {
      seconds: ((Date.now() - startedAt) / 1000).toFixed(1),
    }));
  } catch (error) {
    finishProgress(false);
    setStatus(t("status.llm_failed", { message: error.message }), true);
  } finally {
    $("btnSendLlm").disabled = false;
  }
}
