import { downloadJson, openLightbox, closeLightbox, sendDryRunToLlm, toggleJson } from "./actions.js";
import { health, listSamples, runAnalysis } from "./api.js";
import { finishProgress, startProgress } from "./progress.js";
import {
  prepareForRun,
  renderResult,
  renderSamples,
  setBackendIndicator,
  setStatus,
  showRunFailure,
} from "./render.js";
import { closeSettings, collectAnalysisRequest, openSettings, wireDropzone } from "./settings.js";
import { $ } from "./utils.js";

function tickClock() {
  $("clock").textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false });
}

async function loadSampleOptions() {
  try {
    renderSamples(await listSamples());
  } catch (_) {
    renderSamples({}, true);
  }
}

async function checkBackend() {
  setBackendIndicator(await health().catch(() => false));
}

async function run() {
  const request = collectAnalysisRequest();
  if (!request.file && !request.sample) {
    setStatus("请先选择样片或上传视频", true);
    return;
  }

  $("btnRun").disabled = true;
  prepareForRun();
  startProgress(request.dryRun);
  setStatus(`⏳ 处理中… ${request.dryRun ? "（dry-run，不调 LLM）" : "（含 gpt-4o，约 1 分钟）"}`);

  const startedAt = Date.now();
  try {
    const data = await runAnalysis(request.formData);
    finishProgress(true);
    renderResult(data);
    setStatus(`✓ 完成，用时 ${((Date.now() - startedAt) / 1000).toFixed(1)}s`);
  } catch (error) {
    finishProgress(false);
    setStatus("✗ 失败：" + error.message, true);
    showRunFailure(error.message);
  } finally {
    $("btnRun").disabled = false;
  }
}

function bindEvents() {
  $("btnRun").addEventListener("click", run);
  $("btnSendLlm").addEventListener("click", sendDryRunToLlm);
  $("btnDownloadJson").addEventListener("click", downloadJson);
  $("btnToggleJson").addEventListener("click", toggleJson);
  $("btnSettings").addEventListener("click", openSettings);
  $("btnCloseSettings").addEventListener("click", closeSettings);
  $("btnApplySettings").addEventListener("click", closeSettings);
  $("settingsOverlay").addEventListener("click", closeSettings);
  $("timeline").addEventListener("click", (event) => {
    const frame = event.target.closest(".em-frame");
    if (frame && frame.dataset.kf != null) openLightbox(Number(frame.dataset.kf));
  });
  $("lightboxClose").addEventListener("click", closeLightbox);
  $("lightbox").addEventListener("click", (event) => {
    if (event.target === $("lightbox")) closeLightbox();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closeLightbox();
      closeSettings();
    }
  });
}

window.addEventListener("error", (event) => {
  try {
    setStatus("✗ 前端渲染错误：" + (event.message || event.error || "unknown"), true);
  } catch (_) {
    // 页面还没初始化时忽略。
  }
});

tickClock();
setInterval(tickClock, 1000);
wireDropzone();
bindEvents();
checkBackend();
loadSampleOptions();
