import {
  closeLightbox,
  downloadPrompt,
  openLightbox,
  resetPromptView,
  sendDryRunToLlm,
  viewPrompt,
} from "./actions.js";
import { clockLocale, reportLanguage, t } from "./i18n.js";
import {
  health,
  listLlmModels,
  listReidBackends,
  listSamples,
  listSuperresBackends,
  runAnalysis,
} from "./api.js?v=20260808-foundry-routing";
import { wireChat } from "./chat.js?v=20260808-foundry-routing";
import { finishProgress, startProgress } from "./progress.js";
import {
  prepareForRun,
  renderResult,
  renderSamples,
  setBackendIndicator,
  setStatus,
  showRunFailure,
} from "./render.js?v=20260810-sample-presets";
import {
  closeSettings,
  collectAnalysisRequest,
  applySelectedSamplePreset,
  openSettings,
  renderReidBackends,
  renderLlmModels,
  renderSuperresBackends,
  wireDropzone,
  wireSuperresSettings,
} from "./settings.js?v=20260810-sample-presets";
import { $ } from "./utils.js";

function tickClock() {
  $("clock").textContent = new Date().toLocaleTimeString(clockLocale(), { hour12: false });
}

async function loadSampleOptions() {
  try {
    renderSamples(await listSamples());
  } catch (_) {
    renderSamples({}, true);
  }
}

async function loadSuperresOptions() {
  try {
    renderSuperresBackends(await listSuperresBackends());
  } catch (_) {
    renderSuperresBackends();
  }
}

async function loadReidOptions() {
  try {
    renderReidBackends(await listReidBackends());
  } catch (_) {
    renderReidBackends();
  }
}

async function loadLlmOptions() {
  try {
    renderLlmModels(await listLlmModels(reportLanguage()));
  } catch (_) {
    renderLlmModels();
  }
}

async function checkBackend() {
  setBackendIndicator(await health().catch(() => false));
}

async function run() {
  const request = collectAnalysisRequest();
  if (!request.file && !request.sample) {
    setStatus(t("status.select_input"), true);
    return;
  }

  $("btnRun").disabled = true;
  prepareForRun();
  startProgress(request.dryRun);
  setStatus(
    t("status.processing", {
      mode: request.dryRun ? t("status.processing_dry_run") : t("status.processing_full"),
    })
  );

  const startedAt = Date.now();
  try {
    const data = await runAnalysis(request.formData);
    finishProgress(true);
    renderResult(data);
    setStatus(t("status.completed", {
      seconds: ((Date.now() - startedAt) / 1000).toFixed(1),
    }));
  } catch (error) {
    finishProgress(false);
    setStatus(t("status.failed", { message: error.message }), true);
    showRunFailure(error.message, error.detail);
  } finally {
    $("btnRun").disabled = false;
  }
}

function bindEvents() {
  $("btnRun").addEventListener("click", run);
  $("sampleSelect").addEventListener("change", applySelectedSamplePreset);
  $("btnSendLlm").addEventListener("click", sendDryRunToLlm);
  $("btnDownloadPrompt").addEventListener("click", downloadPrompt);
  $("btnViewPrompt").addEventListener("click", viewPrompt);
  $("promptFormat").addEventListener("change", resetPromptView);
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
    setStatus(t("status.frontend_error", {
      message: event.message || event.error || "unknown",
    }), true);
  } catch (_) {
    // 页面还没初始化时忽略。
  }
});

tickClock();
setInterval(tickClock, 1000);
wireDropzone();
wireSuperresSettings();
wireChat();
bindEvents();
checkBackend();
loadSampleOptions();
loadSuperresOptions();
loadReidOptions();
loadLlmOptions();
