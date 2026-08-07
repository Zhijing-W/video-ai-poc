import { downloadJson, openLightbox, closeLightbox, sendDryRunToLlm, toggleJson } from "./actions.js";
import { clockLocale, t } from "./i18n.js";
import {
  health,
  listAiModels,
  listReidBackends,
  listSamples,
  listSuperresBackends,
  runAnalysis,
} from "./api.js?v=20260807-ai-model-selector";
import { finishProgress, startProgress } from "./progress.js";
import {
  prepareForRun,
  renderResult,
  renderSamples,
  setBackendIndicator,
  setStatus,
  showRunFailure,
} from "./render.js?v=20260807-ai-model-selector";
import {
  closeSettings,
  collectAnalysisRequest,
  openSettings,
  renderReidBackends,
  renderAiModels,
  renderSuperresBackends,
  wireDropzone,
  wireSuperresSettings,
} from "./settings.js?v=20260807-ai-model-selector";
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

async function loadAiModelOptions() {
  try {
    renderAiModels(await listAiModels());
  } catch (_) {
    renderAiModels();
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
bindEvents();
checkBackend();
loadSampleOptions();
loadSuperresOptions();
loadReidOptions();
loadAiModelOptions();
