import { t } from "./i18n.js";
import { $ } from "./utils.js";
import {
  getProgressState,
  resetProgressState,
  setProgressState,
} from "./state.js";

function clearProgressHandles() {
  const progress = getProgressState();
  if (progress.timer) clearInterval(progress.timer);
  if (progress.clock) clearInterval(progress.clock);
  if (progress.hideTimer) clearTimeout(progress.hideTimer);
}

export function setProgress(pct, stage) {
  setProgressState({ pct });
  $("progressBar").style.width = Math.max(2, Math.min(100, pct)).toFixed(1) + "%";
  if (stage) $("progressStage").textContent = `⏳ ${stage}…`;
}

export function startProgress(_dryRun) {
  clearProgressHandles();
  resetProgressState();

  const startedAt = Date.now();
  $("progress").hidden = false;
  $("progressBar").classList.add("indeterminate");
  $("progressBar").style.width = "";
  $("progressStage").textContent = `⏳ ${t("progress.running")}…`;
  $("progressSteps").innerHTML = "";

  const clock = setInterval(() => {
    const progress = getProgressState();
    $("progressTimer").textContent = ((Date.now() - progress.startedAt) / 1000).toFixed(1) + "s";
  }, 100);

  setProgressState({ startedAt, clock, timer: null });
}

export function finishProgress(ok) {
  const progress = getProgressState();
  if (progress.timer) clearInterval(progress.timer);
  if (progress.clock) clearInterval(progress.clock);
  setProgressState({ timer: null, clock: null });

  $("progressBar").classList.remove("indeterminate");
  setProgress(100, ok ? t("progress.done") : t("progress.ended"));
  $("progressStage").textContent = ok ? t("progress.success") : t("progress.failure");
  const step = document.querySelector("#progressSteps .em-step");
  if (step) {
    step.classList.remove("active");
    if (ok) step.classList.add("done");
  }

  const hideTimer = setTimeout(() => {
    $("progress").hidden = true;
    resetProgressState();
  }, ok ? 650 : 1400);

  setProgressState({ hideTimer });
}
