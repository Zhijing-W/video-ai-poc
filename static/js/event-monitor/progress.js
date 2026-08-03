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
  $("progressStage").textContent = "⏳ 服务端正在分析（完成后返回实测阶段耗时）…";
  $("progressSteps").innerHTML =
    '<span class="em-step active">同步任务处理中；不显示估算阶段或虚拟百分比</span>';

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
  setProgress(100, ok ? "完成" : "结束");
  $("progressStage").textContent = ok ? "✓ 处理完成" : "✗ 处理结束";
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
