import { $, esc } from "./utils.js";
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

export function stagesFor(dryRun) {
  const stages = [
    { key: "extract", label: "抽帧", weight: 12 },
    { key: "track", label: "检测 / 跟踪", weight: 22 },
    { key: "reid", label: "认人 ReID / 人脸 / 步态", weight: 30 },
    { key: "window", label: "选帧 / 分窗 / 融合", weight: 12 },
  ];
  if (!dryRun) stages.push({ key: "llm", label: "多帧事件理解 gpt-4o", weight: 24 });
  return stages;
}

export function markStep(index) {
  setProgressState({ stepIndex: index });
  document.querySelectorAll("#progressSteps .em-step").forEach((element) => {
    const stepIndex = Number(element.dataset.i);
    element.classList.toggle("done", stepIndex < index);
    element.classList.toggle("active", stepIndex === index);
  });
}

export function setProgress(pct, stage) {
  setProgressState({ pct });
  $("progressBar").style.width = Math.max(2, Math.min(100, pct)).toFixed(1) + "%";
  if (stage) $("progressStage").textContent = `⏳ ${stage}…`;
}

export function startProgress(dryRun) {
  clearProgressHandles();
  resetProgressState();

  const stages = stagesFor(dryRun);
  const startedAt = Date.now();
  $("progress").hidden = false;
  $("progressSteps").innerHTML = stages
    .map((stage, index) => `<span class="em-step" data-i="${index}">${esc(stage.label)}</span>`)
    .join('<span class="em-step-sep">›</span>');

  setProgressState({ startedAt, stages, stepIndex: 0, pct: 0 });
  setProgress(0, stages[0].label);
  markStep(0);

  const clock = setInterval(() => {
    const progress = getProgressState();
    $("progressTimer").textContent = ((Date.now() - progress.startedAt) / 1000).toFixed(1) + "s";
  }, 100);

  let cumulative = 0;
  const caps = stages.map((stage) => (cumulative += stage.weight));
  const timer = setInterval(() => {
    const progress = getProgressState();
    const cap = Math.min(92, caps[progress.stepIndex] || 92);
    if (progress.pct < cap) {
      const nextPct = Math.min(cap, progress.pct + Math.max(0.3, (cap - progress.pct) * 0.08));
      setProgress(nextPct, stages[Math.min(progress.stepIndex, stages.length - 1)].label);
      return;
    }
    if (progress.stepIndex < stages.length - 1) markStep(progress.stepIndex + 1);
  }, 260);

  setProgressState({ clock, timer });
}

export function finishProgress(ok) {
  const progress = getProgressState();
  if (progress.timer) clearInterval(progress.timer);
  if (progress.clock) clearInterval(progress.clock);
  setProgressState({ timer: null, clock: null });

  setProgress(100, ok ? "完成" : "结束");
  $("progressStage").textContent = ok ? "✓ 处理完成" : "✗ 处理结束";
  document.querySelectorAll("#progressSteps .em-step").forEach((element) => {
    element.classList.remove("active");
    element.classList.add("done");
  });

  const hideTimer = setTimeout(() => {
    $("progress").hidden = true;
    resetProgressState();
  }, ok ? 650 : 1400);

  setProgressState({ hideTimer });
}
