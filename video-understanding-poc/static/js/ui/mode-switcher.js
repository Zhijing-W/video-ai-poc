/**
 * 模式切换模块
 * 提取自 monitor.html.backup-refactor：负责监控/技术模式切换、共享视频挂载与两套控件镜像。
 */

import { appState, setCurrentMode } from '../core/state.js';
import { $ } from '../core/utils.js';

function moveSharedMedia(mode) {
  const video = $('video');
  const boxes = $('boxes');
  const dashStage = $('stage-dash');
  const techStage = $('stage');
  if (!video || !boxes) return;

  if (mode === 'monitor' && dashStage && video.parentElement !== dashStage) {
    dashStage.insertBefore(video, dashStage.firstChild);
    dashStage.insertBefore(boxes, video.nextSibling);
  }

  if (mode === 'tech' && techStage && video.parentElement !== techStage) {
    techStage.insertBefore(video, techStage.firstChild);
    techStage.insertBefore(boxes, video.nextSibling);
  }
}

function prepareSharedVideoStage() {
  const dashStage = $('stage-dash');
  if (!dashStage) return;

  const videoDash = $('video-dash');
  const boxesDash = $('boxes-dash');
  if (videoDash) videoDash.remove();
  if (boxesDash) boxesDash.remove();
}

function bindDashboardActionMirrors() {
  if (!$('btnStart-dash')) return;

  $('btnStart-dash').onclick = () => $('btnStart')?.click();
  $('btnStop-dash').onclick = () => $('btnStop')?.click();
  $('btnCompare-dash').onclick = () => $('btnCompare')?.click();
  $('btnCompareStop-dash').onclick = () => $('btnCompareStop')?.click();
}

function bindDashboardConfigMirrors() {
  if (!$('yoloModel-dash')) return;

  $('yoloModel-dash').onchange = () => {
    $('yoloModel').value = $('yoloModel-dash').value;
    $('yoloModel').dispatchEvent(new Event('change'));
  };
  $('llmModel-dash').onchange = () => {
    $('llmModel').value = $('llmModel-dash').value;
    $('llmModel').dispatchEvent(new Event('change'));
  };
  $('boxToggle-dash').onchange = () => $('boxToggle').checked = $('boxToggle-dash').checked;
  $('gateToggle-dash').onchange = () => $('gateToggle').checked = $('gateToggle-dash').checked;
  $('cruiseToggle-dash').onchange = () => $('cruiseToggle').checked = $('cruiseToggle-dash').checked;
  if ($('trackToggle-dash')) {
    $('trackToggle-dash').onchange = () => { $('trackToggle').checked = $('trackToggle-dash').checked; };
  }
  $('intervalVal-dash').oninput = () => {
    $('intervalVal').value = $('intervalVal-dash').value;
    $('intervalVal').dispatchEvent(new Event('input'));   // 触发 restartTimerIfRunning，运行中也即时生效
  };
  $('intervalUnit-dash').onchange = () => {
    $('intervalUnit').value = $('intervalUnit-dash').value;
    $('intervalUnit').dispatchEvent(new Event('change'));
  };
  $('auditEvery-dash').oninput = () => $('auditEvery').value = $('auditEvery-dash').value;
  $('target-dash').oninput = () => $('target').value = $('target-dash').value;

  if ($('smartFrame-dash')) {
    $('smartFrame-dash').onchange = () => { $('smartFrame').checked = $('smartFrame-dash').checked; };
  }
  if ($('fallbackSec-dash')) {
    $('fallbackSec-dash').oninput = () => { $('fallbackSec').value = $('fallbackSec-dash').value; };
  }

  document.querySelectorAll('input[name="cmode-dash"]').forEach((radio) => {
    radio.onchange = () => {
      const target = document.querySelector(`input[name="cmode"][value="${radio.value}"]`);
      if (target) target.checked = true;
    };
  });
}

export function switchMode(mode) {
  setCurrentMode(mode);
  localStorage.setItem('view-mode', mode);

  const monitorView = $('monitor-view');
  const techView = $('tech-view');
  const btnMonitorMode = $('btnMonitorMode');
  const btnTechMode = $('btnTechMode');

  if (mode === 'monitor') {
    if (monitorView) monitorView.style.display = 'block';
    if (techView) techView.style.display = 'none';
    btnMonitorMode?.classList.add('active');
    btnTechMode?.classList.remove('active');
  } else {
    if (monitorView) monitorView.style.display = 'none';
    if (techView) techView.style.display = 'block';
    btnMonitorMode?.classList.remove('active');
    btnTechMode?.classList.add('active');
  }

  moveSharedMedia(mode);
}

export function initModeSwitcher() {
  prepareSharedVideoStage();
  bindDashboardActionMirrors();
  bindDashboardConfigMirrors();

  $('btnMonitorMode')?.addEventListener('click', () => switchMode('monitor'));
  $('btnTechMode')?.addEventListener('click', () => switchMode('tech'));

  switchMode(appState.currentMode || 'tech');
}
