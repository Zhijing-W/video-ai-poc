/**
 * 视频控制模块
 * 提取自 monitor.html.backup-refactor：负责摄像头、视频上传、参考图与信号源 UI。
 */

import { appState } from '../core/state.js';
import { $, cloneFileList } from '../core/utils.js';
import { clearBoxes } from '../visualization/yolo-boxes.js';

function getVideoElement() {
  return $('video');
}

export function hasSignal() {
  const video = getVideoElement();
  return appState.stream !== null || !!(video && video.src && video.readyState >= 2);
}

export function setCamBtn(on) {
  const btnCam = $('btnCam');
  if (!btnCam) return;
  btnCam.innerHTML = on ? '■ 关闭摄像头' : '📷 摄像头';
  btnCam.classList.toggle('danger', on);
}

export function setSource(label, kind) {
  const emptyEl = $('empty');
  const emptyDashEl = $('empty-dash');
  if (emptyEl) emptyEl.style.display = 'none';
  if (emptyDashEl) emptyDashEl.style.display = 'none';

  const hudEl = $('hud');
  if (hudEl) hudEl.innerHTML = `<span class="chip src">${kind}</span><span class="chip" id="resChip">—</span>`;

  const video = getVideoElement();
  if (video) video.classList.add('on');

  if ($('btnStart')) $('btnStart').disabled = false;
  if ($('btnStart-dash')) $('btnStart-dash').disabled = false;

  const inputNode = $('nd-input');
  if (inputNode) inputNode.classList.add('on');

  const status = $('st-input');
  if (status) status.textContent = kind + (label ? ' · ' + label : '');
}

export function clearStartStop(running) {
  const cooling = appState.cooldownTimer !== null;
  const disabled = running || cooling || !hasSignal();

  if ($('btnStart')) $('btnStart').disabled = disabled;
  if ($('btnStop')) $('btnStop').disabled = !running;
  if ($('btnCam')) $('btnCam').disabled = running;
  if ($('btnFile')) $('btnFile').disabled = running;

  if ($('btnStart-dash')) $('btnStart-dash').disabled = disabled;
  if ($('btnStop-dash')) $('btnStop-dash').disabled = !running;
}

export function clearSourceUI() {
  if (hasSignal()) return;

  const emptyEl = $('empty');
  const emptyDashEl = $('empty-dash');
  if (emptyEl) emptyEl.style.display = '';
  if (emptyDashEl) emptyDashEl.style.display = '';

  const hudEl = $('hud');
  if (hudEl) hudEl.innerHTML = '';

  const inputNode = $('nd-input');
  if (inputNode) inputNode.classList.remove('on');

  if ($('btnStart')) $('btnStart').disabled = true;
  if ($('btnStart-dash')) $('btnStart-dash').disabled = true;

  appState.videoFile = null;
  setReportAvailability(false);

  const status = $('st-input');
  if (status) status.textContent = '未连接信号源';
}

export function setReportAvailability(on) {
  // 整段报告模式仅在选了视频文件时可用；摄像头实时流不支持 → 禁用并关掉开关。
  const tech = $('batchMode');
  if (tech) {
    tech.disabled = !on;
    if (!on && tech.checked) { tech.checked = false; tech.dispatchEvent(new Event('change')); }
  }
  const dash = $('batchMode-dash');
  if (dash) {
    dash.disabled = !on;
    if (!on) dash.checked = false;
  }
}

export function stopCamera() {
  const video = getVideoElement();
  if (appState.stream) {
    appState.stream.getTracks().forEach((track) => track.stop());
    appState.stream = null;
  }
  if (video) video.srcObject = null;
  appState.camOn = false;
  setCamBtn(false);
  clearBoxes();
}

export function stopAll(stopMonitor = () => {}) {
  stopMonitor();
  stopCamera();
}

export function bindDashboardVideoProxy() {
  if (!$('btnCam-dash')) return;

  $('btnCam-dash').onclick = () => $('btnCam')?.click();
  $('btnFile-dash').onclick = () => $('btnFile')?.click();
  $('fileInput-dash').onchange = (e) => {
    const mainInput = $('fileInput');
    if (!mainInput) return;
    mainInput.files = cloneFileList(e.target.files);
    mainInput.dispatchEvent(new Event('change'));
  };
}

export function initVideoController({ stopMonitor = () => {} } = {}) {
  const video = getVideoElement();
  if (!video) return;

  bindDashboardVideoProxy();

  if ($('btnCam')) {
    $('btnCam').onclick = async () => {
      if (appState.camOn) {
        stopCamera();
        clearSourceUI();
        return;
      }

      stopAll(stopMonitor);
      try {
        appState.stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720 },
          audio: false,
        });
        video.srcObject = appState.stream;
        video.src = '';
        await video.play();
        appState.camOn = true;
        appState.videoFile = null;
        setReportAvailability(false);
        setCamBtn(true);
        setSource('摄像头', '📷 摄像头');
      } catch (e) {
        appState.camOn = false;
        setCamBtn(false);
        alert('无法访问摄像头：' + e.message + '\n（需 https 或 localhost，且授权摄像头权限）');
      }
    };
  }

  if ($('btnFile')) {
    $('btnFile').onclick = () => $('fileInput')?.click();
  }

  if ($('fileInput')) {
    $('fileInput').onchange = (e) => {
      const file = e.target.files[0];
      if (!file) return;
      stopCamera();
      video.srcObject = null;
      video.src = URL.createObjectURL(file);
      video.loop = true;
      video.controls = true;
      video.muted = true;
      video.play().catch(() => {});
      appState.videoFile = file;
      setReportAvailability(true);
      setSource(file.name, '🎞️ 视频文件');
    };
  }

  if ($('btnRef')) {
    $('btnRef').onclick = () => $('refInput')?.click();
  }

  if ($('refInput')) {
    $('refInput').onchange = (e) => {
      const file = e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        appState.refDataUri = reader.result;
        if ($('refThumb')) {
          $('refThumb').src = appState.refDataUri;
          $('refThumb').style.display = 'block';
        }
        if ($('btnRefClear')) {
          $('btnRefClear').style.display = 'inline-block';
        }
      };
      reader.readAsDataURL(file);
    };
  }

  if ($('btnRefClear')) {
    $('btnRefClear').onclick = () => {
      appState.refDataUri = null;
      if ($('refThumb')) $('refThumb').style.display = 'none';
      if ($('btnRefClear')) $('btnRefClear').style.display = 'none';
      if ($('refInput')) $('refInput').value = '';
    };
  }
}

export function bindUnloadCleanup(stopMonitor = () => {}) {
  window.addEventListener('beforeunload', () => stopAll(stopMonitor));
}
