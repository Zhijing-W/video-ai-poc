/**
 * 通用工具函数
 * 提供从 monitor.html.backup-refactor 提取的 DOM、时间、转义与媒体辅助能力。
 */

export function $(id) {
  return document.getElementById(id);
}

export function intervalMs(valueId = 'intervalVal', unitId = 'intervalUnit') {
  const v = parseFloat($(valueId)?.value) || 3;
  const unit = parseFloat($(unitId)?.value);
  return Math.max(200, Math.round(v * unit));
}

export function auditEvery(id = 'auditEvery') {
  return Math.max(2, Math.min(60, parseInt($(id)?.value, 10) || 8));
}

export function formatClock(date = new Date()) {
  return date.toLocaleTimeString('zh-CN', { hour12: false });
}

export function fmtSpan(a, b) {
  try {
    const s = new Date(a);
    const en = new Date(b);
    const d = Math.max(0, Math.round((en - s) / 1000));
    return s.toLocaleString('zh-CN', { hour12: false }) + ' · 时长 ' + d + 's';
  } catch (e) {
    return a || '';
  }
}

export function escapeHtml(s) {
  return (s || '').replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
}

export function cloneFileList(files) {
  const dt = new DataTransfer();
  Array.from(files || []).forEach((file) => dt.items.add(file));
  return dt.files;
}

export function captureVideoFrame(videoElement, width = 640, quality = 0.7) {
  const canvas = document.createElement('canvas');
  const height = Math.round(videoElement.videoHeight / videoElement.videoWidth * width);
  canvas.width = width;
  canvas.height = height;
  canvas.getContext('2d').drawImage(videoElement, 0, 0, width, height);
  return { canvas, dataUri: canvas.toDataURL('image/jpeg', quality), width, height };
}

// ===== 前端智能抽帧（实时流模式①：把 Step 7「画面没变就跳过」思路搬到浏览器）=====
// 不依赖 ffmpeg：把画面缩成 32×32 灰度指纹，比较相邻帧的平均差异（0~1），
// 差异小说明画面基本没变 → 可跳过本帧、连后端 YOLO 都不调，进一步省算力。
const _sigCanvas = typeof document !== 'undefined' ? document.createElement('canvas') : null;
if (_sigCanvas) { _sigCanvas.width = 32; _sigCanvas.height = 32; }

export function frameSignature(source) {
  if (!_sigCanvas) return null;
  const ctx = _sigCanvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, 32, 32);
  const data = ctx.getImageData(0, 0, 32, 32).data;
  const sig = new Uint8Array(1024);
  for (let i = 0; i < 1024; i++) {
    sig[i] = (data[i * 4] * 0.299 + data[i * 4 + 1] * 0.587 + data[i * 4 + 2] * 0.114) | 0;
  }
  return sig;
}

export function signatureDiff(a, b) {
  if (!a || !b || a.length !== b.length) return 1;
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return sum / a.length / 255;
}

export function imgToBase64(img) {
  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.9).split(',')[1];
}

export function captureFrame(videoElement) {
  const canvas = document.createElement('canvas');
  canvas.width = videoElement.videoWidth;
  canvas.height = videoElement.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoElement, 0, 0);
  return canvas.toDataURL('image/jpeg', 0.92).split(',')[1];
}

export function uniqueId() {
  return Date.now().toString(36) + Math.random().toString(36).substr(2, 5);
}

export function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj));
}

export function debounce(fn, delay) {
  let timer = null;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), delay);
  };
}
