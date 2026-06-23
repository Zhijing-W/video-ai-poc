/**
 * YOLO 检测框模块
 * 提取自 monitor.html.backup-refactor：负责根据原始检测框与巡航命中信息绘制叠加框层。
 */

import { $, escapeHtml } from '../core/utils.js';

const COLOR_HEX = {
  red: '#ef4444',
  orange: '#f97316',
  yellow: '#eab308',
  green: '#22c55e',
  blue: '#3b82f6',
  purple: '#a855f7',
  pink: '#ec4899',
  white: '#e5e7eb',
  gray: '#9ca3af',
  black: '#4b5563',
  brown: '#a16207',
};

export function colorHex(color) {
  return color ? (COLOR_HEX[color] || null) : null;
}

export function videoRect(videoEl = $('video')) {
  const vw = videoEl?.videoWidth;
  const vh = videoEl?.videoHeight;
  if (!vw || !vh) return null;

  const container = videoEl.parentElement;
  if (!container) return null;

  const Sw = container.clientWidth;
  const Sh = container.clientHeight;
  const scale = Math.min(Sw / vw, Sh / vh);
  const w = vw * scale;
  const h = vh * scale;
  return { x: (Sw - w) / 2, y: (Sh - h) / 2, w, h, Sw, Sh };
}

export function sameBox(a, b) {
  return a && b && a.length === 4 && b.length === 4 && a.every((v, i) => Math.abs(v - b[i]) < 0.5);
}

export function clearBoxes() {
  if ($('boxes')) $('boxes').innerHTML = '';
  if ($('boxes-dash')) $('boxes-dash').innerHTML = '';
}

export function drawBoxes(yolo, opts = {}) {
  const svg = $('boxes');
  const svgDash = $('boxes-dash');
  if (!svg) return;

  if (!$('boxToggle')?.checked || !yolo || !yolo.detections || !yolo.img_w || !yolo.img_h) {
    svg.innerHTML = '';
    if (svgDash) svgDash.innerHTML = '';
    return;
  }

  const rect = videoRect();
  if (!rect) {
    svg.innerHTML = '';
    if (svgDash) svgDash.innerHTML = '';
    return;
  }

  const viewBox = `0 0 ${rect.Sw} ${rect.Sh}`;
  svg.setAttribute('viewBox', viewBox);
  if (svgDash) svgDash.setAttribute('viewBox', viewBox);

  const sx = rect.w / yolo.img_w;
  const sy = rect.h / yolo.img_h;
  const plan = opts.plan;
  const matched = opts.matchedBoxes || [];
  const parts = [];

  for (const detection of yolo.detections) {
    const box = detection.box;
    if (!box || box.length < 4) continue;

    const x = rect.x + box[0] * sx;
    const y = rect.y + box[1] * sy;
    const w = (box[2] - box[0]) * sx;
    const h = (box[3] - box[1]) * sy;
    const isTarget = plan && plan.yolo_class && detection.label === plan.yolo_class;
    const isMatched = matched.some((matchedBox) => sameBox(matchedBox, box));

    let stroke = colorHex(detection.color) || '#38bdf8';
    let sw = 2;
    let dash = '';
    if (isMatched) {
      stroke = '#22c55e';
      sw = 3;
    } else if (isTarget) {
      stroke = '#f59e0b';
      sw = 2.5;
      dash = '6 4';
    }

    let label = `${detection.label} ${Math.round((detection.confidence || 0) * 100)}%`;
    if (detection.color_zh) label += ' ·' + detection.color_zh;
    // Phase 3 · "连"：认出主体则在框上标 #主体号，回头客（跨 track 复用）加 ♻。
    if (detection.subject_id != null) {
      label += ' ·#' + detection.subject_id + (detection.subject_reused ? '♻' : '');
    }
    const tw = label.length * 7.4 + 10;
    const ty = y > 16 ? y - 16 : y + 1;

    parts.push(
      `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(0, w).toFixed(1)}" height="${Math.max(0, h).toFixed(1)}" `
      + `fill="none" stroke="${stroke}" stroke-width="${sw}"${dash ? ` stroke-dasharray="${dash}"` : ''} rx="3"/>`
      + `<rect x="${x.toFixed(1)}" y="${ty.toFixed(1)}" width="${tw.toFixed(1)}" height="15" fill="${stroke}" opacity="0.9" rx="2"/>`
      + `<text x="${(x + 4).toFixed(1)}" y="${(ty + 11).toFixed(1)}" font-size="11" font-family="monospace" fill="#0b0f14">${escapeHtml(label)}</text>`,
    );
  }

  const html = parts.join('');
  svg.innerHTML = html;
  if (svgDash) svgDash.innerHTML = html;
}
