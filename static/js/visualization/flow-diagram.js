/**
 * 流程图动画模块
 * 提取自 monitor.html.backup-refactor：负责节点高亮、状态文案、小预览与模型选择提示。
 */

import { appState } from '../core/state.js';
import { $, formatClock, intervalMs } from '../core/utils.js';

export const STAGES = ['input', 'smart', 'yolo', 'gate', 'compile', 'cruise', 'llm', 'result'];

export function closeAllPops() {
  // 已改为节点内联控件，无浮层（保留空函数兼容旧调用）
}

export function wireModelSelect(id, pillId, setter, planned = false, logRow = () => {}) {
  const sel = $(id);
  const pill = $(pillId);
  if (!sel) return;

  if (pill && sel.selectedIndex >= 0) {
    pill.textContent = sel.options[sel.selectedIndex].text.split(' ·')[0];
  }

  sel.addEventListener('change', () => {
    const value = sel.value;
    const txt = sel.options[sel.selectedIndex].text;
    setter(value);
    if (pill) pill.textContent = txt.split(' ·')[0];

    const isPlanned = planned || /规划/.test(txt);
    if (isPlanned) {
      logRow('attention', '🧩 已切换为「' + txt + '」：当前后端尚未接入，实跑仍为默认方案；该选择会随请求透传，接入后即生效。', false, null);
    } else {
      logRow('normal', '🧩 模型方案 → ' + txt, false, null);
    }
  });
}

export function initFlowDiagram(logRow = () => {}) {
  wireModelSelect('yoloModel', 'pill-yolo', (value) => { appState.yoloModelSel = value; }, false, logRow);
  wireModelSelect('llmModel', 'pill-llm', (value) => { appState.llmModelSel = value; }, false, logRow);
}

export function lightPath(active) {
  STAGES.forEach((stage) => {
    const node = $('nd-' + stage);
    if (node) node.classList.toggle('active', active.includes(stage));
  });

  document.querySelectorAll('.conn-h,.conn-v').forEach((arrow) => {
    const from = arrow.dataset.from;
    const to = arrow.dataset.to;
    arrow.classList.toggle('flow', active.includes(from) && active.includes(to));
  });

  if (appState.stageClear) clearTimeout(appState.stageClear);
  appState.stageClear = setTimeout(dimPath, Math.max(900, intervalMs() + 400));
}

export function dimPath() {
  STAGES.forEach((stage) => $('nd-' + stage)?.classList.remove('active'));
  document.querySelectorAll('.conn-h.flow,.conn-v.flow').forEach((arrow) => arrow.classList.remove('flow'));
}

export function nstat(id, txt) {
  const el = $('st-' + id);
  if (el) el.textContent = txt;
  if (id === 'result') {
    const mini = $('resultMini');
    if (mini) mini.textContent = txt;
  }
}

export function countsSummary(yolo) {
  const counts = (yolo && yolo.counts) || {};
  const keys = Object.keys(counts);
  return keys.length ? keys.map((key) => key + '×' + counts[key]).join(' ') : '无目标';
}

export function updateMini(dataUri) {
  const img = $('ndImg');
  if (img) img.src = dataUri;
}

export function startClock() {
  const clock = $('clock');
  if (!clock) return;
  clock.textContent = formatClock();
  setInterval(() => {
    clock.textContent = formatClock();
  }, 1000);
}
