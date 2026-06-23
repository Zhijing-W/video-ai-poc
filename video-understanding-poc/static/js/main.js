/**
 * 主入口文件
 * 初始化 monitor.html 的模块化业务逻辑。
 */

import { escapeHtml } from './core/utils.js';
import { initVideoController, stopAll as stopMedia } from './ui/video-controller.js';
import { initModeSwitcher, switchMode } from './ui/mode-switcher.js';
import { initFlowDiagram, startClock } from './visualization/flow-diagram.js';
import { initGateHandler } from './monitoring/gate-handler.js';
import { initTicker, restartTimerIfRunning, startMonitor, stopMonitor } from './monitoring/ticker.js';
import { initBatchReport } from './monitoring/batch-report.js';
import { logRow, openFrame, openFrameUrl } from './ui/render-engine.js';

function stopAll() {
  stopMedia(stopMonitor);
}

window.switchMode = switchMode;
window.startMonitoring = startMonitor;
window.stopMonitoring = stopMonitor;
window.restartTimerIfRunning = restartTimerIfRunning;
window.stopAll = stopAll;
window.escapeHtml = escapeHtml;
window.openFrame = openFrame;
window.openFrameUrl = openFrameUrl;

document.addEventListener('DOMContentLoaded', () => {
  console.log('🚀 Video Understanding PoC - 初始化中...');

  initModeSwitcher();
  initFlowDiagram(logRow);
  initGateHandler();
  initTicker();
  initVideoController({ stopMonitor });
  initBatchReport();
  startClock();

  console.log('✅ 初始化完成');
});
