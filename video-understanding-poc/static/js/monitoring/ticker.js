/**
 * 定时采样器模块
 * 提取自 monitor.html.backup-refactor：负责开始/停止监控、冷静期、抓帧节拍与归档。
 */

import { appState, resetCompareRuntime, resetFrameStore, resetStats, startCycle } from '../core/state.js';
import { $, captureVideoFrame, frameSignature, intervalMs, signatureDiff } from '../core/utils.js';
import { saveMonitorSession, trackResetRequest } from '../core/api.js';
import { clearStartStop, hasSignal } from '../ui/video-controller.js';
import { analyzeTick, auditTick, cruiseTickFrame } from './analyzer.js';
import { fetchSummary, showSummary } from './batch-report.js';
import { shouldUseCruise } from './gate-handler.js';
import { logRow } from '../ui/render-engine.js';
import { clearBoxes } from '../visualization/yolo-boxes.js';
import { closeAllPops, dimPath, lightPath, nstat, updateMini } from '../visualization/flow-diagram.js';
import { resetStatsUI } from '../visualization/stats-display.js';

// mode① 智能抽帧：相邻帧 32×32 灰度指纹差异 >= 此阈值才算「画面有变化」。
const SCENE_DIFF_THRESHOLD = 0.04;

export function restartTimerIfRunning() {
  if (!appState.timer) return;
  clearInterval(appState.timer);
  appState.timer = setInterval(tick, intervalMs());
  logRow(
    'normal',
    '⚙ 采样间隔已更新为 ' + $('intervalVal')?.value
      + ($('intervalUnit')?.value === '1000' ? '秒' : $('intervalUnit')?.value === '60000' ? '分钟' : '毫秒'),
    false,
    null,
  );
}

export function startMonitor() {
  if (appState.cooldownTimer) return;

  // 整段视频分析：seek 驱动在视频时间轴上快速逐帧扫描（不等真实播放），扫完总结 + 归档。
  if ($('batchMode')?.checked) {
    if (!appState.videoFile) {
      alert('整段视频分析需要先用「🎞️ 视频」选择一个视频文件（摄像头不支持）。');
      return;
    }
    runFullVideoAnalysis();
    return;
  }

  if (!hasSignal()) return;
  beginMonitoring();
}

// 重置统计/日志/UI 并起一个新周期（实时监控与整段分析共用）。
function prepareRun() {
  clearStartStop(true);
  $('led')?.classList.add('on');
  if ($('liveText')) $('liveText').textContent = appState.fullVideoRun ? '● 整段分析中' : '● 监控中';

  resetStats();
  resetStatsUI();
  resetCompareRuntime();
  resetFrameStore();
  appState.lastSig = null;
  appState.lastSentTs = 0;
  appState.cycleSummary = null;

  // Phase 3 · Step 12：为本轮建一个新的 track 门控会话，并清空后端跟踪/结论缓存，
  // 保证 track_id 从头计数、不串上一轮的轨迹与复用结论。
  appState.trackSessionId = 'mon-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 8);
  trackResetRequest(appState.trackSessionId).catch(() => {});

  const techLog = $('log');
  if (techLog) techLog.innerHTML = '';
  const dashLogContainer = $('dash-log-container');
  if (dashLogContainer) dashLogContainer.innerHTML = '';

  nstat('llm', '0 次调用');
  nstat('result', appState.fullVideoRun ? '整段分析中' : '监控中');
  $('nd-result')?.classList.remove('hit', 'warn');
  closeAllPops();
  startCycle();
}

function beginMonitoring() {
  const video = $('video');
  if (video) video.loop = true;       // 实时模式：视频文件循环播
  prepareRun();
  tick();
  appState.timer = setInterval(tick, intervalMs());
}

// 整段视频分析：用 seek 在视频时间轴上快速跳取帧逐帧分析。
// 速度只取决于「每帧分析耗时 × 帧数」，与视频实际时长无关（10 分钟视频也不用等 10 分钟）。
async function runFullVideoAnalysis() {
  const video = $('video');
  appState.fullVideoRun = true;
  if (video) {
    video.loop = false;
    try { video.pause(); } catch (e) { /* ignore */ }
  }

  prepareRun();

  // 确保拿到时长（metadata 可能还没就绪）
  if (video && (!video.duration || isNaN(video.duration) || !isFinite(video.duration))) {
    await new Promise((res) => {
      const h = () => { video.removeEventListener('loadedmetadata', h); res(); };
      video.addEventListener('loadedmetadata', h);
      setTimeout(res, 3000);
    });
  }

  const dur = (video && video.duration && isFinite(video.duration)) ? video.duration : 0;
  const step = Math.max(0.2, intervalMs() / 1000);   // 每隔 step 秒「视频时间」取一帧
  logRow(
    'normal',
    '🎬 整段视频分析开始：快速逐帧扫描（约 ' + Math.ceil(dur) + 's 视频，每 ' + step + 's 取一帧），不等播放',
    false, null,
  );

  for (let t = 0; t < dur && appState.fullVideoRun; t += step) {
    await seekTo(video, t);
    if (!appState.fullVideoRun) break;
    await tick();                 // 复用实时逐帧逻辑：智能抽帧 → YOLO → 门控 → 命中才 gpt-4o
  }

  await finishFullVideoAnalysis();
}

// 把视频精确定位到某个时间点，等 seeked 后 resolve（带兜底，避免个别帧不触发 seeked）。
function seekTo(video, t) {
  return new Promise((resolve) => {
    if (!video) return resolve();
    let settled = false;
    const done = () => {
      if (settled) return;
      settled = true;
      video.removeEventListener('seeked', done);
      resolve();
    };
    video.addEventListener('seeked', done);
    try {
      video.currentTime = Math.min(t, Math.max(0, (video.duration || t) - 0.05));
    } catch (e) {
      done();
      return;
    }
    setTimeout(done, 2000);   // 兜底
  });
}

async function finishFullVideoAnalysis() {
  appState.fullVideoRun = false;
  if (appState.timer) { clearInterval(appState.timer); appState.timer = null; }

  const entries = (appState.cycle && appState.cycle.entries) || [];
  logRow('normal', '🎬 逐帧扫描完成，正在生成本次总结…', false, null);
  nstat('result', '生成总结中…');
  try {
    const summary = await fetchSummary(entries);
    appState.cycleSummary = summary;
    showSummary(summary);
  } catch (e) {
    logRow('alert', '总结失败：' + (e.message || ''), false, null);
  }

  await saveCycle();

  $('led')?.classList.remove('on');
  if ($('liveText')) $('liveText').textContent = '已停止';
  $('stage')?.classList.remove('analyzing');
  $('stage-dash')?.classList.remove('analyzing');
  clearBoxes();
  dimPath();
  clearStartStop(false);
}

export function stopMonitor() {
  // 整段分析：置标志让扫描循环收尾（循环退出后会自动总结 + 归档）。
  if (appState.fullVideoRun) {
    appState.fullVideoRun = false;
    return;
  }

  const wasRunning = appState.timer !== null;
  if (appState.timer) {
    clearInterval(appState.timer);
    appState.timer = null;
  }

  $('stage')?.classList.remove('analyzing');
  $('stage-dash')?.classList.remove('analyzing');
  clearBoxes();

  $('led')?.classList.remove('on');
  if ($('liveText')) $('liveText').textContent = '已停止';
  dimPath();
  nstat('yolo', '待机');
  nstat('gate', '—');
  nstat('cruise', appState.activeCompare ? '已暂停' : '未启用');

  if (wasRunning) startCooldown();
  else {
    nstat('result', '已停止');
    clearStartStop(false);
  }
}

export function startCooldown() {
  const SEC = 5;
  let left = SEC;

  const tickDown = () => {
    if ($('liveText')) $('liveText').textContent = '● 冷静期 ' + left + 's';
    nstat('result', '冷静期 ' + left + 's · 收尾中（延迟帧仍在收录）');
  };

  tickDown();
  $('led')?.classList.add('cooldown');
  clearStartStop(true);
  if ($('btnStop')) $('btnStop').disabled = true;
  logRow('normal', '监控已停止，进入 ' + SEC + 's 冷静期：继续收录延迟到达的帧，期满后归档（这 ' + SEC + 's 内不可开始新监控）', false, null);

  appState.cooldownTimer = setInterval(() => {
    left--;
    if (left > 0) {
      tickDown();
      return;
    }

    clearInterval(appState.cooldownTimer);
    appState.cooldownTimer = null;
    $('led')?.classList.remove('cooldown');
    if ($('liveText')) $('liveText').textContent = '已停止';
    nstat('result', '已停止');
    saveCycle();
    clearStartStop(false);
  }, 1000);
}

export async function saveCycle() {
  if (!appState.cycle || !appState.cycle.entries.length) {
    appState.cycle = null;
    return;
  }

  const payload = {
    started_at: appState.cycle.startedAt,
    ended_at: new Date().toISOString(),
    target: appState.activeCompare ? (appState.activeCompare.target || '（参考图）') : null,
    mode: appState.activeCompare ? appState.activeCompare.mode : null,
    summary: appState.cycleSummary || null,
    stats: {
      frames: appState.stats.frames,
      match: appState.stats.match,
      alert: appState.stats.alert,
    },
    entries: appState.cycle.entries.slice(-300),
  };
  appState.cycle = null;
  appState.cycleSummary = null;

  try {
    const result = await saveMonitorSession(payload);
    if (result.ok) {
      const data = result.data;
      logRow('normal', '已归档监控周期 ' + data.id + '（' + data.frames + ' 帧），可在『历史日志』查看', false, null);
    }
  } catch (e) {
    // 离线保存失败忽略，保持原逻辑。
  }
}

export async function tick() {
  const video = $('video');
  const stage = $('stage');
  if (appState.inFlight || !video || !video.videoWidth) return;

  // mode① 实时流智能抽帧：画面没变且未到兜底间隔 → 跳过本帧（连后端 YOLO 都不调，更省）。
  if ($('smartFrame')?.checked) {
    const sig = frameSignature(video);
    const since = Date.now() - (appState.lastSentTs || 0);
    const fallbackMs = Math.max(1000, (parseFloat($('fallbackSec')?.value) || 30) * 1000);
    const changed = !appState.lastSig || signatureDiff(appState.lastSig, sig) >= SCENE_DIFF_THRESHOLD;
    appState.lastSig = sig;
    if (!changed && since < fallbackMs) {
      lightPath(['input', 'smart']);
      nstat('smart', '画面静止 · 跳过抽帧 ⏭');
      return;
    }
    appState.lastSentTs = Date.now();
    nstat('smart', changed ? '画面有变化 · 抽帧 ✓' : '兜底抽帧 ✓');
  } else {
    nstat('smart', '智能抽帧关 · 定时抽帧');
  }

  const { dataUri } = captureVideoFrame(video, 640, 0.7);
  updateMini(dataUri);

  const plan = appState.activeCompare && appState.activeCompare.plan;
  const cruiseOn = shouldUseCruise(plan);

  appState.inFlight = true;
  stage?.classList.add('analyzing');
  $('stage-dash')?.classList.add('analyzing');

  try {
    if (cruiseOn) {
      appState.cruiseCount++;
      const tcls = plan.yolo_class;
      const targetSeen = !!(appState.lastCounts && tcls && appState.lastCounts[tcls] > 0);
      const sinceAudit = appState.cruiseCount - appState.lastAuditCount;
      const auditEvery = Math.max(2, Math.min(60, parseInt($('auditEvery')?.value, 10) || 8));
      const doAudit = targetSeen
        ? (appState.cruiseFlip || sinceAudit >= auditEvery)
        : (sinceAudit >= auditEvery * 4);

      if (doAudit) {
        appState.lastAuditCount = appState.cruiseCount;
        appState.cruiseFlip = false;
        await auditTick(dataUri, plan);
      } else {
        await cruiseTickFrame(dataUri, plan);
      }
    } else {
      await analyzeTick(dataUri, plan);
    }
  } catch (e) {
    logRow('alert', '网络错误：' + e.message, false, null);
  } finally {
    appState.inFlight = false;
    stage?.classList.remove('analyzing');
    $('stage-dash')?.classList.remove('analyzing');
  }
}

export function initTicker() {
  if ($('btnStart')) $('btnStart').onclick = startMonitor;
  if ($('btnStop')) $('btnStop').onclick = stopMonitor;
  if ($('boxToggle')) $('boxToggle').onchange = () => { if (!$('boxToggle').checked) clearBoxes(); };
  if ($('intervalVal')) $('intervalVal').oninput = restartTimerIfRunning;
  if ($('intervalUnit')) $('intervalUnit').onchange = restartTimerIfRunning;
  window.addEventListener('resize', clearBoxes);
}
