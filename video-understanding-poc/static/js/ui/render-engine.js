/**
 * 统一渲染引擎
 * 提取自 monitor.html.backup-refactor：负责分析结果渲染、日志、帧预览与历史日志视图。
 */

import { appState, appendCycleEntry, nextFrameId } from '../core/state.js';
import { $, escapeHtml, fmtSpan, formatClock } from '../core/utils.js';
import { deleteMonitorSession, getMonitorSession, listMonitorSessions } from '../core/api.js';
import { drawBoxes } from '../visualization/yolo-boxes.js';
import { countsSummary, nstat } from '../visualization/flow-diagram.js';
import {
  setActiveMatchCard,
  setCompareStoppedCard,
  setCruiseMatchCard,
  setIdleMatchCard,
  setResultChip,
  syncAlertStats,
  syncCorner,
  syncDashMatchMirror,
  syncFrameStats,
  syncHudFromMain,
  syncLlmSavedStats,
  syncMatchStats,
  syncSceneAndTags,
  syncSubjects,
  syncIdentity,
} from '../visualization/stats-display.js';

function storeFrame(dataUri, data, ms) {
  const fid = nextFrameId();
  appState.frameStore[fid] = {
    dataUri,
    res: data,
    ts: formatClock(),
    ms,
  };
  return fid;
}

export function render(data, ms, dataUri) {
  const res = data.result || {};
  const gated = data.gated;
  const stats = appState.stats;

  stats.frames++;
  syncFrameStats(stats);

  if (gated) stats.llm++;
  else stats.skipped++;
  syncLlmSavedStats(stats);

  const tag = gated ? (ms + 'ms · gpt-4o') : (ms + 'ms · YOLO');
  syncCorner(tag);
  setResultChip(gated ? (res.alert_level || 'normal').toUpperCase() : 'YOLO');
  syncHudFromMain();

  nstat('yolo', countsSummary(data.yolo));
  nstat('llm', stats.llm + ' 次调用' + (data.reused ? ' · 复用 ♻' : ''));

  drawBoxes(data.yolo, {
    plan: appState.activeCompare && appState.activeCompare.plan,
    matchedBoxes: data.cruise_match && data.cruise_match.matched_boxes,
  });

  // Phase 3 · "连"：刷新主体记忆读数（track 门控开启时 data.identity 才有值，否则隐藏）。
  syncIdentity(data.identity);

  const sceneText = res.scene || '—';
  syncSceneAndTags(sceneText, res.detected_objects || []);
  syncSubjects(res.subjects || []);

  const match = res.match || {};
  const alert = res.alert_level === 'alert';
  if (!appState.activeCompare) {
    setIdleMatchCard(alert, sceneText);
  } else {
    const hit = setActiveMatchCard(match, alert);
    if (hit) {
      stats.match++;
      syncMatchStats(stats);
    }
  }
  syncDashMatchMirror(appState.activeCompare);

  if (alert) {
    stats.alert++;
    syncAlertStats(stats);
  }

  nstat('result', $('matchBig')?.textContent || '—');
  $('nd-result')?.classList.toggle('hit', match.is_match === true);
  $('nd-result')?.classList.toggle('warn', alert);

  const fid = storeFrame(dataUri, data, ms);
  const level = res.alert_level || 'normal';
  let msg = res.notification || res.scene || '已分析';
  if (!gated) msg = '⏭ ' + msg;
  if (appState.activeCompare && match.is_match === true) {
    msg = '🎯 命中「' + (match.target || '参考图') + '」· ' + msg;
  }

  appendCycleEntry({
    seq: fid,
    ts: appState.frameStore[fid].ts,
    level,
    msg,
    is_match: appState.activeCompare && match.is_match === true,
    image: dataUri,
    result: data,
  });

  logRow(level, msg, appState.activeCompare && match.is_match === true, fid, dataUri);
}

export function renderCruise(data, ms, dataUri, plan) {
  const yolo = data.yolo || {};
  const cruise = data.cruise || {};
  const stats = appState.stats;

  stats.frames++;
  syncFrameStats(stats);

  stats.skipped++;
  syncLlmSavedStats(stats);

  syncCorner(ms + 'ms · YOLO巡航');
  setResultChip('巡航');

  nstat('yolo', countsSummary(yolo));
  nstat('cruise', cruise.is_match === true ? '命中（巡航）' : (cruise.reason ? '巡航中' : '巡航中'));
  nstat('result', cruise.is_match === true ? '✓ 命中（巡航）' : '巡航中');

  $('nd-result')?.classList.toggle('hit', cruise.is_match === true);
  $('nd-result')?.classList.remove('warn');

  drawBoxes(yolo, { plan, matchedBoxes: cruise.matched_boxes });

  const counts = yolo.counts || {};
  const labels = Object.keys(counts);
  const sceneText = cruise.reason || ('巡航中：' + labels.map((key) => key + '×' + counts[key]).join('、'));
  syncSceneAndTags(sceneText, labels);
  if ($('subjects')) $('subjects').innerHTML = '';

  const hit = setCruiseMatchCard(cruise, appState.activeCompare);
  if (hit) {
    stats.match++;
    syncMatchStats(stats);
  }
  syncDashMatchMirror(appState.activeCompare);

  const fid = storeFrame(dataUri, data, ms);
  let msg = '⏭ 巡航：' + (cruise.reason || '无目标');
  if (cruise.is_match === true) {
    msg = '🎯 巡航命中「' + (appState.activeCompare?.target || '') + '」· ' + (cruise.reason || '');
  }

  appendCycleEntry({
    seq: fid,
    ts: appState.frameStore[fid].ts,
    level: 'normal',
    msg,
    is_match: cruise.is_match === true,
    image: dataUri,
    result: data,
  });

  logRow('normal', msg, cruise.is_match === true, fid, dataUri);
  return fid;
}

export function logRow(level, msg, isMatch, fid, dataUri) {
  const t = formatClock();
  const thumb = dataUri ? `<img class="thumb" src="${dataUri}" alt="frame"/>` : `<span class="dot"></span>`;

  const row = document.createElement('div');
  row.className = 'logrow ' + level;
  row.innerHTML = `${thumb}<span class="t">${t}</span>
    <span class="msg">${isMatch ? '<span class="mtag">命中 </span>' : ''}${escapeHtml(msg)}</span>
    ${fid ? '<span class="arrow">查看帧 ›</span>' : ''}`;
  if (fid) row.onclick = () => openFrame(fid);

  const log = $('log');
  if (log) {
    log.prepend(row);
    while (log.children.length > 60) log.removeChild(log.lastChild);
  }

  const dashLogContainer = $('dash-log-container');
  if (dashLogContainer) {
    const emptyEl = dashLogContainer.querySelector('.dash-log-empty');
    if (emptyEl) emptyEl.remove();

    const dashRow = document.createElement('div');
    dashRow.className = 'dash-log-item ' + level;
    const thumbHtml = dataUri
      ? `<img class="thumb" src="${dataUri}" alt="frame" style="width:42px;height:30px;object-fit:cover;border-radius:4px;border:1px solid var(--line);margin-right:4px;"/>`
      : '';

    dashRow.innerHTML = `<span class="dot"></span>
      <div class="content">
        <div class="time">${t}</div>
        <div class="msg" style="display:flex;align-items:center;gap:6px;">
          ${thumbHtml}
          <span>${isMatch ? '<span style="color:var(--ok);font-weight:700;font-size:.75rem">✓ 命中</span> ' : ''}${escapeHtml(msg)}</span>
        </div>
      </div>`;
    if (fid) {
      dashRow.style.cursor = 'pointer';
      dashRow.onclick = () => openFrame(fid);
    }
    dashLogContainer.prepend(dashRow);
    while (dashLogContainer.children.length > 60) {
      dashLogContainer.removeChild(dashLogContainer.lastChild);
    }
  }
}

export function logClickable(level, msg, onClick) {
  const t = formatClock();

  const log = $('log');
  if (log) {
    const row = document.createElement('div');
    row.className = 'logrow ' + level;
    row.style.cursor = 'pointer';
    row.innerHTML = `<span class="dot"></span><span class="t">${t}</span>`
      + `<span class="msg">${escapeHtml(msg)}</span><span class="arrow">查看 ›</span>`;
    row.onclick = onClick;
    log.prepend(row);
    while (log.children.length > 60) log.removeChild(log.lastChild);
  }

  const dash = $('dash-log-container');
  if (dash) {
    const emptyEl = dash.querySelector('.dash-log-empty');
    if (emptyEl) emptyEl.remove();
    const row = document.createElement('div');
    row.className = 'dash-log-item ' + level;
    row.style.cursor = 'pointer';
    row.innerHTML = `<span class="dot"></span><div class="content"><div class="time">${t}</div>`
      + `<div class="msg"><span>${escapeHtml(msg)}</span></div></div>`;
    row.onclick = onClick;
    dash.prepend(row);
    while (dash.children.length > 60) dash.removeChild(dash.lastChild);
  }
}

export function openFrame(fid) {
  const frame = appState.frameStore[fid];
  if (!frame) return;

  const gated = frame.res && frame.res.gated;
  const isCruise = frame.res && frame.res.cruise !== undefined && frame.res.gated === undefined;
  if ($('modalImg')) $('modalImg').src = frame.dataUri;
  if ($('modalTime')) {
    $('modalTime').textContent = '采集时间 ' + frame.ts + ' · 耗时 ' + frame.ms + 'ms · '
      + (gated ? '已调用 gpt-4o 分析' : isCruise ? 'YOLO 自动巡航（未花 LLM）' : '门控跳过 · 仅 YOLO（未花 LLM）');
  }
  if ($('modalJson')) $('modalJson').textContent = JSON.stringify(frame.res, null, 2);
  $('modal')?.classList.add('open');
}

export function openFrameUrl(url, res, ts) {
  if (!url) return;
  if ($('modalImg')) $('modalImg').src = url;
  if ($('modalTime')) {
    $('modalTime').textContent = '采集时间 ' + (ts || '') + ' · 历史归档帧 · 发送给 gpt-4o 的画面';
  }
  if ($('modalJson')) $('modalJson').textContent = JSON.stringify(res || {}, null, 2);
  $('modal')?.classList.add('open');
}

export async function openHistory() {
  if ($('histDetail')) $('histDetail').style.display = 'none';
  if ($('histList')) {
    $('histList').style.display = 'block';
    $('histList').innerHTML = '<div class="hempty">加载中…</div>';
  }
  $('histModal')?.classList.add('open');

  try {
    const payload = await listMonitorSessions();
    const items = payload.sessions || [];
    if (!items.length) {
      if ($('histList')) {
        $('histList').innerHTML = '<div class="hempty">暂无历史周期。完整地「开始监控 → 停止监控」一次后会自动归档。</div>';
      }
      return;
    }

    if ($('histList')) {
      $('histList').innerHTML = items.map((session) => `
        <div class="hsess" data-id="${session.id}">
          <span class="hid">${session.id}</span>
          <span class="hmeta">${fmtSpan(session.started_at, session.ended_at)}${session.target ? ' · 目标:' + escapeHtml(session.target) : ' · 仅理解'}</span>
          <span class="hnum"><b>${session.frames || 0}</b> 帧 · 命中 <b>${session.match || 0}</b> · <span class="ha">告警 ${session.alert || 0}</span></span>
          <button class="hdel" data-del="${session.id}">删</button>
        </div>`).join('');

      $('histList').querySelectorAll('.hsess').forEach((el) => {
        el.onclick = (e) => {
          if (e.target.dataset.del) return;
          loadSession(el.dataset.id);
        };
      });
      $('histList').querySelectorAll('[data-del]').forEach((button) => {
        button.onclick = async (e) => {
          e.stopPropagation();
          if (!confirm('删除该监控周期？')) return;
          await deleteMonitorSession(button.dataset.del);
          openHistory();
        };
      });
    }
  } catch (e) {
    if ($('histList')) $('histList').innerHTML = '<div class="hempty">加载失败：' + escapeHtml(e.message) + '</div>';
  }
}

export async function loadSession(id) {
  try {
    const session = await getMonitorSession(id);
    if ($('histList')) $('histList').style.display = 'none';
    if ($('histDetail')) $('histDetail').style.display = 'block';

    const stats = session.stats || {};
    if ($('histDetailMeta')) {
      $('histDetailMeta').textContent = '周期 ' + id + ' · ' + fmtSpan(session.started_at, session.ended_at)
        + ' · ' + (stats.frames || session.entries.length) + ' 帧 · 命中 ' + (stats.match || 0) + ' · 告警 ' + (stats.alert || 0)
        + (session.target ? ' · 目标:' + session.target : ' · 仅理解');
    }

    const log = $('histLog');
    if (!log) return;
    log.innerHTML = '';
    session.entries.slice().reverse().forEach((entry) => {
      const row = document.createElement('div');
      row.className = 'logrow ' + (entry.level || 'normal');
      const thumb = entry.frame_url ? `<img class="thumb" src="${entry.frame_url}" alt="f"/>` : `<span class="dot"></span>`;
      row.innerHTML = `${thumb}<span class="t">${entry.ts || ''}</span>
        <span class="msg">${entry.is_match ? '<span class="mtag">命中 </span>' : ''}${escapeHtml(entry.msg || '')}</span>
        <span class="arrow">查看帧 ›</span>`;
      row.onclick = () => openFrameUrl(entry.frame_url, entry.result, entry.ts);
      log.appendChild(row);
    });
  } catch (e) {
    alert('加载失败：' + e.message);
  }
}

export function initRenderEngine() {
  if ($('modalClose')) $('modalClose').onclick = () => $('modal')?.classList.remove('open');
  if ($('modal')) {
    $('modal').onclick = (e) => {
      if (e.target === $('modal')) $('modal').classList.remove('open');
    };
  }

  if ($('btnHistory')) $('btnHistory').onclick = openHistory;
  if ($('histClose')) $('histClose').onclick = () => $('histModal')?.classList.remove('open');
  if ($('histModal')) {
    $('histModal').onclick = (e) => {
      if (e.target === $('histModal')) $('histModal').classList.remove('open');
    };
  }
  if ($('histBack')) {
    $('histBack').onclick = () => {
      if ($('histDetail')) $('histDetail').style.display = 'none';
      if ($('histList')) $('histList').style.display = 'block';
    };
  }
}

export { setCompareStoppedCard };
