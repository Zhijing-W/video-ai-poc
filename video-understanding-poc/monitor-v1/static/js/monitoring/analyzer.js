/**
 * 分析编排模块
 * 提取自 monitor.html.backup-refactor：负责普通分析、YOLO 巡航、审计与回填流程。
 */

import { BACKFILL_MAX, appState } from '../core/state.js';
import { analyzeFrameRequest, cruiseFrameRequest } from '../core/api.js';
import { buildAnalyzePayload } from './gate-handler.js';
import { logRow, render, renderCruise } from '../ui/render-engine.js';
import { lightPath, nstat } from '../visualization/flow-diagram.js';
import { syncLlmSavedStats, syncMatchStats } from '../visualization/stats-display.js';

export async function analyzeTick(dataUri, plan) {
  const payload = buildAnalyzePayload(dataUri, plan);
  const t0 = performance.now();
  const result = await analyzeFrameRequest(payload);
  const ms = Math.round(performance.now() - t0);

  if (!result.ok) {
    logRow('alert', '分析失败：' + (result.errorText || '').slice(0, 120), false, null);
    return;
  }

  const data = result.data;
  appState.lastCounts = (data.yolo && data.yolo.counts) || {};

  // Phase 3 · Step 12：track 门控路径 —— 后端权威决定调 LLM / 复用结论。
  // render() 会据 data.gated / data.reused 自动计入「省下次数」并显示「复用 ♻」。
  if (data.track_gate) {
    const usedLlm = data.gated && !data.reused;
    const route = ['input', 'smart', 'yolo', 'gate'];
    if (usedLlm) route.push('llm');
    route.push('result');
    lightPath(route);
    nstat('gate', usedLlm ? '→ gpt-4o（新主体）' : (data.reused ? '→ 复用 ♻（轨迹未变）' : '跳过 ⏭'));
    if (usedLlm) {
      appState.lastLlmTs = Date.now();
      appState.lastLlmData = data;
    }
    render(data, ms, dataUri);
    return;
  }

  const usedLlm = data.gated && !data.reused;
  const route = ['input', 'smart', 'yolo', 'gate'];
  if (appState.activeCompare && plan) route.push('compile');
  if (usedLlm) route.push('llm');
  route.push('result');
  lightPath(route);

  nstat('gate', data.gated ? (data.reused ? '→ 复用 ♻' : '→ gpt-4o') : '跳过 ⏭');

  if (data.gated && !data.reused) {
    appState.lastLlmTs = Date.now();
    appState.lastLlmSig = data.signature || null;
    appState.lastLlmData = data;
    render(data, ms, dataUri);
  } else if (data.signature && data.signature === appState.lastLlmSig && appState.lastLlmData) {
    const shown = Object.assign({}, appState.lastLlmData, {
      yolo: data.yolo,
      gated: false,
      reused: true,
      cruise_match: null,
    });
    shown.result = Object.assign({}, appState.lastLlmData.result || {}, {
      notification: data.reused
        ? '♻ 复用上次 gpt-4o 结论（签名未变，省一次调用）'
        : '⏭ ' + (((appState.lastLlmData.result || {}).scene) || '沿用上次结论'),
    });
    render(shown, ms, dataUri);
  } else {
    render(data, ms, dataUri);
  }
}

export async function cruiseTickFrame(dataUri, plan) {
  const t0 = performance.now();
  const result = await cruiseFrameRequest({
    image: dataUri,
    plan,
    yolo_model: appState.yoloModelSel,
  });
  const ms = Math.round(performance.now() - t0);

  if (!result.ok) {
    logRow('alert', '巡航失败：' + (result.errorText || '').slice(0, 120), false, null);
    return;
  }

  const data = result.data;
  appState.lastCounts = (data.yolo && data.yolo.counts) || {};
  lightPath(['input', 'smart', 'yolo', 'compile', 'cruise', 'result']);
  nstat('gate', '巡航旁路 ⏭');

  const verdict = !!(data.cruise && data.cruise.is_match);
  if (appState.lastCruiseVerdict !== null && verdict !== appState.lastCruiseVerdict) {
    appState.cruiseFlip = true;
  }
  appState.lastCruiseVerdict = verdict;

  const fid = renderCruise(data, ms, dataUri, plan);
  appState.cruiseBuf.push({ fid, dataUri, verdict });
}

export async function auditTick(dataUri, plan) {
  const payload = {
    image: dataUri,
    target: appState.activeCompare.target,
    reference_image: appState.activeCompare.reference,
    gate_enabled: false,
    comparing: true,
    plan,
    yolo_model: appState.yoloModelSel,
    llm_model: appState.llmModelSel,
  };

  const t0 = performance.now();
  const result = await analyzeFrameRequest(payload);
  const ms = Math.round(performance.now() - t0);

  if (!result.ok) {
    logRow('alert', '审计失败：' + (result.errorText || '').slice(0, 120), false, null);
    return;
  }

  const data = result.data;
  appState.lastCounts = (data.yolo && data.yolo.counts) || {};
  lightPath(['input', 'smart', 'yolo', 'compile', 'cruise', 'llm', 'result']);
  nstat('gate', '审计帧 · 强制 gpt-4o');

  appState.lastLlmTs = Date.now();
  appState.lastLlmSig = data.signature || null;
  appState.lastLlmData = data;

  if (data.cruise_match) {
    appState.lastCruiseVerdict = !!data.cruise_match.is_match;
    appState.cruiseFlip = false;
  }

  render(data, ms, dataUri);

  const llmMatch = (data.result && data.result.match) ? data.result.match.is_match : null;
  const yoloMatch = data.cruise_match ? data.cruise_match.is_match : null;
  if (llmMatch !== null && yoloMatch !== null && llmMatch !== yoloMatch) {
    const n = Math.min(appState.cruiseBuf.length, BACKFILL_MAX);
    logRow(
      'alert',
      '🔎 审计发现偏差：YOLO 巡航判为「' + (yoloMatch ? '命中' : '未命中')
        + '」，gpt-4o 实为「' + (llmMatch ? '命中' : '未命中') + '」，后台回填最近 ' + n + ' 帧（不阻塞实时）…',
      false,
      null,
    );
    backfillCruise(plan);
  } else if (appState.cruiseBuf.length) {
    logRow('normal', '✓ 审计通过：YOLO 巡航与 gpt-4o 一致，继续巡航（已确认 ' + appState.cruiseBuf.length + ' 帧）', false, null);
    appState.cruiseBuf = [];
  }
}

export async function backfillCruise(plan) {
  if (appState.backfilling) return;
  appState.backfilling = true;
  const buf = appState.cruiseBuf.slice(-BACKFILL_MAX);
  appState.cruiseBuf = [];

  try {
    for (const item of buf) {
      if (!appState.activeCompare) break;
      try {
        const result = await analyzeFrameRequest({
          image: item.dataUri,
          target: appState.activeCompare.target,
          reference_image: appState.activeCompare.reference,
          gate_enabled: false,
          comparing: true,
          plan,
        });
        if (!result.ok) continue;

        const data = result.data;
        const match = (data.result && data.result.match) || {};
        if (match.is_match !== item.verdict) {
          appState.stats.skipped = Math.max(0, appState.stats.skipped - 1);
          appState.stats.llm++;
          syncLlmSavedStats(appState.stats);

          if (match.is_match === true && appState.activeCompare) {
            appState.stats.match++;
            syncMatchStats(appState.stats);
          }

          if (appState.frameStore[item.fid]) {
            appState.frameStore[item.fid].res = data;
          }

          logRow(
            match.is_match === true ? 'alert' : 'normal',
            '⟲ 回填纠正 帧#' + item.fid + '：' + (match.is_match ? '实为命中「' + (match.target || '') + '」' : '实为未命中')
              + ' · ' + (match.reason || ''),
            match.is_match === true,
            item.fid,
            item.dataUri,
          );
        }
      } catch (e) {
        // 单帧回填失败忽略，保持与原脚本一致。
      }
    }
  } finally {
    appState.backfilling = false;
  }
}
