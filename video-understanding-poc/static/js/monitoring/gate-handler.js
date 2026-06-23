/**
 * 门控与比对模块
 * 提取自 monitor.html.backup-refactor：负责目标编译、巡航开关判断与比对状态切换。
 */

import { appState, clearActiveCompare, resetCruiseState } from '../core/state.js';
import { $, escapeHtml } from '../core/utils.js';
import { compileTargetRequest } from '../core/api.js';
import { nstat } from '../visualization/flow-diagram.js';
import { setCompareStoppedCard, syncDashMatchMirror } from '../visualization/stats-display.js';

export function shouldUseCruise(plan) {
  return !!(plan && plan.can_yolo_handle && $('cruiseToggle')?.checked);
}

export function buildAnalyzePayload(dataUri, plan) {
  // Phase 3 · Step 12：Track 门控仅在「非比对」实时监控启用（比对/审计仍每帧裁决）。
  const trackEnabled = !!($('trackToggle')?.checked) && !appState.activeCompare;
  const payload = {
    image: dataUri,
    target: null,
    reference_image: null,
    gate_enabled: $('gateToggle')?.checked,
    prev_counts: appState.lastCounts,
    since_last_llm_ms: appState.lastLlmTs ? (Date.now() - appState.lastLlmTs) : null,
    comparing: !!appState.activeCompare,
    plan: plan || null,
    last_llm_signature: appState.lastLlmSig,
    track_enabled: trackEnabled,
    session_id: appState.trackSessionId,
    yolo_model: appState.yoloModelSel,
    llm_model: appState.llmModelSel,
  };

  if (appState.activeCompare) {
    payload.target = appState.activeCompare.target;
    payload.reference_image = appState.activeCompare.reference;
  }
  return payload;
}

export function showPlanStatus(plan) {
  if (!plan) {
    if ($('planStatus')) $('planStatus').innerHTML = '⚠ 未能编译，回落每帧 gpt-4o。';
    return;
  }

  const attr = plan.attribute;
  const desc = (plan.yolo_class || '—')
    + (attr && attr.value ? ('（' + attr.value + (attr.region && attr.region !== 'whole' ? '/' + attr.region : '') + '）') : '');

  if (plan.can_yolo_handle) {
    if ($('planStatus')) {
      $('planStatus').innerHTML = '✅ YOLO 可独立巡航：监视 <b>' + escapeHtml(desc) + '</b>，每 '
        + (Math.max(2, Math.min(60, parseInt($('auditEvery')?.value, 10) || 8)))
        + ' 帧用 gpt-4o 审计回填。<br><span style=\'color:var(--muted)\'>' + escapeHtml(plan.summary || '') + '</span>';
    }
    $('nd-cruise')?.classList.add('on');
    nstat('cruise', 'YOLO 巡航：' + desc);
  } else {
    if ($('planStatus')) {
      $('planStatus').innerHTML = '⚠ 需每帧 gpt-4o：YOLO 无法判断该目标（姿态/动作/身份等）。<br><span style=\'color:var(--muted)\'>' + escapeHtml(plan.summary || '') + '</span>';
    }
    $('nd-cruise')?.classList.remove('on');
    nstat('cruise', '不可巡航 · 每帧 gpt-4o');
  }
}

export async function handleCompareStart() {
  const mode = document.querySelector('input[name="cmode"]:checked')?.value;
  const target = $('target')?.value.trim();

  if ((mode === 'text' || mode === 'both') && !target) {
    alert('请先填写文字目标描述');
    return;
  }
  if ((mode === 'image' || mode === 'both') && !appState.refDataUri) {
    alert('请先上传参考图片');
    return;
  }

  appState.activeCompare = {
    mode,
    target: (mode === 'image') ? null : target,
    reference: (mode === 'text') ? null : appState.refDataUri,
    plan: null,
  };
  resetCruiseState();

  const label = { text: '文字描述', image: '参考图片', both: '文字+图片' }[mode];
  if ($('compareState')) {
    $('compareState').textContent = '比对中 · 标准：' + label + (appState.activeCompare.target ? '（' + appState.activeCompare.target + '）' : '');
    $('compareState').className = 'hint compareState on';
  }
  if ($('btnCompare')) $('btnCompare').disabled = true;
  if ($('btnCompareStop')) $('btnCompareStop').disabled = false;
  $('nd-compile')?.classList.add('on');
  nstat('compile', '比对中 · ' + label + (appState.activeCompare.target ? '「' + appState.activeCompare.target + '」' : ''));
  nstat('cruise', '编译中…');

  if (appState.activeCompare.target) {
    if ($('planStatus')) $('planStatus').innerHTML = '⏳ 正在编译目标…';
    try {
      const result = await compileTargetRequest({
        target: appState.activeCompare.target,
        reference_image: appState.activeCompare.reference,
      });
      if (result.ok) {
        const data = result.data;
        appState.activeCompare.plan = data.plan || null;
        showPlanStatus(data.plan);
      } else if ($('planStatus')) {
        $('planStatus').innerHTML = '⚠ 目标编译失败，回落每帧 gpt-4o';
      }
    } catch (e) {
      if ($('planStatus')) $('planStatus').innerHTML = '⚠ 目标编译异常，回落每帧 gpt-4o';
    }
  } else if ($('planStatus')) {
    $('planStatus').innerHTML = '参考图模式：YOLO 无法独立比对，使用每帧 gpt-4o。';
  }
}

export function handleCompareStop() {
  clearActiveCompare();
  if ($('compareState')) {
    $('compareState').textContent = '未比对（仅做画面理解）';
    $('compareState').className = 'hint';
  }
  if ($('planStatus')) $('planStatus').innerHTML = '编译目标后显示巡航计划。';
  if ($('btnCompare')) $('btnCompare').disabled = false;
  if ($('btnCompareStop')) $('btnCompareStop').disabled = true;
  $('nd-compile')?.classList.remove('on');
  $('nd-cruise')?.classList.remove('on');
  nstat('compile', '未设比对目标');
  nstat('cruise', '未启用');
  $('nd-result')?.classList.remove('hit', 'warn');
  setCompareStoppedCard();
  syncDashMatchMirror(null);
}

export function initGateHandler() {
  if ($('btnCompare')) $('btnCompare').onclick = handleCompareStart;
  if ($('btnCompareStop')) $('btnCompareStop').onclick = handleCompareStop;
}
