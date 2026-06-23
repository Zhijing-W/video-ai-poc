/**
 * 整段视频分析 · 末尾总结
 * 视频文件走和实时流一样的逐帧管线（结果实时进日志、点亮流程图）；播完后把累积的
 * 逐帧事件发后端 /summarize 归纳成一段总结，显示在日志与结果区，并随归档保存。
 * 不再有"上传 + 弹窗报告"那套。
 */

import { $, escapeHtml } from '../core/utils.js';
import { logRow } from '../ui/render-engine.js';
import { nstat } from '../visualization/flow-diagram.js';

// 把本次累积的逐帧事件发后端，归纳成末尾总结。
export async function fetchSummary(entries) {
  const events = (entries || []).map((e) => ({
    timestamp: e.ts,
    observation: e.msg,
    alert_level: e.level,
  }));
  const r = await fetch('/summarize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ events }),
  });
  if (!r.ok) throw new Error((await r.text()) || ('HTTP ' + r.status));
  return await r.json();
}

// 把总结显示到日志（一条醒目记录）+ 结果区的"当前画面理解"。
export function showSummary(summary) {
  if (!summary) return;
  const level = summary.overall_alert_level || 'normal';
  const objs = (summary.detected_objects || []).join('、');
  const msg = '🎬 整段分析完成 · ' + (summary.summary || '')
    + (objs ? ' · 对象：' + objs : '') + ' · 告警 ' + level;
  logRow(level === 'alert' ? 'alert' : 'normal', msg, false, null);

  if ($('scene')) $('scene').textContent = summary.summary || '—';
  if ($('dash-scene')) $('dash-scene').textContent = summary.summary || '—';
  if ($('resultMini')) {
    $('resultMini').style.display = 'block';
    $('resultMini').innerHTML = '🎬 <b>本次总结</b>：' + escapeHtml(summary.summary || '')
      + '<br>告警 ' + escapeHtml(level) + ' · 置信度 ' + escapeHtml(summary.confidence || '-');
  }
  nstat('result', '整段总结完成 ✓');
}

export function initBatchReport() {
  // 整段视频分析开关：dash 镜像到 tech，并把"开始"按钮文案改成对应动作。
  const sync = () => {
    const on = !!$('batchMode')?.checked;
    const txt = on ? '▶ 分析整段' : '▶ 开始监控';
    if ($('btnStart')) $('btnStart').textContent = txt;
    if ($('btnStart-dash')) $('btnStart-dash').textContent = txt;
  };
  if ($('batchMode')) $('batchMode').onchange = sync;
  if ($('batchMode-dash')) {
    $('batchMode-dash').onchange = () => {
      if ($('batchMode')) $('batchMode').checked = $('batchMode-dash').checked;
      sync();
    };
  }
}
