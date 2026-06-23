/**
 * 统计与摘要展示模块
 * 提取自 monitor.html.backup-refactor：负责计数面板、画面理解、目标档案与比对卡同步。
 */

import { $, escapeHtml } from '../core/utils.js';

function setText(id, value) {
  const el = $(id);
  if (el) el.textContent = value;
}

function setHtml(id, value) {
  const el = $(id);
  if (el) el.innerHTML = value;
}

export function resetStatsUI() {
  setText('sFrames', 0);
  setText('sMatch', 0);
  setText('sAlert', 0);
  setText('sLlm', 0);
  setText('sSaved', 0);
  setText('sSavePct', '0%');
  setText('dash-frames', 0);
  setText('dash-match-count', 0);
  setText('dash-alert', 0);
  setText('dash-llm', 0);
  setText('dash-save-pct', '0%');
  // Phase 3 · "连"：主体记忆读数复位 + 隐藏卡片
  setText('sSubjects', 0);
  setText('sReappear', 0);
  setHtml('identityNow', '');
  const idCard = $('identityCard');
  if (idCard) idCard.style.display = 'none';
}

export function syncFrameStats(stats) {
  setText('sFrames', stats.frames);
  setText('dash-frames', stats.frames);
}

export function syncLlmSavedStats(stats) {
  setText('sLlm', stats.llm);
  setText('sSaved', stats.skipped);
  const savePct = stats.frames ? Math.round(stats.skipped / stats.frames * 100) + '%' : '0%';
  setText('sSavePct', savePct);
  setText('dash-llm', stats.llm);
  setText('dash-save-pct', savePct);
}

export function syncMatchStats(stats) {
  setText('sMatch', stats.match);
  setText('dash-match-count', stats.match);
}

export function syncAlertStats(stats) {
  setText('sAlert', stats.alert);
  setText('dash-alert', stats.alert);
}

export function syncCorner(tag) {
  setText('corner', '● REC  ' + tag);
  setText('corner-dash', '● REC  ' + tag);
}

export function setResultChip(text) {
  const chip = $('resChip');
  if (chip) chip.textContent = text;
}

export function syncHudFromMain() {
  const hudEl = $('hud');
  const hudDashEl = $('hud-dash');
  if (hudEl && hudDashEl) hudDashEl.innerHTML = hudEl.innerHTML;
}

export function buildTagsHtml(detectedObjects) {
  return (detectedObjects || []).map((obj) => `<span class="tag">${escapeHtml(obj)}</span>`).join('');
}

export function syncSceneAndTags(sceneText, detectedObjects) {
  setText('scene', sceneText);
  setText('dash-scene', sceneText);

  const tagsHtml = buildTagsHtml(detectedObjects);
  setHtml('tags', tagsHtml);
  setHtml('dash-tags', tagsHtml.replace(/class="tag"/g, 'class="dash-tag"'));
}

export function buildSubjectsHtml(subjects) {
  if (!subjects.length) return '';
  return '<div class="subjhead">目标档案 · 基于 YOLO 框</div>' + subjects.map((subject) => {
    const attrs = (subject.attributes || []).map((attr) => `<span class="atag">${escapeHtml(attr)}</span>`).join('');
    const box = (subject.box || []).join(', ');
    return `<div class="subj"><div class="subjline"><b>${escapeHtml(subject.ref || '')} ${escapeHtml(subject.label || '')}</b>`
      + (box ? `<span class="box">[${box}]</span>` : '') + `</div>`
      + `<div class="subjapp">${escapeHtml(subject.appearance || '')}</div>`
      + (attrs ? `<div class="atags">${attrs}</div>` : '') + `</div>`;
  }).join('');
}

export function syncSubjects(subjects) {
  const html = buildSubjectsHtml(subjects || []);
  setHtml('subjects', html);
  setHtml('dash-subjects', html);
}

export function syncIdentity(identity) {
  // Phase 3 · "连"：把"认人"结果显示出来——记忆库主体数、回头客命中、本帧各 track 的身份。
  const card = $('identityCard');
  if (!identity) {
    if (card) card.style.display = 'none';
    return;
  }
  if (card) card.style.display = 'block';
  setText('sSubjects', identity.known_subjects != null ? identity.known_subjects : 0);
  setText('sReappear', identity.cross_track_hits != null ? identity.cross_track_hits : 0);

  const items = (identity.per_track || [])
    .filter((t) => t.subject_id != null)
    .map((t) => {
      const mark = t.decision === 'resolved' ? '✓' : (t.decision === 'uncertain' ? '?' : '·');
      const cls = t.reused ? 'atag' : 'tag';
      return `<span class="${cls}">主体#${t.subject_id} ${mark}${t.reused ? ' ♻' : ''}</span>`;
    })
    .join('');
  setHtml('identityNow', items
    || '<span style="font-size:.75rem;color:var(--muted)">本帧无已识别主体</span>');
}

export function setIdleMatchCard(alert, sceneText) {
  const card = $('matchCard');
  if (card) card.className = 'matchcard ' + (alert ? 'alert' : 'nomatch');
  setText('matchTarget', '未设置比对目标');
  setText('matchBig', alert ? '⚠ 注意' : '监控中');
  setText('matchConf', '');
  setText('matchReason', sceneText || '');
}

export function setActiveMatchCard(match, alert) {
  const card = $('matchCard');
  const hasTarget = match.target && match.target.length > 0;
  setText('matchTarget', hasTarget ? ('目标：' + match.target) : '目标：参考图片');
  if (card) card.className = 'matchcard ' + (match.is_match === true ? 'match' : alert ? 'alert' : 'nomatch');

  if (match.is_match === true) setText('matchBig', '✓ 命中');
  else if (match.is_match === false) setText('matchBig', '✗ 未命中');
  else setText('matchBig', alert ? '⚠ 注意' : '监控中');

  setText('matchConf', match.confidence ? ('置信度 ' + match.confidence) : '');
  setText('matchReason', match.reason || '');
  return match.is_match === true;
}

export function setCruiseMatchCard(cruise, activeCompare) {
  const card = $('matchCard');
  setText('matchTarget', '目标：' + (activeCompare?.target || '参考图片') + ' · YOLO 巡航');

  if (cruise.is_match === true) {
    if (card) card.className = 'matchcard match';
    setText('matchBig', '✓ 命中（巡航）');
  } else {
    if (card) card.className = 'matchcard nomatch';
    setText('matchBig', '巡航中');
  }
  setText('matchConf', '');
  setText('matchReason', cruise.reason || '');
  return cruise.is_match === true;
}

export function setCompareStoppedCard() {
  const card = $('matchCard');
  if (card) card.className = 'matchcard nomatch';
  setText('matchBig', '监控中');
  setText('matchTarget', '未设置比对目标');
  setText('matchConf', '');
  setText('matchReason', '');
}

export function syncDashMatchMirror(activeCompare) {
  setText('dash-match-big', $('matchBig')?.textContent || '');
  setText('dash-match-reason', $('matchReason')?.textContent || '');
  const dashCard = $('dash-match-card');
  if (dashCard) dashCard.style.display = activeCompare ? 'flex' : 'none';
}
