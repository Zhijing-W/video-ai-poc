import { esc, subjectHue } from "./utils.js";

export function routeBadges(record) {
  const face = record.face || null;
  const gait = record.gait || null;
  const fused = record.fused || null;
  const badges = [];

  if (face) {
    const quality = face.quality || "?";
    const good = face.matched || quality === "clear";
    badges.push(
      `<span class="em-rb ${good ? "hit" : "weak"}" title="人脸 · 质量${esc(quality)}${face.match_score != null ? ` · 相似${(+face.match_score).toFixed(2)}` : ""}">脸</span>`
    );
  } else {
    badges.push('<span class="em-rb off" title="无脸 → 退人形/步态">脸</span>');
  }

  if (record.score != null) {
    badges.push(`<span class="em-rb hit" title="人形 ReID 相似 ${(+record.score).toFixed(2)}">形</span>`);
  } else {
    badges.push('<span class="em-rb off" title="无人形分">形</span>');
  }

  if (gait && gait.score != null) {
    badges.push(
      `<span class="em-rb ${gait.decision === "hit" ? "hit" : "weak"}" title="步态 · ${esc(gait.decision || "")} · 相似 ${(+gait.score).toFixed(2)} · ${gait.frames || 0}帧">步</span>`
    );
  } else {
    badges.push('<span class="em-rb off" title="步态未启用/帧不足">步</span>');
  }

  let confidence = "";
  if (fused && fused.confidence != null) {
    const level = fused.resolved ? "hit" : "weak";
    const primaryCn = { face: "脸", body: "形", gait: "步" }[fused.primary] || "—";
    const multiSource = fused.multi_source ?? fused.agreed;
    confidence =
      `<span class="em-conf ${level}" title="多路线身份置信 · 主导线索 ${esc(primaryCn)}${multiSource ? " · 多路线参与" : ""}">` +
      `置信 ${(fused.confidence * 100).toFixed(0)}%${multiSource ? " ✓" : ""}</span>`;
  }

  return `<span class="em-routes">${badges.join("")}${confidence}</span>`;
}

export function renderSubjectGallery(data) {
  const tracks = data.tracks || {};
  const subjectMap = {};

  Object.entries(tracks).forEach(([trackId, identity]) => {
    const subjectId = identity.subject_id == null ? "?" : identity.subject_id;
    const group = (subjectMap[subjectId] = subjectMap[subjectId] || {
      tracks: [],
      reused: false,
      local: false,
      split: false,
      score: 0,
      best: null,
      bestScore: -1,
    });

    group.tracks.push(trackId);
    if (identity.reused) group.reused = true;
    if (identity.local_subject) group.local = true;
    if (identity.subject_conflict_split) group.split = true;
    group.score = Math.max(group.score, identity.score || 0);

    const score = identity.score || 0;
    if (identity.thumb && score >= group.bestScore) {
      group.bestScore = score;
      group.best = identity;
    }
    if (!group.best && identity.thumb) group.best = identity;
    if (!group.best) group.best = group.best || identity;
  });

  const cards = Object.entries(subjectMap).map(([subjectId, group]) => {
    const record = group.best || {};
    const hue = subjectHue(subjectId === "?" ? null : subjectId);
    const title = subjectId === "?" ? "未入库 · 待定身份" : `主体 #${esc(subjectId)}`;
    const thumb = record.thumb
      ? `<img src="${record.thumb}" alt="${esc(title)}" loading="lazy"/>`
      : '<span class="em-avatar-ph">?</span>';
    const flags = [];

    if (group.reused) flags.push('<span class="reused">♻ 回头客</span>');
    if (group.local) flags.push("本地subject");
    if (group.split) flags.push("时间冲突拆分");

    return (
      `<div class="em-subcard" style="--hue:${hue}">` +
      `<div class="em-avatar">${thumb}</div>` +
      `<div class="em-subinfo">` +
      `<div class="em-subtitle">${esc(title)}</div>` +
      `<div class="em-submeta">${group.tracks.length} 条轨迹${flags.length ? ` · ${flags.join(" · ")}` : ""}</div>` +
      routeBadges(record) +
      "</div></div>"
    );
  });

  if (!cards.length) return "";
  return (
    `<div class="em-gallery-head">👥 身份画廊（本段共 ${cards.length} 个主体）</div>` +
    `<div class="em-gallery">${cards.join("")}</div>`
  );
}
