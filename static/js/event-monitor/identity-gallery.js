import { common, t } from "./i18n.js";
import { esc, subjectHue } from "./utils.js";

export function routeBadges(record) {
  const face = record.face || null;
  const gait = record.gait || null;
  const fused = record.fused || null;
  const badges = [];

  if (face && face.observed !== false && face.eligibility !== "none") {
    const quality = face.quality || "?";
    const eligibility = face.eligibility || "?";
    const source = face.match_source || "none";
    const frame = face.evidence && face.evidence.frame_index != null
      ? t("gallery.face_frame", { frame: face.evidence.frame_index })
      : "";
    const good = face.matched || face.match_ready;
    const score = face.match_score != null
      ? t("gallery.face_score", { score: (+face.match_score).toFixed(2) })
      : "";
    badges.push(
      `<span class="em-rb ${good ? "hit" : "weak"}" title="${esc(t("gallery.face_title", {
        eligibility,
        quality,
        source,
        frame,
        score,
      }))}">${esc(t("gallery.face_badge"))}</span>`
    );
  } else {
    badges.push(`<span class="em-rb off" title="${esc(t("gallery.face_off_title"))}">${esc(t("gallery.face_badge"))}</span>`);
  }

  if (record.score != null) {
    badges.push(`<span class="em-rb hit" title="${esc(t("gallery.body_title", { score: (+record.score).toFixed(2) }))}">${esc(t("gallery.body_badge"))}</span>`);
  } else {
    badges.push(`<span class="em-rb off" title="${esc(t("gallery.body_off_title"))}">${esc(t("gallery.body_badge"))}</span>`);
  }

  if (gait && gait.score != null) {
    badges.push(
      `<span class="em-rb ${gait.decision === "hit" ? "hit" : "weak"}" title="${esc(t("gallery.gait_title", {
        decision: gait.decision || "",
        score: (+gait.score).toFixed(2),
        frames: gait.frames || 0,
      }))}">${esc(t("gallery.gait_badge"))}</span>`
    );
  } else {
    badges.push(`<span class="em-rb off" title="${esc(t("gallery.gait_off_title"))}">${esc(t("gallery.gait_badge"))}</span>`);
  }

  let confidence = "";
  if (fused && fused.confidence != null) {
    const level = fused.resolved ? "hit" : "weak";
    const multiSource = fused.multi_source ?? fused.agreed;
    const agreed = fused.agreed === true;
    confidence =
      `<span class="em-conf ${level}" title="${esc(t("gallery.confidence_title", {
        primary: { face: common("primary_face"), body: common("primary_body"), gait: common("primary_gait") }[fused.primary] || common("not_available"),
        multi: multiSource ? t("gallery.confidence_multi") : "",
        agreed: agreed ? t("gallery.confidence_agreed") : "",
      }))}">` +
      `${esc(t("gallery.confidence_label", {
        score: (fused.confidence * 100).toFixed(0),
        agreed_mark: agreed ? " ✓" : "",
      }))}</span>`;
  }

  return `<span class="em-routes">${badges.join("")}${confidence}</span>`;
}

function decisionLabel(decision) {
  const known = new Set(["hit", "new", "grey", "stitched", "conflict_split"]);
  return t(`gallery.decision_${known.has(decision) ? decision : "unknown"}`);
}

function evidenceStatus(record, route) {
  if (route === "face") {
    const face = record.face;
    return face && face.observed !== false && face.eligibility !== "none"
      ? t("gallery.evidence_available")
      : t("gallery.evidence_unavailable");
  }
  const gait = record.gait;
  return gait && gait.score != null
    ? t("gallery.evidence_available")
    : t("gallery.evidence_unavailable");
}

function displayScore(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(2) : common("not_available");
}

function detailRow(label, value) {
  return `<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`;
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
      bestBodyScore: null,
      best: null,
      bestScore: -1,
      decisions: new Set(),
    });

    group.tracks.push(trackId);
    if (identity.decision) group.decisions.add(identity.decision);
    if (identity.reused) group.reused = true;
    if (identity.local_subject) group.local = true;
    if (identity.subject_conflict_split) group.split = true;
    if (Number.isFinite(Number(identity.score))) {
      group.bestBodyScore = group.bestBodyScore === null
        ? Number(identity.score)
        : Math.max(group.bestBodyScore, Number(identity.score));
    }

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
    const title = subjectId === "?"
      ? t("gallery.unknown_title")
      : t("gallery.subject_title", { id: esc(subjectId) });
    const thumb = record.thumb
      ? `<img src="${record.thumb}" alt="${esc(title)}" loading="lazy"/>`
      : '<span class="em-avatar-ph">?</span>';
    const flags = [];

    if (group.reused) flags.push(`<span class="reused">${esc(t("gallery.return_visitor"))}</span>`);
    if (group.local) flags.push(esc(t("gallery.local_subject")));
    if (group.split) flags.push(esc(t("gallery.split_subject")));
    const decisions = group.decisions.size
      ? [...group.decisions].map(decisionLabel).join(", ")
      : decisionLabel();
    const fused = record.fused || {};
    const primary = {
      face: common("primary_face"),
      body: common("primary_body"),
      gait: common("primary_gait"),
    }[fused.primary] || common("not_available");
    const details = [
      detailRow(t("gallery.track_ids"), group.tracks.join(", ")),
      detailRow(t("gallery.decision"), decisions),
      detailRow(t("gallery.reused"), t(group.reused ? "gallery.yes" : "gallery.no")),
      detailRow(t("gallery.local"), t(group.local ? "gallery.yes" : "gallery.no")),
      detailRow(t("gallery.conflict"), t(group.split ? "gallery.yes" : "gallery.no")),
      detailRow(t("gallery.best_body_score"), displayScore(group.bestBodyScore)),
      detailRow(t("gallery.face_evidence"), evidenceStatus(record, "face")),
      detailRow(t("gallery.gait_evidence"), evidenceStatus(record, "gait")),
    ];
    if (Number.isFinite(Number(fused.confidence))) {
      details.push(detailRow(t("gallery.fused_confidence"), `${(Number(fused.confidence) * 100).toFixed(0)}%`));
    }
    if (fused.primary) details.push(detailRow(t("gallery.primary_route"), primary));

    return (
      `<details class="em-subcard" style="--hue:${hue}">` +
      `<summary class="em-subcard-summary" aria-label="${esc(t("gallery.details_toggle", { subject: title }))}">` +
      `<div class="em-avatar">${thumb}</div>` +
      `<div class="em-subinfo">` +
      `<div class="em-subtitle">${esc(title)}</div>` +
      `<div class="em-submeta">${esc(t("gallery.tracks", { count: group.tracks.length }))}${flags.length ? ` · ${flags.join(" · ")}` : ""}</div>` +
      routeBadges(record) +
      `</div></summary>` +
      `<div class="em-subdetails"><div class="em-subdetails-title">${esc(t("gallery.details"))}</div>` +
      `<dl class="em-subdetails-grid">${details.join("")}</dl></div></details>`
    );
  });

  if (!cards.length) return "";
  return (
    `<section class="em-gallery-panel">` +
    `<div class="em-gallery-head">${esc(t("gallery.head", { count: cards.length }))}</div>` +
    `<div class="em-gallery">${cards.join("")}</div>` +
    `</section>`
  );
}
