import { getLastPayload, registerKeyframe } from "./state.js";
import { labelLevel, t } from "./i18n.js";
import { esc, subjectHue } from "./utils.js";

function boxesForFrame(groundingFrame) {
  if (!groundingFrame || !Array.isArray(groundingFrame.objects)) return [];
  return groundingFrame.objects
    .map((object) => {
      const bboxNorm = object.bbox_norm || [];
      if (bboxNorm.length < 4) return null;
      const [x1, y1, x2, y2] = bboxNorm;
      return {
        x: x1 * 100,
        y: y1 * 100,
        w: Math.max(0, x2 - x1) * 100,
        h: Math.max(0, y2 - y1) * 100,
        label: object.subject_id != null ? `#${object.subject_id}` : `t${object.track_id}`,
        hue: subjectHue(object.subject_id),
      };
    })
    .filter(Boolean);
}

export function boxesHtml(boxes) {
  return boxes
    .map(
      (box) =>
        `<div class="em-box" style="left:${box.x.toFixed(2)}%;top:${box.y.toFixed(2)}%;width:${box.w.toFixed(2)}%;height:${box.h.toFixed(2)}%;--hue:${box.hue}">` +
        `<span class="em-box-tag">${esc(box.label)}</span></div>`
    )
    .join("");
}

function thumbForTrack(person) {
  const tracks = (getLastPayload() && getLastPayload().tracks) || {};
  const ids =
    (person.source_track_ids && person.source_track_ids.length ? person.source_track_ids : [person.track_id]) || [];
  for (const trackId of ids) {
    const record = tracks[String(trackId)];
    if (record && record.thumb) return record.thumb;
  }
  return null;
}

function renderGrounding(grounding) {
  if (!grounding || !Array.isArray(grounding.frames)) return "";
  const frameRows = grounding.frames
    .slice(0, 8)
    .map((frame) => {
      const objects = frame.objects || [];
      const sample = objects
        .slice(0, 4)
        .map((object) => `${esc(object.label)} c=${esc(JSON.stringify(object.center_norm || []))}`)
        .join(" · ");
      return (
        `<div class="em-event"><span class="et">${esc(t("timeline.frame_row", {
          frame: frame.frame_index,
          timestamp: frame.timestamp,
        }))}</span>` +
        `<span class="ea">${esc(t("timeline.objects_suffix", {
          count: objects.length,
          sample: sample ? t("timeline.objects_sample", { sample }) : "",
        }))}</span></div>`
      );
    })
    .join("");

  const trajectoryRows = (grounding.trajectories || [])
    .slice(0, 8)
    .map(
      (trajectory) =>
        `<div class="em-event"><span class="es">${esc(trajectory.label)}</span>` +
        `<span class="ea">${esc(t("timeline.trajectory_row", {
          direction: trajectory.direction,
          path: JSON.stringify(trajectory.path_sample || []),
        }))}</span></div>`
    )
    .join("");

  return `<details class="em-aux"><summary>${esc(t("timeline.grounding_summary"))}</summary>${frameRows}${trajectoryRows}</details>`;
}

function renderWindow(windowData) {
  const event = windowData.event || null;
  const level = (event && event.alert_level) || "normal";
  const groundingFrames = (windowData.spatial_grounding && windowData.spatial_grounding.frames) || [];
  const frames = (windowData.keyframes || [])
    .map((keyframe, index) => {
      const boxes = boxesForFrame(groundingFrames[index]);
      const caption = t("timeline.keyframe_caption", {
        timestamp: keyframe.timestamp,
        objects: boxes.length ? t("timeline.keyframe_objects", { count: boxes.length }) : "",
      });
      const keyframeIndex = registerKeyframe({
        image: keyframe.image,
        boxes,
        caption,
      });
      return (
        `<div class="em-frame" data-kf="${keyframeIndex}" title="${esc(t("timeline.click_to_zoom"))}">` +
        `<img src="${keyframe.image}" loading="lazy"/>` +
        `<div class="em-boxes">${boxesHtml(boxes)}</div>` +
        `<span class="em-frame-ts">${esc(keyframe.timestamp)}</span></div>`
      );
    })
    .join("");

  const people = (windowData.people || [])
    .map((person) => {
      const label = person.subject_id != null
        ? t("timeline.person_subject", { id: person.subject_id })
        : t("timeline.person_track", { id: person.track_id });
      const hue = subjectHue(person.subject_id);
      const thumb = thumbForTrack(person);
      const avatar = thumb ? `<img class="em-pavatar" src="${thumb}" loading="lazy"/>` : "";
      const cues = [];
      if (person.reid && person.reid.score != null) cues.push(t("timeline.person_body_score", { score: (+person.reid.score).toFixed(2) }));
      const faceUsable =
        person.face &&
        person.face.observed !== false &&
        person.face.eligibility !== "none" &&
        person.face.match_ready;
      cues.push(faceUsable
        ? t(person.face.match_source === "superres" ? "timeline.person_face_superres" : "timeline.person_face_ready")
        : t("timeline.person_face_missing"));
      if (person.reused) cues.push(t("timeline.person_return"));
      if (person.local_subject) cues.push(t("timeline.person_local"));
      if (person.subject_conflict_split) cues.push(t("timeline.person_split"));
      return (
        `<div class="em-person" style="--hue:${hue}">${avatar}` +
        `<span class="pl">${esc(label)}</span> <span class="pc">${esc(cues.join(" · "))}</span></div>`
      );
    })
    .join("");

  const scene = windowData.scene_context
    ? `<details class="em-aux"><summary>${esc(t("timeline.scene_summary"))}</summary><pre>${esc(windowData.scene_context)}</pre></details>`
    : "";
  const objects = windowData.object_context
    ? `<details class="em-aux"><summary>${esc(t("timeline.objects_summary"))}</summary><pre>${esc(windowData.object_context)}</pre></details>`
    : "";
  const grounding = renderGrounding(windowData.spatial_grounding);

  let body = "";
  if (event) {
    const events = (event.events || [])
      .map(
        (item) =>
          `<div class="em-event ${item.abnormal ? "abnormal" : ""}">` +
          `<span class="et">${esc(item.time)}</span>` +
          `<span class="es">${esc(item.subject)}</span>` +
          `<span class="ea">${item.abnormal ? '<span class="flag">⚠</span>' : ""}${esc(item.action)}</span></div>`
      )
      .join("");

    body =
      `<div class="em-summary">${esc(event.summary)}</div>` +
      (event.notification ? `<div class="em-notify">🔔 ${esc(event.notification)}</div>` : "") +
      (frames ? `<div class="em-frames">${frames}</div>` : "") +
      (events ? `<div class="em-events">${events}</div>` : "") +
      scene +
      objects +
      grounding;
  } else {
    body =
      `<div class="em-summary" style="color:var(--muted)">${esc(t("results.dry_run_summary"))}</div>` +
      (frames ? `<div class="em-frames">${frames}</div>` : "") +
      scene +
      objects +
      grounding;
  }

  return (
    `<div class="em-window ${esc(level)}">` +
    `<div class="em-window-head">` +
    `<span class="em-time">${esc(t("timeline.window_head_time", {
      start: windowData.time_range[0],
      end: windowData.time_range[1],
    }))}</span>` +
    `<span class="em-badge ${esc(level)}">${esc(labelLevel(level))}</span>` +
    `<span class="em-time">${esc(t("timeline.window_head_counts", {
      frames: windowData.frame_count || 0,
      keyframes: (windowData.keyframe_indices || []).length,
    }))}</span>` +
    "</div>" +
    body +
    (people ? `<div class="em-people">${people}</div>` : "") +
    "</div>"
  );
}

export function renderTimeline(data) {
  return (data.windows || []).map(renderWindow).join("");
}
