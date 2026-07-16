import { getLastPayload, registerKeyframe } from "./state.js";
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
        .join("；");
      return (
        `<div class="em-event"><span class="et">frame ${esc(frame.frame_index)} @ ${esc(frame.timestamp)}</span>` +
        `<span class="ea">${objects.length} objects${sample ? ` · ${sample}` : ""}</span></div>`
      );
    })
    .join("");

  const trajectoryRows = (grounding.trajectories || [])
    .slice(0, 8)
    .map(
      (trajectory) =>
        `<div class="em-event"><span class="es">${esc(trajectory.label)}</span>` +
        `<span class="ea">${esc(trajectory.direction)} · path ${esc(JSON.stringify(trajectory.path_sample || []))}</span></div>`
    )
    .join("");

  return `<details class="em-aux"><summary>📍 空间 grounding：关键帧坐标 + 轨迹摘要</summary>${frameRows}${trajectoryRows}</details>`;
}

function renderWindow(windowData) {
  const event = windowData.event || null;
  const level = (event && event.alert_level) || "normal";
  const groundingFrames = (windowData.spatial_grounding && windowData.spatial_grounding.frames) || [];
  const frames = (windowData.keyframes || [])
    .map((keyframe, index) => {
      const boxes = boxesForFrame(groundingFrames[index]);
      const caption = `关键帧 · ${keyframe.timestamp}${boxes.length ? ` · ${boxes.length} 个目标` : ""}`;
      const keyframeIndex = registerKeyframe({
        image: keyframe.image,
        boxes,
        caption,
      });
      return (
        `<div class="em-frame" data-kf="${keyframeIndex}" title="点击放大">` +
        `<img src="${keyframe.image}" loading="lazy"/>` +
        `<div class="em-boxes">${boxesHtml(boxes)}</div>` +
        `<span class="em-frame-ts">${esc(keyframe.timestamp)}</span></div>`
      );
    })
    .join("");

  const people = (windowData.people || [])
    .map((person) => {
      const label = person.subject_id != null ? `主体#${person.subject_id}` : `track ${person.track_id}`;
      const hue = subjectHue(person.subject_id);
      const thumb = thumbForTrack(person);
      const avatar = thumb ? `<img class="em-pavatar" src="${thumb}" loading="lazy"/>` : "";
      const cues = [];
      if (person.reid && person.reid.score != null) cues.push(`人形ReID ${(+person.reid.score).toFixed(2)}`);
      cues.push(person.face ? "有脸" : "无脸→人形为准");
      if (person.reused) cues.push("♻回头客");
      if (person.local_subject) cues.push("本视频本地subject");
      if (person.subject_conflict_split) cues.push("时间冲突已拆分");
      return (
        `<div class="em-person" style="--hue:${hue}">${avatar}` +
        `<span class="pl">${esc(label)}</span> <span class="pc">${esc(cues.join(" · "))}</span></div>`
      );
    })
    .join("");

  const scene = windowData.scene_context
    ? `<details class="em-aux"><summary>🔤 场景文字 OCR</summary><pre>${esc(windowData.scene_context)}</pre></details>`
    : "";
  const objects = windowData.object_context
    ? `<details class="em-aux"><summary>📦 物体 / 包裹</summary><pre>${esc(windowData.object_context)}</pre></details>`
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
      '<div class="em-summary" style="color:var(--muted)">（dry-run：未调用 LLM。以下为将喂给模型的关键帧与身份。）</div>' +
      (frames ? `<div class="em-frames">${frames}</div>` : "") +
      scene +
      objects +
      grounding;
  }

  return (
    `<div class="em-window ${esc(level)}">` +
    `<div class="em-window-head">` +
    `<span class="em-time">⏱ ${esc(windowData.time_range[0])} ~ ${esc(windowData.time_range[1])}</span>` +
    `<span class="em-badge ${esc(level)}">${esc(level)}</span>` +
    `<span class="em-time">${windowData.frame_count || 0} 帧 → ${(windowData.keyframe_indices || []).length} 关键帧</span>` +
    "</div>" +
    body +
    (people ? `<div class="em-people">${people}</div>` : "") +
    "</div>"
  );
}

export function renderTimeline(data) {
  return (data.windows || []).map(renderWindow).join("");
}
