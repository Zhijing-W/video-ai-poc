/* 事件监控台前端逻辑（Phase 4 · Step 26）
   选样片/上传 → POST /eventmonitor/understand → 渲染事件窗时间线。
   纯原生 JS，无依赖；复用 eventmonitor.css 的 .em-* 类。 */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  let lastPayload = null;
  const esc = (s) =>
    String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );

  // ---- 时钟 ----
  setInterval(() => {
    const d = new Date();
    $("clock").textContent = d.toLocaleTimeString("zh-CN", { hour12: false });
  }, 1000);

  // ---- 载入样片列表 ----
  async function loadSamples() {
    try {
      const r = await fetch("/eventmonitor/samples");
      const data = await r.json();
      const sel = $("sampleSelect");
      sel.innerHTML = "";
      (data.samples || []).forEach((s) => {
        const o = document.createElement("option");
        o.value = s.name;
        o.textContent = `${s.name} (${s.size_mb} MB)`;
        sel.appendChild(o);
      });
      if (!data.samples || !data.samples.length) {
        sel.innerHTML = '<option value="">（data/samples 下没有样片）</option>';
      }
    } catch (e) {
      $("sampleSelect").innerHTML = '<option value="">加载样片失败</option>';
    }
  }

  // ---- 提交分析 ----
  async function run() {
    const file = $("fileInput").files[0];
    const sample = $("sampleSelect").value;
    if (!file && !sample) {
      setStatus("请先选择样片或上传视频", true);
      return;
    }

    const fd = new FormData();
    if (file) fd.append("file", file);
    else fd.append("sample", sample);
    fd.append("fps", $("fps").value || "2");
    fd.append("max_keyframes", $("maxKeyframes").value || "8");
    if ($("objective").value.trim()) fd.append("objective", $("objective").value.trim());
    fd.append("with_face", $("withFace").checked ? "true" : "false");
    fd.append("with_gait", $("withGait").checked ? "true" : "false");
    fd.append("dry_run", $("dryRun").checked ? "true" : "false");
    // 模型/能力开关（本次生效；留空=用默认）
    if ($("faceRecBackend").value) fd.append("face_rec_backend", $("faceRecBackend").value);
    if ($("faceSuperres").value) fd.append("face_superres", $("faceSuperres").value);
    if ($("reidBackend").value) fd.append("reid_backend", $("reidBackend").value);
    fd.append("face_3d_cue", $("face3d").checked ? "true" : "false");
    if ($("maxWindowSeconds").value) fd.append("max_window_seconds", $("maxWindowSeconds").value);
    if ($("stitchThresh").value) fd.append("stitch_thresh", $("stitchThresh").value);

    $("btnRun").disabled = true;
    $("empty").style.display = "none";
    $("overall").hidden = true;
    $("timeline").innerHTML = "";
    $("tracks").innerHTML = "";
    $("meta").innerHTML = "";
    const llm = $("dryRun").checked ? "（dry-run，不调 LLM）" : "（含 gpt-4o，约 1 分钟）";
    setStatus("⏳ 处理中…抽帧→检测/跟踪→认人→事件理解 " + llm);

    const t0 = Date.now();
    try {
      const r = await fetch("/eventmonitor/understand", { method: "POST", body: fd });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${r.status}`);
      }
      const data = await r.json();
      render(data);
      setStatus(`✓ 完成，用时 ${((Date.now() - t0) / 1000).toFixed(1)}s`);
    } catch (e) {
      setStatus("✗ 失败：" + e.message, true);
      $("empty").style.display = "block";
      $("empty").textContent = "处理失败：" + e.message;
    } finally {
      $("btnRun").disabled = false;
    }
  }

  function setStatus(msg, err) {
    const el = $("status");
    el.textContent = msg;
    el.style.color = err ? "var(--alert)" : "var(--muted)";
  }

  // ---- 渲染结果 ----
  function render(data) {
    lastPayload = data;
    $("resultTools").hidden = false;
    $("jsonView").hidden = true;
    $("btnToggleJson").textContent = "查看原始 JSON";
    renderOverall(data.overall);
    // 元信息
    const cu = data.config_used || {};
    const cfgLine = cu.with_face
      ? ` · 人脸<b>${esc(cu.face_rec_backend)}</b>` +
        `${cu.face_superres && cu.face_superres !== "off" ? "+超分" : ""}` +
        `${cu.face_3d_cue ? "+3D" : ""}`
      : "";
    const gaitLine = cu.with_gait ? " · 步态<b>on</b>" : "";
    $("meta").innerHTML =
      `视频 <b>${esc(baseName(data.video))}</b> · ${data.frames_total} 帧 @ ${data.fps}fps · ` +
      `${(data.windows || []).length} 个事件窗 · ReID <b>${esc(data.reid_backend)}</b>(${data.reid_dim}d) · ` +
      `模型 <b>${esc(data.model)}</b>${data.dry_run ? " · <b>dry-run</b>" : ""}${cfgLine}${gaitLine} · ${data.elapsed_seconds}s`;

    // 主体记忆 chips（按 subject 聚合）
    const subjMap = {};
    Object.entries(data.tracks || {}).forEach(([tid, idn]) => {
      const sid = idn.subject_id == null ? "?" : idn.subject_id;
      (subjMap[sid] = subjMap[sid] || { tracks: [], reused: false, score: 0 });
      subjMap[sid].tracks.push(tid);
      if (idn.reused) subjMap[sid].reused = true;
      subjMap[sid].score = Math.max(subjMap[sid].score, idn.score || 0);
    });
    $("tracks").innerHTML = Object.entries(subjMap)
      .map(
        ([sid, v]) =>
          `<span class="em-chip">主体#${esc(sid)} · ${v.tracks.length} 条轨迹` +
          (v.reused ? ' · <span class="reused">♻ 回头客</span>' : "") +
          (v.score ? ` · ReID ${v.score.toFixed(2)}` : "") +
          `</span>`
      )
      .join("");

    // 事件窗时间线
    $("timeline").innerHTML = (data.windows || []).map(renderWindow).join("");
  }

  // ---- 整段事件总结（跨窗整合，置顶 headline）----
  function renderOverall(ov) {
    const el = $("overall");
    if (!ov || ov.error) {
      el.hidden = true;
      el.innerHTML = "";
      return;
    }
    const level = ov.overall_alert_level || "normal";
    const story = (ov.story || [])
      .map(
        (s) =>
          `<div class="em-event"><span class="et">${esc(s.time)}</span>` +
          `<span class="es">${esc(s.subject)}</span><span class="ea">${esc(s.action)}</span></div>`
      )
      .join("");
    const subs = (ov.subjects || []).map((s) => `<li>${esc(s)}</li>`).join("");
    el.hidden = false;
    el.className = "em-overall " + level;
    el.innerHTML =
      `<div class="em-window-head"><span class="em-otitle">📋 整段事件总结（跨窗整合）</span>` +
      `<span class="em-badge ${esc(level)}">${esc(level)}</span></div>` +
      `<div class="em-summary">${esc(ov.overall_summary)}</div>` +
      (ov.notification ? `<div class="em-notify">🔔 ${esc(ov.notification)}</div>` : "") +
      (story ? `<div class="em-events">${story}</div>` : "") +
      (subs ? `<ul class="em-subjects">${subs}</ul>` : "");
  }

  function renderWindow(w) {
    const ev = w.event || null;
    const level = (ev && ev.alert_level) || "normal";
    const frames = (w.keyframes || [])
      .map(
        (k) =>
          `<div class="em-frame"><img src="${k.image}" loading="lazy"/><span>${esc(k.timestamp)}</span></div>`
      )
      .join("");

    const people = (w.people || [])
      .map((p) => {
        const label =
          p.subject_id != null ? `主体#${p.subject_id}` : `track ${p.track_id}`;
        const cues = [];
        if (p.reid && p.reid.score != null) cues.push(`人形ReID ${(+p.reid.score).toFixed(2)}`);
        cues.push(p.face ? "有脸" : "无脸→人形为准");
        if (p.reused) cues.push("♻回头客");
        return `<div class="em-person"><span class="pl">${esc(label)}</span> <span class="pc">${esc(cues.join(" · "))}</span></div>`;
      })
      .join("");

    let body = "";
    if (ev) {
      const events = (ev.events || [])
        .map(
          (e) =>
            `<div class="em-event ${e.abnormal ? "abnormal" : ""}">` +
            `<span class="et">${esc(e.time)}</span>` +
            `<span class="es">${esc(e.subject)}</span>` +
            `<span class="ea">${e.abnormal ? '<span class="flag">⚠</span>' : ""}${esc(e.action)}</span></div>`
        )
        .join("");
      body =
        `<div class="em-summary">${esc(ev.summary)}</div>` +
        (ev.notification ? `<div class="em-notify">🔔 ${esc(ev.notification)}</div>` : "") +
        (frames ? `<div class="em-frames">${frames}</div>` : "") +
        (events ? `<div class="em-events">${events}</div>` : "");
    } else {
      // dry-run：无 LLM 事件，展示将要喂模型的关键帧 + 身份
      body =
        `<div class="em-summary" style="color:var(--muted)">（dry-run：未调用 LLM。以下为将喂给模型的关键帧与身份。）</div>` +
        (frames ? `<div class="em-frames">${frames}</div>` : "");
    }

    return (
      `<div class="em-window ${esc(level)}">` +
      `<div class="em-window-head">` +
      `<span class="em-time">⏱ ${esc(w.time_range[0])} ~ ${esc(w.time_range[1])}</span>` +
      `<span class="em-badge ${esc(level)}">${esc(level)}</span>` +
      `<span class="em-time">${w.frame_count} 帧 → ${w.keyframe_indices.length} 关键帧</span>` +
      `</div>` +
      body +
      (people ? `<div class="em-people">${people}</div>` : "") +
      `</div>`
    );
  }

  function baseName(p) {
    return String(p || "").split(/[\\/]/).pop();
  }

  // ---- 原始 JSON：下载 / 查看（剥掉关键帧 base64 大图，只留结构化数据）----
  function cleanedPayload() {
    if (!lastPayload) return null;
    const obj = JSON.parse(JSON.stringify(lastPayload));
    (obj.windows || []).forEach((w) => {
      (w.keyframes || []).forEach((k) => {
        if (k.image) k.image = "<data-uri omitted>";
      });
    });
    return obj;
  }

  function downloadJson() {
    const obj = cleanedPayload();
    if (!obj) return;
    const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    const stem = baseName(obj.video).replace(/\.[^.]+$/, "") || "result";
    a.download = `eventmonitor_${stem}${obj.dry_run ? "_dryrun" : ""}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function toggleJson() {
    const el = $("jsonView");
    if (el.hidden) {
      el.textContent = JSON.stringify(cleanedPayload(), null, 2);
      el.hidden = false;
      $("btnToggleJson").textContent = "收起 JSON";
    } else {
      el.hidden = true;
      $("btnToggleJson").textContent = "查看原始 JSON";
    }
  }

  // ---- 设置抽屉 ----
  function openSettings() {
    $("settingsDrawer").hidden = false;
    $("settingsOverlay").hidden = false;
  }
  function closeSettings() {
    $("settingsDrawer").hidden = true;
    $("settingsOverlay").hidden = true;
  }

  // ---- 初始化 ----
  $("btnRun").addEventListener("click", run);
  $("btnDownloadJson").addEventListener("click", downloadJson);
  $("btnToggleJson").addEventListener("click", toggleJson);
  $("btnSettings").addEventListener("click", openSettings);
  $("btnCloseSettings").addEventListener("click", closeSettings);
  $("btnApplySettings").addEventListener("click", closeSettings);
  $("settingsOverlay").addEventListener("click", closeSettings);
  loadSamples();
})();
