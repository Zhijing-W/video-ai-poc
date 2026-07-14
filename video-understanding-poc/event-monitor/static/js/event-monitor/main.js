/* 事件监控台前端逻辑（Phase 4 · Step 26）
   选样片/上传 → POST /api/event-monitor/understand → 渲染事件窗时间线。
   纯原生 JS，无依赖；使用 event-monitor.css 的 .em-* 类。
   本版新增：分阶段进度条、身份画廊（头像+三路徽章）、关键帧检测框叠加 + 灯箱放大、设置回显。 */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  let lastPayload = null;
  let kfRegistry = [];       // 关键帧灯箱注册表：index → {image, boxes, caption}
  const esc = (s) =>
    String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );
  window.addEventListener("error", (e) => {
    try {
      setStatus("✗ 前端渲染错误：" + (e.message || e.error || "unknown"), true);
    } catch (_) {
      // 页面还没初始化时忽略。
    }
  });

  // ---- 时钟 ----
  setInterval(() => {
    const d = new Date();
    $("clock").textContent = d.toLocaleTimeString("zh-CN", { hour12: false });
  }, 1000);

  // ---- 载入样片列表 ----
  async function loadSamples() {
    try {
      const r = await fetch("/api/event-monitor/samples");
      const data = await r.json();
      const sel = $("sampleSelect");
      sel.innerHTML = "";
      (data.samples || []).forEach((s) => {
        const o = document.createElement("option");
        o.value = s.name;
        o.textContent = `${s.name} (${s.size_mb} MB)`;
        sel.appendChild(o);
      });
      const n = (data.samples || []).length;
      const cnt = $("sampleCount");
      if (cnt) cnt.textContent = n ? `${n} 个` : "";
      if (!data.samples || !data.samples.length) {
        sel.innerHTML = '<option value="">（data/samples 下没有样片）</option>';
      }
    } catch (e) {
      $("sampleSelect").innerHTML = '<option value="">加载样片失败</option>';
    }
  }

  // ============================================================
  //  分阶段进度条（后端同步返回，前端做"乐观"分段动画：跑到 ~92% 停住，
  //  真正完成后瞬间补到 100%。让用户对长任务有可见反馈，不再"转圈猜"。）
  // ============================================================
  let progTimer = null, progClock = null, progT0 = 0, progStep = 0, progPct = 0;
  function stagesFor(dryRun) {
    const s = [
      { key: "extract", label: "抽帧", weight: 12 },
      { key: "track", label: "检测 / 跟踪", weight: 22 },
      { key: "reid", label: "认人 ReID / 人脸 / 步态", weight: 30 },
      { key: "window", label: "选帧 / 分窗 / 融合", weight: 12 },
    ];
    if (!dryRun) s.push({ key: "llm", label: "多帧事件理解 gpt-4o", weight: 24 });
    return s;
  }
  function startProgress(dryRun) {
    const stages = stagesFor(dryRun);
    progStep = 0; progPct = 0; progT0 = Date.now();
    $("progress").hidden = false;
    $("progressSteps").innerHTML = stages
      .map((s, i) => `<span class="em-step" data-i="${i}">${esc(s.label)}</span>`)
      .join('<span class="em-step-sep">›</span>');
    setProgress(0, stages[0].label);
    markStep(0);
    // 计时器
    clearInterval(progClock);
    progClock = setInterval(() => {
      $("progressTimer").textContent = ((Date.now() - progT0) / 1000).toFixed(1) + "s";
    }, 100);
    // 分段推进：按各阶段 weight 累计上限，缓慢逼近但不越过当前阶段封顶。
    let cum = 0;
    const caps = stages.map((s) => (cum += s.weight));
    clearInterval(progTimer);
    progTimer = setInterval(() => {
      const cap = Math.min(92, caps[progStep] || 92);
      if (progPct < cap) {
        // 越接近封顶越慢
        progPct = Math.min(cap, progPct + Math.max(0.3, (cap - progPct) * 0.08));
        setProgress(progPct, stages[Math.min(progStep, stages.length - 1)].label);
      } else if (progStep < stages.length - 1) {
        progStep += 1;
        markStep(progStep);
      }
    }, 260);
  }
  function markStep(i) {
    document.querySelectorAll("#progressSteps .em-step").forEach((el) => {
      const idx = +el.dataset.i;
      el.classList.toggle("done", idx < i);
      el.classList.toggle("active", idx === i);
    });
  }
  function setProgress(pct, stage) {
    $("progressBar").style.width = Math.max(2, Math.min(100, pct)).toFixed(1) + "%";
    if (stage) $("progressStage").textContent = "⏳ " + stage + "…";
  }
  function finishProgress(ok) {
    clearInterval(progTimer); progTimer = null;
    // 补满 + 收尾动画
    setProgress(100, ok ? "完成" : "结束");
    $("progressStage").textContent = ok ? "✓ 处理完成" : "✗ 处理结束";
    document.querySelectorAll("#progressSteps .em-step").forEach((el) => {
      el.classList.remove("active"); el.classList.add("done");
    });
    setTimeout(() => {
      clearInterval(progClock); progClock = null;
      $("progress").hidden = true;
    }, ok ? 650 : 1400);
  }

  // ---- 提交分析 ----
  async function run() {
    const file = $("fileInput").files[0];
    const sample = $("sampleSelect").value;
    if (!file && !sample) {
      setStatus("请先选择样片或上传视频", true);
      return;
    }

    const dryRun = $("dryRun").checked;
    const fd = new FormData();
    if (file) fd.append("file", file);
    else fd.append("sample", sample);
    fd.append("fps", $("fps").value || "2");
    fd.append("max_keyframes", $("maxKeyframes").value || "8");
    if ($("objective").value.trim()) fd.append("objective", $("objective").value.trim());
    fd.append("with_face", $("withFace").checked ? "true" : "false");
    fd.append("with_gait", $("withGait").checked ? "true" : "false");
    fd.append("with_ocr", $("withOcr").checked ? "true" : "false");
    fd.append("with_objects", $("withObjects").checked ? "true" : "false");
    fd.append("dry_run", dryRun ? "true" : "false");
    // 模型/能力开关（本次生效；留空=用默认）
    if ($("faceRecBackend").value) fd.append("face_rec_backend", $("faceRecBackend").value);
    if ($("faceSuperres").value) fd.append("face_superres", $("faceSuperres").value);
    if ($("reidBackend").value) fd.append("reid_backend", $("reidBackend").value);
    if ($("trackBackend").value) fd.append("track_backend", $("trackBackend").value);
    fd.append("face_3d_cue", $("face3d").checked ? "true" : "false");
    fd.append("reid_consistency_enabled", $("reidConsistency").checked ? "true" : "false");
    if ($("reidTopK").value) fd.append("reid_decision_top_k", $("reidTopK").value);
    if ($("reidVoteThresh").value) fd.append("reid_vote_score_thresh", $("reidVoteThresh").value);
    if ($("reidConsistencyRatio").value) fd.append("reid_consistency_ratio", $("reidConsistencyRatio").value);
    if ($("reidTop1Margin").value) fd.append("reid_top1_margin", $("reidTop1Margin").value);
    if ($("maxWindowSeconds").value) fd.append("max_window_seconds", $("maxWindowSeconds").value);
    if ($("stitchThresh").value) fd.append("stitch_thresh", $("stitchThresh").value);

    $("btnRun").disabled = true;
    $("empty").style.display = "none";
    $("overall").hidden = true;
    $("cfgSummary").hidden = true;
    $("timings").hidden = true;
    $("resultTools").hidden = true;
    $("timeline").innerHTML = "";
    $("tracks").innerHTML = "";
    $("meta").innerHTML = "";
    startProgress(dryRun);
    const llm = dryRun ? "（dry-run，不调 LLM）" : "（含 gpt-4o，约 1 分钟）";
    setStatus("⏳ 处理中… " + llm);

    const t0 = Date.now();
    try {
      const r = await fetch("/api/event-monitor/understand", { method: "POST", body: fd });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${r.status}`);
      }
      const text = await r.text();
      const data = JSON.parse(text);
      finishProgress(true);
      render(data);
      setStatus(`✓ 完成，用时 ${((Date.now() - t0) / 1000).toFixed(1)}s`);
    } catch (e) {
      finishProgress(false);
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

  // ---- 颜色：按 subject_id 稳定映射一个色相（关键帧框 + 徽章配色一致）----
  function subjectHue(sid) {
    if (sid == null) return 205;
    let h = 0;
    for (const c of String(sid)) h = (h * 31 + c.charCodeAt(0)) % 360;
    return h;
  }

  // ---- 渲染结果 ----
  function render(data) {
    lastPayload = data;
    $("resultTools").hidden = false;
    $("jsonView").hidden = true;
    $("btnToggleJson").textContent = "查看原始 JSON";
    $("btnSendLlm").hidden = !data.dry_run;
    renderOverall(data.overall);
    renderMeta(data);
    renderConfigSummary(data);
    renderTimings(data);
    renderSubjectGallery(data);

    // 事件窗时间线
    kfRegistry = [];
    $("timeline").innerHTML = (data.windows || []).map(renderWindow).join("");
  }

  function renderMeta(data) {
    const cu = data.config_used || {};
    $("meta").innerHTML =
      `视频 <b>${esc(baseName(data.video))}</b> · ${data.frames_total} 帧 @ ${data.fps}fps · ` +
      `${(data.windows || []).length} 个事件窗 · Tracker <b>${esc(data.tracker_backend || cu.track_backend || "botsort_reid")}</b> · ` +
      `ReID <b>${esc(data.reid_backend)}</b>(${data.reid_dim}d) · ` +
      `模型 <b>${esc(data.model)}</b>${data.dry_run ? " · <b>dry-run</b>" : ""} · ${data.elapsed_seconds}s`;
  }

  // ---- 本次实际生效的能力/模型（关掉设置抽屉后仍能看到"这次用了什么"）----
  function renderConfigSummary(data) {
    const cu = data.config_used || {};
    const chips = [];
    const on = (b) => (b ? "on" : "off");
    chips.push(`人脸 <b>${on(cu.with_face)}</b>` +
      (cu.with_face ? `（${esc(cu.face_rec_backend || "adaface")}` +
        `${cu.face_superres && cu.face_superres !== "off" ? "+超分" : ""}` +
        `${cu.face_3d_cue ? "+3D" : ""}）` : ""));
    chips.push(`步态 <b>${on(cu.with_gait)}</b>`);
    chips.push(`OCR <b>${cu.with_ocr ? esc(data.ocr_backend || "on") : "off"}</b>`);
    chips.push(`物体 <b>${on(cu.with_objects)}</b>`);
    chips.push(`ReID <b>${esc(cu.reid_backend || "auto")}</b>` +
      `${cu.reid_consistency_enabled ? `（top-${cu.reid_decision_top_k} 一致性）` : "（top-1）"}`);
    if (data.gait_error) chips.push(`<span class="warn">步态告警: ${esc(data.gait_error)}</span>`);
    if (data.ocr_error) chips.push(`<span class="warn">OCR告警: ${esc(data.ocr_error)}</span>`);
    $("cfgSummary").hidden = false;
    $("cfgSummary").innerHTML = "本次生效： " + chips.map((c) => `<span class="em-cfgchip">${c}</span>`).join(" ");
  }

  // ---- 实测各阶段耗时（后端 stage_timings；从大到小条形图，demo 讲解用）----
  const STAGE_CN = {
    extract_frames: "① 抽帧",
    detect_track: "② YOLO 检测 + 跟踪",
    gait_collect: "· 步态采集（Pose+Seg 逐帧）",
    reid_identify: "③ 人形 ReID 认人",
    face: "· 人脸分支（检测+对齐+AdaFace）",
    gait_embed: "· 步态向量提取",
    merge_fusion_thumb: "⑥ 三路融合 + 头像",
    windows_select: "④⑤ 选帧 / 分窗",
    windows_llm: "⑦ gpt-4o 事件理解",
    overall_summary: "· 整段总结",
  };
  function renderTimings(data) {
    const st = data.stage_timings || {};
    const rows = Object.entries(st)
      .map(([k, v]) => ({ k, v: +v }))
      .filter((r) => isFinite(r.v) && r.v >= 0)
      .sort((a, b) => b.v - a.v);
    if (!rows.length) { $("timings").hidden = true; return; }
    const max = Math.max(...rows.map((r) => r.v), 0.01);
    const total = rows.reduce((s, r) => s + r.v, 0);
    const bars = rows
      .map((r, i) => {
        const pct = (r.v / max) * 100;
        const share = total > 0 ? ((r.v / total) * 100).toFixed(0) : "0";
        return (
          `<div class="em-tbar ${i === 0 ? "top" : ""}">` +
          `<span class="em-tbar-label">${esc(STAGE_CN[r.k] || r.k)}</span>` +
          `<span class="em-tbar-track"><span class="em-tbar-fill" style="width:${pct.toFixed(1)}%"></span></span>` +
          `<span class="em-tbar-val">${r.v.toFixed(1)}s · ${share}%</span></div>`
        );
      })
      .join("");
    $("timings").hidden = false;
    $("timings").innerHTML =
      `<div class="em-timings-head">⏱ 本次各阶段实测耗时` +
      `<span class="tot">总 ${esc(data.elapsed_seconds)}s · 本地 CPU</span></div>` + bars;
  }

  // ---- 三路身份徽章（脸/形/步 是否命中 + 融合置信）----
  function routeBadges(rec) {
    const face = rec.face || null;
    const gait = rec.gait || null;
    const fused = rec.fused || null;
    const b = [];
    // 人脸
    if (face) {
      const q = face.quality || "?";
      const good = face.matched || q === "clear";
      b.push(`<span class="em-rb ${good ? "hit" : "weak"}" title="人脸 · 质量${esc(q)}${face.match_score != null ? " · 相似" + (+face.match_score).toFixed(2) : ""}">脸</span>`);
    } else {
      b.push(`<span class="em-rb off" title="无脸 → 退人形/步态">脸</span>`);
    }
    // 人形 ReID
    if (rec.score != null) {
      b.push(`<span class="em-rb hit" title="人形 ReID 相似 ${(+rec.score).toFixed(2)}">形</span>`);
    } else {
      b.push(`<span class="em-rb off" title="无人形分">形</span>`);
    }
    // 步态
    if (gait && gait.score != null) {
      b.push(`<span class="em-rb ${gait.decision === "hit" ? "hit" : "weak"}" title="步态 · ${esc(gait.decision || "")} · 相似 ${(+gait.score).toFixed(2)} · ${gait.frames || 0}帧">步</span>`);
    } else {
      b.push(`<span class="em-rb off" title="步态未启用/帧不足">步</span>`);
    }
    let conf = "";
    if (fused && fused.confidence != null) {
      const lv = fused.resolved ? "hit" : "weak";
      const primaryCn = { face: "脸", body: "形", gait: "步" }[fused.primary] || "—";
      conf = `<span class="em-conf ${lv}" title="三路融合置信 · 主导线索 ${esc(primaryCn)}${fused.agreed ? " · 多路印证" : ""}">融合 ${(fused.confidence * 100).toFixed(0)}%${fused.agreed ? " ✓" : ""}</span>`;
    }
    return `<span class="em-routes">${b.join("")}${conf}</span>`;
  }

  // ---- 身份画廊：按 subject 聚合，头像 + 轨迹数 + 三路徽章 + 回头客/本地/拆分标记 ----
  function renderSubjectGallery(data) {
    const tracks = data.tracks || {};
    const subjMap = {};
    Object.entries(tracks).forEach(([tid, idn]) => {
      const sid = idn.subject_id == null ? "?" : idn.subject_id;
      const g = (subjMap[sid] = subjMap[sid] || {
        tracks: [], reused: false, local: false, split: false,
        score: 0, best: null, bestScore: -1,
      });
      g.tracks.push(tid);
      if (idn.reused) g.reused = true;
      if (idn.local_subject) g.local = true;
      if (idn.subject_conflict_split) g.split = true;
      g.score = Math.max(g.score, idn.score || 0);
      // 代表 track：分最高且有头像者
      const s = idn.score || 0;
      if (idn.thumb && s >= g.bestScore) { g.bestScore = s; g.best = idn; }
      if (!g.best && idn.thumb) g.best = idn;
      if (!g.best) g.best = g.best || idn;
    });

    const cards = Object.entries(subjMap).map(([sid, v]) => {
      const rec = v.best || {};
      const hue = subjectHue(sid === "?" ? null : sid);
      const title = sid === "?" ? "未入库 · 待定身份" : `主体 #${esc(sid)}`;
      const thumb = rec.thumb
        ? `<img src="${rec.thumb}" alt="${esc(title)}" loading="lazy"/>`
        : `<span class="em-avatar-ph">?</span>`;
      const flags = [];
      if (v.reused) flags.push('<span class="reused">♻ 回头客</span>');
      if (v.local) flags.push("本地subject");
      if (v.split) flags.push("时间冲突拆分");
      return (
        `<div class="em-subcard" style="--hue:${hue}">` +
        `<div class="em-avatar">${thumb}</div>` +
        `<div class="em-subinfo">` +
        `<div class="em-subtitle">${esc(title)}</div>` +
        `<div class="em-submeta">${v.tracks.length} 条轨迹` +
        (flags.length ? " · " + flags.join(" · ") : "") + `</div>` +
        routeBadges(rec) +
        `</div></div>`
      );
    });

    $("tracks").innerHTML = cards.length
      ? `<div class="em-gallery-head">👥 身份画廊（本段共 ${cards.length} 个主体）</div>` +
        `<div class="em-gallery">${cards.join("")}</div>`
      : "";
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

  // ---- 关键帧 + 检测框叠加（用 spatial_grounding 的 bbox_norm 画 subject 标注框）----
  function boxesForFrame(gFrame) {
    if (!gFrame || !Array.isArray(gFrame.objects)) return [];
    return gFrame.objects
      .map((o) => {
        const bn = o.bbox_norm || [];
        if (bn.length < 4) return null;
        const [x1, y1, x2, y2] = bn;
        return {
          x: x1 * 100, y: y1 * 100,
          w: Math.max(0, (x2 - x1)) * 100, h: Math.max(0, (y2 - y1)) * 100,
          label: o.subject_id != null ? `#${o.subject_id}` : `t${o.track_id}`,
          hue: subjectHue(o.subject_id),
        };
      })
      .filter(Boolean);
  }
  function boxesHtml(boxes) {
    return boxes
      .map(
        (b) =>
          `<div class="em-box" style="left:${b.x.toFixed(2)}%;top:${b.y.toFixed(2)}%;` +
          `width:${b.w.toFixed(2)}%;height:${b.h.toFixed(2)}%;--hue:${b.hue}">` +
          `<span class="em-box-tag">${esc(b.label)}</span></div>`
      )
      .join("");
  }

  function renderWindow(w) {
    const ev = w.event || null;
    const level = (ev && ev.alert_level) || "normal";
    const gFrames = (w.spatial_grounding && w.spatial_grounding.frames) || [];
    const frames = (w.keyframes || [])
      .map((k, i) => {
        const boxes = boxesForFrame(gFrames[i]);
        const cap = `关键帧 · ${k.timestamp}` + (boxes.length ? ` · ${boxes.length} 个目标` : "");
        const kfi = kfRegistry.push({ image: k.image, boxes, caption: cap }) - 1;
        return (
          `<div class="em-frame" data-kf="${kfi}" title="点击放大">` +
          `<img src="${k.image}" loading="lazy"/>` +
          `<div class="em-boxes">${boxesHtml(boxes)}</div>` +
          `<span class="em-frame-ts">${esc(k.timestamp)}</span></div>`
        );
      })
      .join("");
    const grounding = renderGrounding(w.spatial_grounding);

    const people = (w.people || [])
      .map((p) => {
        const label =
          p.subject_id != null ? `主体#${p.subject_id}` : `track ${p.track_id}`;
        const hue = subjectHue(p.subject_id);
        const thumb = thumbForTrack(p);
        const avatar = thumb
          ? `<img class="em-pavatar" src="${thumb}" loading="lazy"/>`
          : "";
        const cues = [];
        if (p.reid && p.reid.score != null) cues.push(`人形ReID ${(+p.reid.score).toFixed(2)}`);
        cues.push(p.face ? "有脸" : "无脸→人形为准");
        if (p.reused) cues.push("♻回头客");
        if (p.local_subject) cues.push("本视频本地subject");
        if (p.subject_conflict_split) cues.push("时间冲突已拆分");
        return `<div class="em-person" style="--hue:${hue}">${avatar}` +
          `<span class="pl">${esc(label)}</span> <span class="pc">${esc(cues.join(" · "))}</span></div>`;
      })
      .join("");

    const scene = w.scene_context
      ? `<details class="em-aux"><summary>🔤 场景文字 OCR</summary><pre>${esc(w.scene_context)}</pre></details>`
      : "";
    const objs = w.object_context
      ? `<details class="em-aux"><summary>📦 物体 / 包裹</summary><pre>${esc(w.object_context)}</pre></details>`
      : "";

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
        (events ? `<div class="em-events">${events}</div>` : "") +
        scene + objs + grounding;
    } else {
      // dry-run：无 LLM 事件，展示将要喂模型的关键帧 + 身份
      body =
        `<div class="em-summary" style="color:var(--muted)">（dry-run：未调用 LLM。以下为将喂给模型的关键帧与身份。）</div>` +
        (frames ? `<div class="em-frames">${frames}</div>` : "") +
        scene + objs + grounding;
    }

    return (
      `<div class="em-window ${esc(level)}">` +
      `<div class="em-window-head">` +
      `<span class="em-time">⏱ ${esc(w.time_range[0])} ~ ${esc(w.time_range[1])}</span>` +
      `<span class="em-badge ${esc(level)}">${esc(level)}</span>` +
      `<span class="em-time">${w.frame_count || 0} 帧 → ${(w.keyframe_indices || []).length} 关键帧</span>` +
      `</div>` +
      body +
      (people ? `<div class="em-people">${people}</div>` : "") +
      `</div>`
    );
  }

  // 从 lastPayload.tracks 里按代表 track_id 取头像（people 里只带 track_id）
  function thumbForTrack(p) {
    const tracks = (lastPayload && lastPayload.tracks) || {};
    const ids = (p.source_track_ids && p.source_track_ids.length ? p.source_track_ids : [p.track_id]) || [];
    for (const t of ids) {
      const rec = tracks[String(t)];
      if (rec && rec.thumb) return rec.thumb;
    }
    return null;
  }

  function baseName(p) {
    return String(p || "").split(/[\\/]/).pop();
  }

  function renderGrounding(g) {
    if (!g || !Array.isArray(g.frames)) return "";
    const frameRows = g.frames
      .slice(0, 8)
      .map((fr) => {
        const objs = fr.objects || [];
        const sample = objs
          .slice(0, 4)
          .map((o) => `${esc(o.label)} c=${esc(JSON.stringify(o.center_norm || []))}`)
          .join("；");
        return `<div class="em-event"><span class="et">frame ${esc(fr.frame_index)} @ ${esc(fr.timestamp)}</span>` +
          `<span class="ea">${objs.length} objects${sample ? " · " + sample : ""}</span></div>`;
      })
      .join("");
    const trajRows = (g.trajectories || [])
      .slice(0, 8)
      .map((t) => `<div class="em-event"><span class="es">${esc(t.label)}</span>` +
        `<span class="ea">${esc(t.direction)} · path ${esc(JSON.stringify(t.path_sample || []))}</span></div>`)
      .join("");
    return `<details class="em-aux"><summary>📍 空间 grounding：关键帧坐标 + 轨迹摘要</summary>${frameRows}${trajRows}</details>`;
  }

  // ============================================================
  //  关键帧灯箱（点击关键帧放大，保留检测框标注）
  // ============================================================
  function openLightbox(kfi) {
    const kf = kfRegistry[kfi];
    if (!kf) return;
    $("lightboxStage").innerHTML =
      `<img src="${kf.image}"/><div class="em-boxes">${boxesHtml(kf.boxes)}</div>`;
    $("lightboxCap").textContent = kf.caption || "";
    $("lightbox").hidden = false;
  }
  function closeLightbox() {
    $("lightbox").hidden = true;
    $("lightboxStage").innerHTML = "";
  }

  // ---- 原始 JSON：下载 / 查看（剥掉关键帧 base64 大图 + 头像，只留结构化数据）----
  function cleanedPayload() {
    if (!lastPayload) return null;
    const obj = JSON.parse(JSON.stringify(lastPayload));
    (obj.windows || []).forEach((w) => {
      (w.keyframes || []).forEach((k) => {
        if (k.image) k.image = "<data-uri omitted>";
      });
    });
    Object.values(obj.tracks || {}).forEach((t) => {
      if (t && t.thumb) t.thumb = "<data-uri omitted>";
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
    a.download = `event-monitor_${stem}${obj.dry_run ? "_dryrun" : ""}.json`;
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

  async function sendDryRunToLlm() {
    if (!lastPayload || !lastPayload.dry_run) return;
    $("btnSendLlm").disabled = true;
    const t0 = Date.now();
    startProgress(false);
    setStatus("⏳ 正在复用 dry-run 的关键帧和身份上下文调用大模型…");
    try {
      const r = await fetch("/api/event-monitor/complete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          payload: lastPayload,
          objective: $("objective").value.trim() || null,
        }),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${r.status}`);
      }
      const data = await r.json();
      finishProgress(true);
      render(data);
      setStatus(`✓ 大模型事件理解完成，用时 ${((Date.now() - t0) / 1000).toFixed(1)}s`);
    } catch (e) {
      finishProgress(false);
      setStatus("✗ 调用大模型失败：" + e.message, true);
    } finally {
      $("btnSendLlm").disabled = false;
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
  $("btnSendLlm").addEventListener("click", sendDryRunToLlm);
  $("btnDownloadJson").addEventListener("click", downloadJson);
  $("btnToggleJson").addEventListener("click", toggleJson);
  $("btnSettings").addEventListener("click", openSettings);
  $("btnCloseSettings").addEventListener("click", closeSettings);
  $("btnApplySettings").addEventListener("click", closeSettings);
  $("settingsOverlay").addEventListener("click", closeSettings);
  // 关键帧点击放大（事件委托）
  $("timeline").addEventListener("click", (e) => {
    const fr = e.target.closest(".em-frame");
    if (fr && fr.dataset.kf != null) openLightbox(+fr.dataset.kf);
  });
  $("lightboxClose").addEventListener("click", closeLightbox);
  $("lightbox").addEventListener("click", (e) => {
    if (e.target === $("lightbox")) closeLightbox();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { closeLightbox(); closeSettings(); }
  });

  // ---- 后端在线状态（顶栏绿点）----
  async function checkBackend() {
    const el = $("backendStatus");
    if (!el) return;
    try {
      const r = await fetch("/api/event-monitor/samples", { method: "GET" });
      if (r.ok) { el.className = "em-svc online"; el.lastChild.textContent = "服务在线"; }
      else throw new Error();
    } catch (_) {
      el.className = "em-svc offline"; el.lastChild.textContent = "服务离线";
    }
  }

  // ---- 拖拽上传：拖入高亮 + 选中后显示文件名 ----
  function setDropFile(name) {
    const main = $("dropMain");
    const zone = $("dropZone");
    if (name) {
      if (main) main.textContent = "📄 " + name;
      if (zone) zone.classList.add("has-file");
      // 选了文件就清空样片下拉的选择意图（上传优先，后端逻辑一致）
    } else {
      if (main) main.textContent = "拖拽视频到此，或点击选择";
      if (zone) zone.classList.remove("has-file");
    }
  }
  (function wireDropzone() {
    const zone = $("dropZone");
    const input = $("fileInput");
    if (!zone || !input) return;
    input.addEventListener("change", () => setDropFile(input.files[0] ? input.files[0].name : ""));
    ["dragenter", "dragover"].forEach((ev) =>
      zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add("dragover"); })
    );
    ["dragleave", "drop"].forEach((ev) =>
      zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove("dragover"); })
    );
    zone.addEventListener("drop", (e) => {
      const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) { input.files = e.dataTransfer.files; setDropFile(f.name); }
    });
  })();

  checkBackend();
  loadSamples();
})();
