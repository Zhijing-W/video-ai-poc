import { reportLanguage, t, uiLocale } from "./i18n.js";
import { $, esc } from "./utils.js";

function appendIfValue(formData, key, value) {
  if (value) formData.append(key, value);
}

export function collectAnalysisRequest() {
  const file = $("fileInput").files[0];
  const sample = $("sampleSelect").value;
  const dryRun = $("dryRun").checked;
  const objective = $("objective").value.trim();
  const formData = new FormData();

  if (file) formData.append("file", file);
  else if (sample) formData.append("sample", sample);

  formData.append("fps", $("fps").value || "2");
  formData.append("max_keyframes", $("maxKeyframes").value || "8");
  appendIfValue(formData, "objective", objective);
  formData.append("with_body", $("withBody").checked ? "true" : "false");
  formData.append("with_face", $("withFace").checked ? "true" : "false");
  formData.append("with_gait", $("withGait").checked ? "true" : "false");
  formData.append("with_ocr", $("withOcr").checked ? "true" : "false");
  formData.append("with_objects", $("withObjects").checked ? "true" : "false");
  formData.append("dry_run", dryRun ? "true" : "false");
  formData.append("language", reportLanguage());
  appendIfValue(formData, "analysis_model", getAnalysisModelValue());
  appendIfValue(formData, "face_rec_backend", $("faceRecBackend").value);
  appendIfValue(formData, "face_superres", $("faceSuperres").value);
  if (isCodeFormerSelected()) {
    appendIfValue(formData, "face_codeformer_fidelity", $("faceCodeformerFidelity").value);
  }
  appendIfValue(formData, "reid_backend", $("reidBackend").value);
  appendIfValue(formData, "track_backend", $("trackBackend").value);
  formData.append("face_3d_cue", $("face3d").checked ? "true" : "false");
  formData.append("reid_consistency_enabled", $("reidConsistency").checked ? "true" : "false");
  appendIfValue(formData, "reid_decision_top_k", $("reidTopK").value);
  appendIfValue(formData, "reid_vote_score_thresh", $("reidVoteThresh").value);
  appendIfValue(formData, "reid_consistency_ratio", $("reidConsistencyRatio").value);
  appendIfValue(formData, "reid_top1_margin", $("reidTop1Margin").value);
  appendIfValue(formData, "max_window_seconds", $("maxWindowSeconds").value);
  appendIfValue(formData, "stitch_thresh", $("stitchThresh").value);

  return { file, sample, dryRun, objective, analysisModel: getAnalysisModelValue(), formData };
}

export function getObjectiveValue() {
  return $("objective").value.trim();
}

export function getAnalysisModelValue() {
  return $("analysisModel")?.value || "auto";
}

export function getChatModelValue() {
  return $("chatModel")?.value || "auto";
}

function replaceModelOptions(select, options, defaultAlias) {
  if (!select) return;
  const available = (options || []).filter((option) => option.available !== false);
  select.replaceChildren();
  available.forEach((option) => {
    const item = new Option(option.label || option.alias, option.alias);
    item.title = option.description || "";
    select.add(item);
  });
  select.value = available.some((option) => option.alias === defaultAlias)
    ? defaultAlias
    : (available[0]?.alias || "auto");
  select.disabled = !available.length;
}

const modelPickerState = new Map();
let modelPickerDocumentEventsWired = false;

function providerMark(provider) {
  if (provider === "microsoft") {
    return `<svg class="em-provider-mark" viewBox="0 0 24 24" aria-hidden="true">
      <path fill="currentColor" d="M3 3h8v8H3zm10 0h8v8h-8zM3 13h8v8H3zm10 0h8v8h-8z"/>
    </svg>`;
  }
  if (provider === "deepseek") {
    return `<svg class="em-provider-mark" viewBox="0 0 24 24" aria-hidden="true">
      <path fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round" d="M3 7.5 8.5 4l7 2 5.5 5-4 7H8l-5-4.5 4-3z"/>
      <path fill="currentColor" d="M10 10h2v2h-2zm4 0h2v2h-2z"/>
    </svg>`;
  }
  if (provider === "moonshot") {
    return `<svg class="em-provider-mark" viewBox="0 0 24 24" aria-hidden="true">
      <path fill="none" stroke="currentColor" stroke-width="2" d="M15.8 3.5a8.5 8.5 0 1 0 4.7 14.8A7.2 7.2 0 0 1 15.8 3.5Z"/>
    </svg>`;
  }
  if (provider !== "openai") return "";
  return `<svg class="em-provider-mark" viewBox="0 0 24 24" aria-hidden="true">
    <path fill="currentColor" d="M12 2.5a4.5 4.5 0 0 1 4.1 2.65 4.5 4.5 0 0 1 4.6 3.85 4.5 4.5 0 0 1-2.1 4.4 4.5 4.5 0 0 1-1.3 5.85 4.5 4.5 0 0 1-4.9-.3 4.5 4.5 0 0 1-5.65-2.1 4.5 4.5 0 0 1-3.15-3.75 4.5 4.5 0 0 1 2.1-4.35A4.5 4.5 0 0 1 7.1 3.1 4.5 4.5 0 0 1 12 2.5Zm0 2a2.5 2.5 0 0 0-2.45 2l-.03.18 3.73 2.15v4.3l-3.75 2.16A2.5 2.5 0 0 0 12 17.5a2.5 2.5 0 0 0 2.45-2l.03-.18-3.73-2.15v-4.3l3.75-2.16A2.5 2.5 0 0 0 12 4.5Z"/>
  </svg>`;
}

function recentModels(task) {
  try {
    return JSON.parse(localStorage.getItem(`event-monitor:model-recent:${task}`) || "[]");
  } catch (_) {
    return [];
  }
}

function rememberModel(task, alias) {
  if (alias === "auto") return;
  const models = [alias, ...recentModels(task).filter((item) => item !== alias)].slice(0, 3);
  try {
    localStorage.setItem(`event-monitor:model-recent:${task}`, JSON.stringify(models));
  } catch (_) {
    // Private browsing or strict storage settings must not affect model selection.
  }
}

function pickerOptions(state) {
  const query = (state.query || "").trim().toLowerCase();
  return state.options.filter((option) => {
    const haystack = [option.label, option.model, option.description, option.group]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return !query || haystack.includes(query);
  });
}

function orderedPickerOptions(state) {
  const options = pickerOptions(state);
  const recents = new Set(recentModels(state.task));
  return [
    ...options.filter((option) => option.alias === "auto"),
    ...options.filter((option) => option.alias !== "auto" && recents.has(option.alias)),
    ...options.filter((option) => option.alias !== "auto" && !recents.has(option.alias)),
  ];
}

function renderPickerOptions(state) {
  const root = $(state.rootId);
  if (!root) return;
  const container = root.querySelector(".em-model-options");
  const triggerValue = $(`${state.task}ModelValue`);
  if (!container || !triggerValue) return;
  const available = orderedPickerOptions(state);
  const selected = state.options.find((option) => option.alias === state.value);
  triggerValue.innerHTML = selected
    ? `${providerMark(selected.provider)}<span>${esc(selected.label)}</span>`
    : esc(t("panel.ai_loading"));

  const recents = new Set(recentModels(state.task));
  const groups = new Map();
  const addToGroup = (name, option) => {
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push(option);
  };
  available.filter((option) => option.alias === "auto").forEach((option) => {
    addToGroup("", option);
  });
  available.filter((option) => recents.has(option.alias)).forEach((option) => {
    addToGroup(t("panel.model_recent"), option);
  });
  available.filter((option) => option.alias !== "auto" && !recents.has(option.alias)).forEach((option) => {
    addToGroup(option.group || t("panel.model_versatile"), option);
  });
  let index = 0;
  container.innerHTML = [...groups].map(([group, options]) => {
    const heading = group ? `<div class="em-model-group-title">${esc(group)}</div>` : "";
    const choices = options.map((option) => {
      const optionIndex = index++;
      const selectedOption = option.alias === state.value;
      const deployment = option.alias === "auto"
        ? option.description
        : `${option.description || ""} · ${option.context || ""} · ${option.performance || ""} · ${option.alias}`;
      const preview = option.preview ? `<span class="em-model-preview">Preview</span>` : "";
      return `<button id="${state.task}ModelOption${optionIndex}" class="em-model-option${optionIndex === state.activeIndex ? " is-active" : ""}" type="button" role="option" aria-selected="${selectedOption}" data-alias="${esc(option.alias)}" title="${esc(deployment)}">
        ${providerMark(option.provider)}
        <span><span class="em-model-primary">${esc(option.label)}${preview}</span><span class="em-model-secondary">${esc(deployment)}</span></span>
        <span class="em-model-check">${selectedOption ? "✓" : `<span class="em-model-info" aria-label="${esc(t("panel.model_details"))}">ⓘ</span>`}</span>
      </button>`;
    }).join("");
    return `<section class="em-model-group">${heading}${choices}</section>`;
  }).join("") || `<div class="em-model-group-title">${esc(t("panel.model_no_results"))}</div>`;
  container.querySelectorAll("[data-alias]").forEach((button) => {
    button.addEventListener("click", () => selectPickerOption(state, button.dataset.alias));
  });
}

function closePicker(state, restoreFocus = false) {
  const popover = $(`${state.task}ModelPopover`);
  const trigger = $(`${state.task}ModelTrigger`);
  if (!popover || popover.hidden) return;
  popover.hidden = true;
  trigger?.setAttribute("aria-expanded", "false");
  if (restoreFocus) trigger?.focus();
}

function selectPickerOption(state, alias) {
  if (!state.options.some((option) => option.alias === alias)) return;
  state.value = alias;
  const input = $(`${state.task}Model`);
  if (input) input.value = alias;
  rememberModel(state.task, alias);
  renderPickerOptions(state);
  closePicker(state, true);
}

function movePickerActive(state, direction) {
  const options = orderedPickerOptions(state);
  if (!options.length) return;
  state.activeIndex = (state.activeIndex + direction + options.length) % options.length;
  renderPickerOptions(state);
  $(`${state.task}ModelOption${state.activeIndex}`)?.focus();
}

function wireModelPicker(state) {
  const root = $(state.rootId);
  const trigger = $(`${state.task}ModelTrigger`);
  const popover = $(`${state.task}ModelPopover`);
  if (!root || !trigger || !popover || root.dataset.wired) return;
  root.dataset.wired = "true";
  const search = root.querySelector(".em-model-search");
  trigger.addEventListener("click", () => {
    const opening = popover.hidden;
    modelPickerState.forEach((other) => {
      if (other !== state) closePicker(other);
    });
    popover.hidden = !opening;
    trigger.setAttribute("aria-expanded", String(opening));
    if (opening) {
      state.activeIndex = 0;
      renderPickerOptions(state);
      search?.focus();
    }
  });
  search?.addEventListener("input", () => {
    state.query = search.value;
    state.activeIndex = 0;
    renderPickerOptions(state);
  });
  root.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      movePickerActive(state, 1);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      movePickerActive(state, -1);
    } else if (event.key === "Enter") {
      const option = orderedPickerOptions(state)[state.activeIndex];
      if (option && !popover.hidden) {
        event.preventDefault();
        selectPickerOption(state, option.alias);
      }
    } else if (event.key === "Escape") {
      event.preventDefault();
      closePicker(state, true);
    }
  });
  if (!modelPickerDocumentEventsWired) {
    modelPickerDocumentEventsWired = true;
    document.addEventListener("pointerdown", (event) => {
      modelPickerState.forEach((item) => {
        const picker = $(item.rootId);
        if (picker && !picker.contains(event.target)) closePicker(item);
      });
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") modelPickerState.forEach((item) => closePicker(item));
    });
  }
}

function renderModelPicker(task, options, defaultAlias) {
  const rootId = `${task}ModelPicker`;
  if (!$(rootId)) return false;
  const available = (options || []).filter((option) => option.available !== false);
  const prior = modelPickerState.get(task);
  const state = prior || {
    task,
    rootId,
    query: "",
    activeIndex: 0,
    value: "auto",
    options: [],
  };
  state.options = available;
  state.value = available.some((option) => option.alias === state.value)
    ? state.value
    : (available.some((option) => option.alias === defaultAlias)
      ? defaultAlias
      : (available[0]?.alias || "auto"));
  modelPickerState.set(task, state);
  const input = $(`${task}Model`);
  if (input) input.value = state.value;
  wireModelPicker(state);
  renderPickerOptions(state);
  return true;
}

export function renderLlmModels(catalog = {}) {
  const pickerRendered = renderModelPicker(
    "analysis", catalog.analysis, catalog.defaults?.analysis
  ) && renderModelPicker("chat", catalog.chat, catalog.defaults?.chat);
  if (!pickerRendered) {
    replaceModelOptions($("analysisModel"), catalog.analysis, catalog.defaults?.analysis);
    replaceModelOptions($("chatModel"), catalog.chat, catalog.defaults?.chat);
  }
  const badge = $("llmAuthBadge");
  if (badge) {
    badge.textContent = catalog.auth === "managed_identity"
      ? t("panel.auth_managed_identity")
      : t("panel.auth_api_key");
    badge.className = `em-model-auth ${catalog.auth === "managed_identity" ? "managed" : ""}`;
  }
  const hint = $("llmModelHint");
  if (hint) {
    hint.textContent = catalog.analysis?.length
      ? t("panel.routing_hint")
      : t("panel.routing_unavailable");
  }
}

function isCodeFormerSelected() {
  const select = $("faceSuperres");
  return select.value === "codeformer"
    || (!select.value && select.dataset.defaultBackend === "codeformer");
}

function updateCodeFormerFidelityVisibility() {
  const field = $("faceCodeformerFidelityField");
  if (field) field.hidden = !isCodeFormerSelected();
}

export function renderSuperresBackends(catalog = {}) {
  const select = $("faceSuperres");
  if (!select) return;
  const names = Array.isArray(catalog.backends) ? catalog.backends : [];
  const allowed = [...new Set(["off", "gfpgan", "codeformer", "realesrgan_x2plus", ...names])];
  const labels = {
    off: t("settings.face_superres_off"),
    gfpgan: "GFP-GAN",
    codeformer: "CodeFormer",
    realesrgan_x2plus: "Real-ESRGAN x2plus",
  };
  const defaultBackend = catalog.default || "off";
  select.dataset.defaultBackend = defaultBackend;
  select.replaceChildren();
  const defaultLabel = uiLocale() === "en"
    ? `${labels[defaultBackend] || defaultBackend} (default)`
    : `${labels[defaultBackend] || defaultBackend}（默认）`;
  select.add(new Option(defaultLabel, ""));
  allowed
    .filter((name) => name !== defaultBackend)
    .forEach((name) => select.add(new Option(labels[name] || name, name)));

  const fidelity = catalog.metadata?.codeformer?.fidelity_default;
  const input = $("faceCodeformerFidelity");
  if (input && Number.isFinite(Number(fidelity))) input.value = String(fidelity);
  updateCodeFormerFidelityVisibility();
}

export function renderReidBackends(catalog = {}) {
  const select = $("reidBackend");
  if (!select) return;
  const catalogLoaded = Array.isArray(catalog.backends);
  const names = catalogLoaded ? catalog.backends : [];
  const productBackends = ["differ", "clipreid", "siglip2", "osnet"];
  const registered = new Set(names);
  const allowed = catalogLoaded
    ? productBackends.filter((name) => registered.has(name))
    : [];
  const fallbackLabels = {
    osnet: "OSNet-AIN MSMT17",
    clipreid: "CLIP-ReID ViT-B/16",
    siglip2: "SigLIP2 Person ReID",
    differ: "DIFFER EVA02-L",
  };
  const metadata = catalog.metadata || {};
  const labelFor = (name) => metadata[name]?.label || fallbackLabels[name] || name;
  const configuredDefault = allowed.includes(catalog.default) ? catalog.default : "";
  select.replaceChildren();
  const defaultLabel = configuredDefault
    ? `${labelFor(configuredDefault)}${
      uiLocale() === "en" ? " (default · accuracy first)" : "（默认 · 精度优先）"
    }`
    : (
      catalogLoaded
        ? (uiLocale() === "en" ? "Use server default" : "使用服务端默认")
        : (uiLocale() === "en" ? "Use server default (backend catalog unavailable)" : "使用服务端默认（后端目录不可用）")
    );
  select.add(new Option(defaultLabel, ""));
  allowed
    .filter((name) => name !== configuredDefault)
    .forEach((name) => select.add(new Option(labelFor(name), name)));
  select.value = "";
  select.dataset.defaultBackend = configuredDefault;
  select.disabled = !catalogLoaded;
}

export function wireSuperresSettings() {
  const select = $("faceSuperres");
  if (select) select.addEventListener("change", updateCodeFormerFidelityVisibility);
  updateCodeFormerFidelityVisibility();
}

export function openSettings() {
  $("settingsDrawer").hidden = false;
  $("settingsOverlay").hidden = false;
}

export function closeSettings() {
  $("settingsDrawer").hidden = true;
  $("settingsOverlay").hidden = true;
}

export function setDropFile(name) {
  const main = $("dropMain");
  const zone = $("dropZone");

  if (name) {
    if (main) main.textContent = `📄 ${name}`;
    if (zone) zone.classList.add("has-file");
    return;
  }

  if (main) main.textContent = t("panel.drop_main");
  if (zone) zone.classList.remove("has-file");
}

export function wireDropzone() {
  const zone = $("dropZone");
  const input = $("fileInput");
  if (!zone || !input) return;

  input.addEventListener("change", () => setDropFile(input.files[0] ? input.files[0].name : ""));

  ["dragenter", "dragover"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    zone.addEventListener(eventName, (event) => {
      event.preventDefault();
      zone.classList.remove("dragover");
    });
  });

  zone.addEventListener("drop", (event) => {
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    if (!file) return;
    input.files = event.dataTransfer.files;
    setDropFile(file.name);
  });
}
