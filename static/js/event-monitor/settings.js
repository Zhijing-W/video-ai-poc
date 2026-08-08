import { reportLanguage, t, uiLocale } from "./i18n.js";
import { $ } from "./utils.js";

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

export function renderLlmModels(catalog = {}) {
  replaceModelOptions($("analysisModel"), catalog.analysis, catalog.defaults?.analysis);
  replaceModelOptions($("chatModel"), catalog.chat, catalog.defaults?.chat);
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
