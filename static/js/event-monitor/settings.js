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

  return { file, sample, dryRun, objective, formData };
}

export function getObjectiveValue() {
  return $("objective").value.trim();
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
    off: "关闭",
    gfpgan: "GFP-GAN",
    codeformer: "CodeFormer",
    realesrgan_x2plus: "Real-ESRGAN x2plus",
  };
  const defaultBackend = catalog.default || "off";
  select.dataset.defaultBackend = defaultBackend;
  select.replaceChildren();
  select.add(new Option(`默认（${labels[defaultBackend] || defaultBackend}）`, ""));
  allowed.forEach((name) => select.add(new Option(labels[name] || name, name)));

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
  const productBackends = ["differ", "clipreid", "siglip2", "osnet", "resnet50", "auto"];
  const registered = new Set(names);
  const allowed = catalogLoaded
    ? [
      ...productBackends.filter((name) => registered.has(name)),
      ...names.filter((name) => !productBackends.includes(name)),
    ]
    : [];
  const fallbackLabels = {
    osnet: "OSNet-AIN MSMT17",
    resnet50: "ResNet50 ImageNet",
    coarse: "颜色直方图",
    clipreid: "CLIP-ReID ViT-B/16",
    siglip2: "SigLIP2 Person ReID",
    differ: "DIFFER EVA02-L",
    auto: "自动回退（OSNet → ResNet50 → coarse）",
  };
  const metadata = catalog.metadata || {};
  const labelFor = (name) => metadata[name]?.label || fallbackLabels[name] || name;
  const configuredDefault = allowed.includes(catalog.default) ? catalog.default : "";
  select.replaceChildren();
  const defaultLabel = configuredDefault
    ? `使用服务端默认（${labelFor(configuredDefault)}）`
    : (catalogLoaded ? "使用服务端默认" : "使用服务端默认（后端目录不可用）");
  select.add(new Option(defaultLabel, ""));
  allowed.forEach((name) => {
    const suffix = name === configuredDefault
      ? (name === "differ" ? "（默认·精度优先）" : "（配置默认）")
      : "";
    const label = `${labelFor(name)}${suffix}`;
    select.add(new Option(label, name));
  });
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

  if (main) main.textContent = "拖拽视频到此，或点击选择";
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
