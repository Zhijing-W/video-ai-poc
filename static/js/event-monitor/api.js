async function readError(response) {
  const error = await response.json().catch(() => ({}));
  const detail = error.detail;
  const failure = new Error(
    (detail && typeof detail === "object" ? detail.message : detail) ||
      `HTTP ${response.status}`
  );
  if (detail && typeof detail === "object") failure.detail = detail;
  throw failure;
}

export async function listSamples() {
  const response = await fetch("/api/event-monitor/samples");
  if (!response.ok) await readError(response);
  return response.json();
}

export async function listSuperresBackends() {
  const response = await fetch("/api/event-monitor/superres-backends");
  if (!response.ok) await readError(response);
  return response.json();
}

export async function listReidBackends() {
  const response = await fetch("/api/event-monitor/reid-backends");
  if (!response.ok) await readError(response);
  return response.json();
}

export async function listLlmModels(language) {
  const query = language ? `?language=${encodeURIComponent(language)}` : "";
  const response = await fetch(`/api/event-monitor/llm-models${query}`);
  if (!response.ok) await readError(response);
  return response.json();
}

export async function runAnalysis(formData) {
  const response = await fetch("/api/event-monitor/understand", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) await readError(response);
  return JSON.parse(await response.text());
}

export async function completeDryRun(payload, objective, language, analysisModel) {
  const response = await fetch("/api/event-monitor/complete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      payload,
      objective: objective || null,
      language: language || null,
      analysis_model: analysisModel || null,
    }),
  });
  if (!response.ok) await readError(response);
  return response.json();
}

export async function chatAboutRun(runId, question, model, language) {
  const response = await fetch(`/api/event-monitor/runs/${encodeURIComponent(runId)}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, model: model || null, language: language || null }),
  });
  if (!response.ok) await readError(response);
  return response.json();
}

export function runPromptUrl(runId, format) {
  return `/api/event-monitor/runs/${encodeURIComponent(runId)}/prompt?format=${encodeURIComponent(format)}`;
}

export async function getRunPrompt(runId, format) {
  const response = await fetch(runPromptUrl(runId, format));
  if (!response.ok) await readError(response);
  return response.text();
}

export async function health() {
  const response = await fetch("/api/event-monitor/samples", { method: "GET" });
  return response.ok;
}
