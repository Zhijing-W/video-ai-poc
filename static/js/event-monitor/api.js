async function readError(response) {
  const error = await response.json().catch(() => ({}));
  throw new Error(error.detail || `HTTP ${response.status}`);
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

export async function runAnalysis(formData) {
  const response = await fetch("/api/event-monitor/understand", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) await readError(response);
  return JSON.parse(await response.text());
}

export async function completeDryRun(payload, objective) {
  const response = await fetch("/api/event-monitor/complete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      payload,
      objective: objective || null,
    }),
  });
  if (!response.ok) await readError(response);
  return response.json();
}

export async function health() {
  const response = await fetch("/api/event-monitor/samples", { method: "GET" });
  return response.ok;
}
