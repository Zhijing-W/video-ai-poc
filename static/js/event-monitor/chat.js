import { chatAboutRun } from "./api.js?v=20260808-foundry-routing";
import { t } from "./i18n.js";
import { setStatus } from "./render.js?v=20260808-foundry-routing";
import { getChatModelValue } from "./settings.js?v=20260808-foundry-routing";
import { getLastPayload } from "./state.js";
import { $, esc } from "./utils.js";

function appendMessage(role, text, meta = "") {
  const messages = $("chatMessages");
  const item = document.createElement("div");
  item.className = `em-chat-message ${role}`;
  item.innerHTML =
    `<div class="em-chat-role">${role === "user" ? t("chat.user") : "AI"}</div>` +
    `<div class="em-chat-text">${esc(text)}</div>` +
    (meta ? `<div class="em-chat-meta">${esc(meta)}</div>` : "");
  messages.appendChild(item);
  messages.scrollTop = messages.scrollHeight;
}

function evidenceText(data) {
  return (data.evidence || [])
    .map((item) => {
      const range = Array.isArray(item.time_range) ? item.time_range.join("~") : "";
      return t("chat.evidence_window", { index: item.window_index ?? "?", range });
    })
    .join(" · ");
}

export async function sendChatQuestion() {
  const payload = getLastPayload();
  const input = $("chatQuestion");
  const button = $("btnChatSend");
  const question = input.value.trim();
  if (!payload?.run_id || !question) return;

  appendMessage("user", question);
  input.value = "";
  button.disabled = true;
  setStatus(t("chat.answering"));
  try {
    const data = await chatAboutRun(payload.run_id, question, getChatModelValue());
    const selection = data.selection || {};
    const meta = [
      selection.model,
      selection.reason,
      Number.isFinite(Number(data.latency_ms)) ? `${data.latency_ms}ms` : "",
      data.usage?.total_tokens ? `${data.usage.total_tokens} tokens` : "",
      evidenceText(data),
    ].filter(Boolean).join(" · ");
    appendMessage("assistant", data.answer, meta);
    $("chatModelStatus").textContent =
      `${selection.requested || "auto"} → ${selection.model || selection.selected || t("results.meta_unknown")}`;
    setStatus(t("chat.completed"));
  } catch (error) {
    appendMessage("assistant", t("chat.failed_answer", { message: error.message }));
    setStatus(t("chat.failed", { message: error.message }), true);
  } finally {
    button.disabled = false;
    input.focus();
  }
}

export function wireChat() {
  $("btnChatSend").addEventListener("click", sendChatQuestion);
  $("chatQuestion").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendChatQuestion();
    }
  });
}
