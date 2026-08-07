import { chatAboutRun } from "./api.js?v=20260807-foundry-routing";
import { setStatus } from "./render.js?v=20260807-foundry-routing";
import { getChatModelValue } from "./settings.js?v=20260807-foundry-routing";
import { getLastPayload } from "./state.js";
import { $, esc } from "./utils.js";

function appendMessage(role, text, meta = "") {
  const messages = $("chatMessages");
  const item = document.createElement("div");
  item.className = `em-chat-message ${role}`;
  item.innerHTML =
    `<div class="em-chat-role">${role === "user" ? "你" : "AI"}</div>` +
    `<div class="em-chat-text">${esc(text)}</div>` +
    (meta ? `<div class="em-chat-meta">${esc(meta)}</div>` : "");
  messages.appendChild(item);
  messages.scrollTop = messages.scrollHeight;
}

function evidenceText(data) {
  return (data.evidence || [])
    .map((item) => {
      const range = Array.isArray(item.time_range) ? item.time_range.join("~") : "";
      return `窗${item.window_index ?? "?"}${range ? ` ${range}` : ""}`;
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
  setStatus("⏳ 正在基于本次分析证据回答…");
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
      `${selection.requested || "auto"} → ${selection.model || selection.selected || "unknown"}`;
    setStatus("✓ 问答完成");
  } catch (error) {
    appendMessage("assistant", `回答失败：${error.message}`);
    setStatus("✗ 大模型问答失败：" + error.message, true);
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
