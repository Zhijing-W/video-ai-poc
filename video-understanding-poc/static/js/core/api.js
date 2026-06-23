/**
 * API 调用模块
 * 为拆分后的监控、比对、历史日志模块提供统一请求封装。
 */

import { API } from './config.js';

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    return {
      ok: false,
      status: response.status,
      errorText: await response.text(),
      response,
    };
  }

  return {
    ok: true,
    status: response.status,
    data: await response.json(),
    response,
  };
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error((await response.text()) || `GET ${url} failed: ${response.status}`);
  }
  return await response.json();
}

async function deleteRequest(url) {
  const response = await fetch(url, { method: 'DELETE' });
  if (!response.ok) {
    throw new Error((await response.text()) || `DELETE ${url} failed: ${response.status}`);
  }
  return true;
}

export function analyzeFrameRequest(payload) {
  return postJson(API.analyzeFrame, payload);
}

export function cruiseFrameRequest(payload) {
  return postJson(API.cruiseFrame, payload);
}

export function trackResetRequest(sessionId) {
  return postJson(API.trackReset, { session_id: sessionId });
}

export function compileTargetRequest(payload) {
  return postJson(API.compileTarget, payload);
}

export function saveMonitorSession(payload) {
  return postJson(API.sessions, payload);
}

export function listMonitorSessions() {
  return getJson(API.sessions);
}

export function getMonitorSession(id) {
  return getJson(`${API.sessions}/${id}`);
}

export function deleteMonitorSession(id) {
  return deleteRequest(`${API.sessions}/${id}`);
}

export async function analyzeFrame(frameB64, targetDesc = '', targetImg = null, enableGate = true) {
  const result = await analyzeFrameRequest({
    image: frameB64,
    target: targetDesc || null,
    reference_image: targetImg,
    gate_enabled: enableGate,
  });
  if (!result.ok) {
    throw new Error(result.errorText || `API error: ${result.status}`);
  }
  return result.data;
}

export async function compileTarget(targetDesc, targetImg = null) {
  const result = await compileTargetRequest({
    target: targetDesc,
    reference_image: targetImg,
  });
  if (!result.ok) {
    throw new Error(result.errorText || `Compile target error: ${result.status}`);
  }
  return result.data;
}

export async function cruiseFrame(frameB64, targetDesc = '', targetImg = null) {
  const result = await cruiseFrameRequest({
    image: frameB64,
    target: targetDesc || null,
    reference_image: targetImg,
  });
  if (!result.ok) {
    throw new Error(result.errorText || `Cruise API error: ${result.status}`);
  }
  return result.data;
}

export function getSessions() {
  return listMonitorSessions();
}
