const initialProgressState = () => ({
  timer: null,
  clock: null,
  hideTimer: null,
  startedAt: 0,
  stepIndex: 0,
  pct: 0,
  stages: [],
});

const state = {
  lastPayload: null,
  keyframes: [],
  promptArtifact: null,
  progress: initialProgressState(),
};

export const getLastPayload = () => state.lastPayload;

export function setLastPayload(payload) {
  state.lastPayload = payload;
  return state.lastPayload;
}

export function clearLastPayload() {
  state.lastPayload = null;
}

export const getPromptArtifact = () => state.promptArtifact;

export function setPromptArtifact(format, content) {
  state.promptArtifact = { format, content };
  return state.promptArtifact;
}

export function clearPromptArtifact() {
  state.promptArtifact = null;
}

export function resetKeyframeRegistry() {
  state.keyframes = [];
}

export function registerKeyframe(frame) {
  state.keyframes.push(frame);
  return state.keyframes.length - 1;
}

export function getKeyframe(index) {
  return state.keyframes[index] || null;
}

export const getProgressState = () => state.progress;

export function setProgressState(patch) {
  state.progress = { ...state.progress, ...patch };
  return state.progress;
}

export function resetProgressState() {
  state.progress = initialProgressState();
  return state.progress;
}
