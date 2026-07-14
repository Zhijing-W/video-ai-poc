/**
 * 全局运行时状态
 * 集中保存从 monitor.html.backup-refactor 抽离出的共享变量和重置逻辑。
 */

export const BACKFILL_MAX = 4;

const initialMode = typeof localStorage !== 'undefined'
  ? (localStorage.getItem('view-mode') || 'tech')
  : 'tech';

export const appState = {
  currentMode: initialMode,
  stream: null,
  timer: null,
  inFlight: false,
  refDataUri: null,
  stats: { frames: 0, match: 0, alert: 0, llm: 0, skipped: 0 },
  activeCompare: null,
  frameStore: {},
  frameSeq: 0,
  cycle: null,
  lastCounts: null,
  lastLlmTs: 0,
  cruiseBuf: [],
  cruiseCount: 0,
  lastLlmSig: null,
  lastLlmData: null,
  lastAuditCount: 0,
  lastCruiseVerdict: null,
  cruiseFlip: false,
  camOn: false,
  cooldownTimer: null,
  stageClear: null,
  yoloModelSel: 'yolov8m',
  llmModelSel: 'gpt-4o',
  backfilling: false,
  videoFile: null,      // 整段视频分析用：当前选中的视频文件（摄像头时为 null）
  lastSig: null,        // 智能抽帧：上一帧灰度指纹
  lastSentTs: 0,        // 智能抽帧：上次真正抽帧（送后端）的时间戳
  fullVideoRun: false,  // 整段视频分析模式：正在把视频当流跑完整段
  cycleSummary: null,   // 本次整段分析的末尾总结（随归档保存）
  trackSessionId: null, // Phase 3 · Step 12：本轮监控的 track 门控会话 id（开始监控时生成）
};

export function setCurrentMode(mode) {
  appState.currentMode = mode;
}

export function resetStats() {
  appState.stats = { frames: 0, match: 0, alert: 0, llm: 0, skipped: 0 };
}

export function resetFrameStore() {
  appState.frameStore = {};
  appState.frameSeq = 0;
}

export function nextFrameId() {
  appState.frameSeq += 1;
  return appState.frameSeq;
}

export function resetCruiseState() {
  appState.cruiseBuf = [];
  appState.cruiseCount = 0;
  appState.lastAuditCount = 0;
  appState.lastCruiseVerdict = null;
  appState.cruiseFlip = false;
  appState.backfilling = false;
}

export function resetLlmCache() {
  appState.lastCounts = null;
  appState.lastLlmTs = 0;
  appState.lastLlmSig = null;
  appState.lastLlmData = null;
}

export function resetCompareRuntime() {
  resetCruiseState();
  resetLlmCache();
  appState.inFlight = false;
}

export function startCycle() {
  appState.cycle = { startedAt: new Date().toISOString(), entries: [] };
}

export function appendCycleEntry(entry) {
  if (appState.cycle) {
    appState.cycle.entries.push(entry);
  }
}

export function clearCycle() {
  appState.cycle = null;
}

export function clearActiveCompare() {
  appState.activeCompare = null;
  resetCruiseState();
}
