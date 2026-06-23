/**
 * 配置常量
 * 存储应用的配置参数和常量定义
 */

// 流程阶段定义
export const STAGES = [
  "frame", "detect", "yolo_llm", "yolo_direct",
  "gate", "gate_pass", "gate_skip",
  "llm", "compile", "cruise",
  "result", "match", "mismatch"
];

// 冷却时间（秒）
export const COOLDOWN_DURATION = 2;

// API 端点
export const API = {
  analyzeFrame: "/analyze-frame",
  detect: "/detect",
  track: "/track",
  trackReset: "/track/reset",
  compileTarget: "/compile-target",
  cruiseFrame: "/cruise-frame",
  sessions: "/monitor-sessions"
};

// 颜色映射
export const CLASS_COLORS = {
  "person": "#4a9eff",
  "car": "#ff6b6b",
  "bicycle": "#51cf66",
  "motorbike": "#ff922b",
  "bus": "#cc5de8",
  "truck": "#ffd43b",
  "cat": "#ff6b6b",
  "dog": "#74c0fc",
  "default": "#868e96"
};

// 获取 YOLO 类别颜色
export function getClassColor(className) {
  return CLASS_COLORS[className] || CLASS_COLORS.default;
}
