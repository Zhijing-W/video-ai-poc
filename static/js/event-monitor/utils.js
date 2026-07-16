const ESCAPES = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

export const $ = (id) => document.getElementById(id);

export const esc = (value) =>
  String(value == null ? "" : value).replace(/[&<>"']/g, (char) => ESCAPES[char]);

export function subjectHue(subjectId) {
  if (subjectId == null) return 205;
  let hue = 0;
  for (const char of String(subjectId)) hue = (hue * 31 + char.charCodeAt(0)) % 360;
  return hue;
}

export function baseName(filePath) {
  return String(filePath || "").split(/[\\/]/).pop();
}
