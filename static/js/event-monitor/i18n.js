const bundle = globalThis.__EVENT_MONITOR_I18N__ || {
  locale: "zh-CN",
  reportLanguage: "zh-CN",
  messages: {},
};

function getByPath(path) {
  return String(path)
    .split(".")
    .reduce((value, segment) => (value && value[segment] != null ? value[segment] : undefined), bundle.messages);
}

export function t(path, vars = {}) {
  const template = getByPath(path);
  if (typeof template !== "string") return path;
  return template.replace(/\{(\w+)\}/g, (_, key) => String(vars[key] ?? `{${key}}`));
}

export function labelLevel(level) {
  return t(`levels.${level}`);
}

export function uiLocale() {
  return bundle.locale || "zh-CN";
}

export function reportLanguage() {
  return bundle.reportLanguage || "zh-CN";
}

export function clockLocale() {
  return uiLocale() === "en" ? "en-GB" : "zh-CN";
}

export function common(path) {
  return t(`common.${path}`);
}
