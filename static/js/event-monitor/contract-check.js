const fs = require("node:fs");
const path = require("node:path");

const repoRoot = path.resolve(__dirname, "..", "..", "..");
const templatePath = path.join(repoRoot, "templates", "event-monitor.html");
const cssEntryPath = path.join(repoRoot, "static", "css", "event-monitor.css");
const jsDir = path.join(repoRoot, "static", "js", "event-monitor");

const template = fs.readFileSync(templatePath, "utf8");
const templateIds = new Set([...template.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]));
const jsFiles = fs
  .readdirSync(jsDir)
  .filter((name) => name.endsWith(".js") && name !== "contract-check.js")
  .sort();

const requiredIds = new Set();
for (const file of jsFiles) {
  const content = fs.readFileSync(path.join(jsDir, file), "utf8");
  for (const match of content.matchAll(/\$\("([^"]+)"\)/g)) requiredIds.add(match[1]);
  for (const match of content.matchAll(/querySelector(?:All)?\("#([^"\s.]+)/g)) requiredIds.add(match[1]);
}

const missingIds = [...requiredIds].filter((id) => !templateIds.has(id)).sort();
const cssEntry = fs.readFileSync(cssEntryPath, "utf8");
const imports = [...cssEntry.matchAll(/@import\s+url\("([^"]+)"\)\s*;/g)].map((match) => match[1]);
const missingImports = imports
  .filter((importPath) => !fs.existsSync(path.resolve(path.dirname(cssEntryPath), importPath)))
  .sort();
const moduleScriptOk = /<script\s+type="module"\s+src="\/static\/js\/event-monitor\/main\.js\?v=[^"]+"><\/script>/.test(
  template
);

console.log(`Template IDs: ${templateIds.size}`);
console.log(`JS queried IDs: ${requiredIds.size}`);
console.log(`CSS imports: ${imports.length}`);

if (!moduleScriptOk) {
  console.error("Template script tag is not a type=module main.js reference.");
}
if (missingIds.length) {
  console.error("Missing template IDs:", missingIds.join(", "));
}
if (missingImports.length) {
  console.error("Missing CSS imports:", missingImports.join(", "));
}

if (!moduleScriptOk || missingIds.length || missingImports.length) {
  process.exitCode = 1;
} else {
  console.log("Event Monitor static contract check passed.");
}
