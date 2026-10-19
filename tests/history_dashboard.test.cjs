"use strict";
// Dependency-free tests of the actual private-history browser export logic.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "../demo/history.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/i)?.[1];
assert.ok(script);

function mount(handler) {
  const elements = new Map(), calls = [], downloaded = [], revoked = [];
  function create(tag) {
    return {
      tag, value: "", textContent: "", disabled: false, children: [],
      listeners: {},
      addEventListener(name, callback) { this.listeners[name] = callback; },
      replaceChildren() { this.children = []; },
      appendChild(child) { this.children.push(child); },
      click() { downloaded.push({download: this.download, href: this.href}); },
      setAttribute() {},
    };
  }
  function get(id) {
    if (!elements.has(id)) elements.set(id, create(id));
    return elements.get(id);
  }
  const fetch = async (url, options) => {
    calls.push({url, options});
    return handler(url, options);
  };
  vm.runInNewContext(script, {
    document: {getElementById: get, createElement: create},
    window: {confirm: () => true},
    fetch,
    URL: {createObjectURL: () => "blob:export", revokeObjectURL: value => revoked.push(value)},
    setTimeout: callback => callback(),
  }, {filename: "demo/history.html"});
  return {get, calls, downloaded, revoked};
}

test("requires an operator key before exporting any private history", async () => {
  const ui = mount(async () => { throw new Error("must not fetch"); });
  await ui.get("export-csv").listeners.click();
  assert.match(ui.get("error").textContent, /operator API key/);
  assert.equal(ui.calls.length, 0);
});

test("valid lot is escaped and key only appears in header", async () => {
  const ui = mount(async () => ({
    ok: true, status: 200, blob: async () => "private csv",
    headers: {get(name) {return name === "X-Export-Rows" ? "3" : "false";}},
  }));
  ui.get("api-key").value = "private-operator-key";
  ui.get("lot-filter").value = "LOT-2026-A01";
  await ui.get("export-csv").listeners.click();
  assert.equal(ui.calls.length, 1);
  assert.equal(ui.calls[0].url, "/inspections/export.csv?limit=500&lot_id=LOT-2026-A01");
  assert.equal(ui.calls[0].options.headers["X-IhsanVision-Key"], "private-operator-key");
  assert.equal(ui.calls[0].options.cache, "no-store");
  assert.ok(!ui.calls[0].url.includes("private-operator-key"));
  assert.equal(ui.downloaded[0].download, "ihsanvision-inspections.csv");
  assert.deepEqual(ui.revoked, ["blob:export"]);
  assert.match(ui.get("status").textContent, /Exported 3 private records/);
});

test("refuses malformed lot IDs and handles server errors without download", async () => {
  const ui = mount(async () => ({ok:false,status:401}));
  ui.get("api-key").value = "some-key";
  ui.get("lot-filter").value = "=FORMULA";
  await ui.get("export-csv").listeners.click();
  assert.equal(ui.calls.length, 0);
  assert.match(ui.get("error").textContent, /Invalid lot/);
  ui.get("lot-filter").value = "";
  await ui.get("export-csv").listeners.click();
  assert.equal(ui.calls.length, 1);
  assert.match(ui.get("error").textContent, /Invalid operator API key/);
  assert.equal(ui.downloaded.length, 0);
});


test("locking clears key and all private views without deleting server records", async () => {
  const ui = mount(async () => { throw new Error("must not fetch"); });
  ui.get("api-key").value = "private-key";
  for (const id of ["rows", "lot-rows", "version-rows"]) ui.get(id).children.push({textContent:"private"});
  ui.get("lock").listeners.click();
  assert.equal(ui.get("api-key").value, "");
  for (const id of ["rows", "lot-rows", "version-rows"]) assert.equal(ui.get(id).children.length, 0);
  assert.equal(ui.calls.length, 0);
  assert.match(ui.get("status").textContent, /Locked/);
});

test("changing credentials or lot filter clears every stale summary", () => {
  const ui = mount(async () => { throw new Error("must not fetch"); });
  for (const control of ["api-key", "lot-filter"]) {
    for (const id of ["rows", "lot-rows", "version-rows"]) ui.get(id).children.push({});
    ui.get(control).listeners.input();
    for (const id of ["rows", "lot-rows", "version-rows"]) assert.equal(ui.get(id).children.length, 0);
  }
});

test("revoked key clears previously displayed records and credentials", async () => {
  const ui = mount(async () => ({ok:false,status:401}));
  ui.get("api-key").value = "revoked-key";
  for (const id of ["rows", "lot-rows", "version-rows"]) ui.get(id).children.push({});
  await ui.get("export-csv").listeners.click();
  assert.equal(ui.get("api-key").value, "");
  for (const id of ["rows", "lot-rows", "version-rows"]) assert.equal(ui.get(id).children.length, 0);
  assert.equal(ui.downloaded.length, 0);
  assert.equal(ui.calls[0].options.cache, "no-store");
});
