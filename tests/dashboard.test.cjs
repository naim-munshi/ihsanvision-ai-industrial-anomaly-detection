"use strict";
// Dependency-free UI behavior tests run against the real dashboard inline script.
// Mocking document, fetch and URL keeps this runnable in CI without ML weights.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "../demo/index.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/i)?.[1];
assert.ok(script, "Expected inline dashboard logic");

async function mount({response, models} = {}) {
  const elements = new Map();
  const fetchCalls = [];
  const revocations = [];
  const modelData = models ?? {available_models: ["padim", "patchcore"], default_model: "padim"};
  function elem(id) {
    if (!elements.has(id)) {
      elements.set(id, {
        id, style: {}, textContent: "", value: "", disabled: false, files: [],
        listeners: {}, children: [],
        addEventListener(name, listener) { this.listeners[name] = listener; },
        replaceChildren() { this.children = []; this.value = ""; },
        appendChild(child) { this.children.push(child); },
        removeAttribute(attr) { delete this[attr]; },
      });
    }
    return elements.get(id);
  }
  const fetch = async (url, options) => {
    fetchCalls.push({url, options});
    if (url === "/models") return {ok: true, json: async () => modelData};
    if (typeof response === "function") return response(url, options);
    return response ?? {
      ok: true, status: 200,
      json: async () => ({
        model: "padim", anomaly_score: 0.864, predicted_anomaly: true,
        latency_ms: 123, heatmap_png_base64: "QUJD",
      }),
    };
  };
  class FormData {
    constructor() { this.parts = []; }
    append(name, file) { this.parts.push([name, file]); }
  }
  vm.runInNewContext(script, {
    document: {getElementById: elem, createElement: () => elem("option-"+Math.random())},
    fetch, URL: {createObjectURL: () => "blob:test-image", revokeObjectURL: (u) => revocations.push(u)},
    FormData, console,
  }, {filename: "demo/index.html"});
  await new Promise(resolve => setImmediate(resolve)); // let /models discovery finish
  return {elem, fetchCalls, revocations};
}
const validFile = {name: "normal_bottle.png", type: "image/png", size: 2048};

function selectFile(ui, file) {
  ui.elem("file").files = file ? [file] : [];
  ui.elem("file").listeners.change();
}

test("loads two model options and blocks Analyze without a valid file", async () => {
  const ui = await mount();
  assert.equal(ui.elem("model-choice").children.length, 2);
  assert.equal(ui.elem("model-choice").value, "padim");
  assert.equal(ui.elem("analyze").disabled, true);
  selectFile(ui, validFile);
  assert.equal(ui.elem("analyze").disabled, false);
  assert.equal(ui.elem("preview").src, "blob:test-image");
});

test("rejects empty, oversized, unsupported files before API submission", async () => {
  const ui = await mount();
  for (const file of [
    {name: "empty.png", type: "image/png", size: 0},
    {name: "large.png", type: "image/png", size: 10 * 1024 * 1024 + 1},
    {name: "text.txt", type: "text/plain", size: 30},
  ]) {
    selectFile(ui, file);
    assert.equal(ui.elem("analyze").disabled, true);
    assert.notEqual(ui.elem("error").textContent, "");
    assert.equal(ui.elem("preview").style.display, "none");
  }
  assert.equal(ui.fetchCalls.filter(row => row.url.startsWith("/predict")).length, 0);
});

test("submits selected file to selected model and clears stale result on changes", async () => {
  const ui = await mount();
  selectFile(ui, validFile);
  await ui.elem("analyze").listeners.click();
  assert.equal(ui.elem("result").style.display, "block");
  assert.equal(ui.elem("result-file").textContent, validFile.name);
  assert.equal(ui.elem("model").textContent, "PaDiM");
  assert.equal(ui.elem("score").textContent, "0.8640");
  assert.equal(ui.elem("label").textContent, "Flagged anomaly");
  assert.equal(ui.elem("heatmap").src, "data:image/png;base64,QUJD");
  assert.equal(ui.elem("file").disabled, false);
  const prediction = ui.fetchCalls.find(row => row.url === "/predict?model=padim");
  assert.ok(prediction);
  assert.equal(prediction.options.body.parts[0][1], validFile);

  ui.elem("model-choice").value = "patchcore";
  ui.elem("model-choice").listeners.change();
  assert.equal(ui.elem("result").style.display, "none");
  assert.equal(ui.elem("heatmap").src, undefined);

  selectFile(ui, {...validFile, name: "defective_bottle.png"});
  assert.equal(ui.elem("result").style.display, "none");
  assert.equal(ui.elem("heatmap").src, undefined);
  assert.deepEqual(ui.revocations, ["blob:test-image"]);
});

test("shows actionable 401 error, not stale score", async () => {
  const ui = await mount({
    response: async () => ({ok: false, status: 401, json: async () => ({detail: "Unauthorized"})}),
  });
  selectFile(ui, validFile);
  await ui.elem("analyze").listeners.click();
  assert.match(ui.elem("error").textContent, /API key is missing or invalid/);
  assert.equal(ui.elem("result").style.display, "none");
  assert.equal(ui.elem("analyze").disabled, false);
  assert.equal(ui.elem("file").disabled, false);
});

test("rejects stale/mismatched model response without displaying it", async () => {
  const ui = await mount({
    response: async () => ({
      ok: true, status: 200,
      json: async () => ({model: "patchcore", anomaly_score: 0.3, predicted_anomaly: false}),
    }),
  });
  selectFile(ui, validFile);
  await ui.elem("analyze").listeners.click();
  assert.match(ui.elem("error").textContent, /does not match the selected model/);
  assert.equal(ui.elem("result").style.display, "none");
});

test("ignores in-flight file-change events and unlocks after response", async () => {
  let release;
  const pending = new Promise(resolve => { release = resolve; });
  const ui = await mount({
    response: async () => {
      await pending;
      return {
        ok: true, status: 200,
        json: async () => ({
          model: "padim", anomaly_score: 0.3, predicted_anomaly: false,
          latency_ms: 40, heatmap_png_base64: "",
        }),
      };
    },
  });
  selectFile(ui, validFile);
  const call = ui.elem("analyze").listeners.click();
  assert.equal(ui.elem("file").disabled, true);
  assert.equal(ui.elem("model-choice").disabled, true);
  ui.elem("file").listeners.change();
  release();
  await call;
  assert.equal(ui.elem("file").disabled, false);
  assert.equal(ui.elem("model-choice").disabled, false);
  assert.equal(ui.elem("result-file").textContent, validFile.name);
});

test("multi-category selector routes only configured category/model and resets result", async () => {
  const categories = {
    available_models: ["padim"], default_model: "padim",
    default_category: "bottle",
    available_categories: ["bottle", "cable"],
    models_by_category: {bottle: ["padim"], cable: ["patchcore"]},
  };
  const ui = await mount({
    models: categories,
    response: async () => ({
      ok: true, status: 200,
      json: async () => ({
        category: "cable", model: "patchcore", anomaly_score: 0.912,
        predicted_anomaly: true, latency_ms: 21, heatmap_png_base64: "QUJD",
      }),
    }),
  });
  assert.equal(ui.elem("category-choice").value, "bottle");
  assert.equal(ui.elem("model-choice").value, "padim");
  assert.equal(ui.elem("category-choice").children.length, 2);
  selectFile(ui, validFile);
  ui.elem("category-choice").value = "cable";
  ui.elem("category-choice").listeners.change();
  assert.equal(ui.elem("model-choice").value, "patchcore");
  assert.equal(ui.elem("model-choice").children.length, 1);
  await ui.elem("analyze").listeners.click();
  assert.equal(ui.elem("result").style.display, "block");
  assert.equal(ui.elem("result-category").textContent, "cable");
  assert.equal(ui.elem("score").textContent, "0.9120");
  assert.ok(ui.fetchCalls.some(row => row.url === "/predict?model=patchcore&category=cable"));
  ui.elem("category-choice").value = "bottle";
  ui.elem("category-choice").listeners.change();
  assert.equal(ui.elem("result").style.display, "none");
  assert.equal(ui.elem("heatmap").src, undefined);
});

test("rejects response that reports the wrong category", async () => {
  const ui = await mount({
    models: {available_models: [], default_category: "cable",
      models_by_category: {cable: ["patchcore"]}},
    response: async () => ({ok: true, status: 200,
      json: async () => ({
        category: "bottle", model: "patchcore", anomaly_score: 0.4,
        predicted_anomaly: true, latency_ms: 15,
      }),
    }),
  });
  selectFile(ui, validFile);
  await ui.elem("analyze").listeners.click();
  assert.match(ui.elem("error").textContent, /invalid or does not match/);
  assert.equal(ui.elem("result").style.display, "none");
});

test("legacy /models response still defaults to bottle without category query", async () => {
  const ui = await mount();
  assert.equal(ui.elem("category-choice").value, "bottle");
  selectFile(ui, validFile);
  await ui.elem("analyze").listeners.click();
  assert.ok(ui.fetchCalls.some(row => row.url === "/predict?model=padim"));
});
