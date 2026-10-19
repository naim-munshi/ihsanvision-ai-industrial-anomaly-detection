"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "../demo/batch.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/i)?.[1];
assert.ok(script, "Batch workflow must contain executable browser logic");

async function mount({models, predict} = {}) {
  const elements = new Map(), fetchCalls = [], blobs = [], downloads = [];
  function element(tag = "div") {
    return {
      tag, value: "", disabled: false, textContent: "", files: [],
      children: [], handlers: {}, style: {}, className: "",
      addEventListener(name, callback) { this.handlers[name] = callback; },
      appendChild(child) { this.children.push(child); },
      replaceChildren() { this.children = []; this.value = ""; },
      click() { downloads.push({href: this.href, filename: this.download}); },
    };
  }
  function get(id) {
    if (!elements.has(id)) elements.set(id, element(id));
    return elements.get(id);
  }
  const defaultCatalog = {
    available_models: ["padim"], default_model: "padim",
    available_categories: ["bottle", "cable"],
    default_category: "bottle",
    models_by_category: {bottle: ["padim"], cable: ["patchcore"]},
  };
  const fetch = async (url, options) => {
    fetchCalls.push({url, options});
    if (url === "/models") return {
      ok: true, status: 200,
      json: async () => models ?? defaultCatalog,
    };
    if (predict) return predict(url, options);
    const requestedModel = url.includes("patchcore") ? "patchcore" : "padim";
    const category = url.includes("category=cable") ? "cable" : "bottle";
    return {ok: true, status: 200, json: async () => ({
      model: requestedModel, category, anomaly_score: 0.85,
      predicted_anomaly: true, latency_ms: 43, heatmap_png_base64: "NOT_PERSISTED",
    })};
  };
  const context = {
    document: {getElementById: get, createElement: element},
    fetch,
    FormData: class { constructor() { this.files = []; } append(name, file) { this.files.push([name, file]); } },
    Blob: class {constructor(chunks) {this.parts = chunks; blobs.push(this);}},
    URL: {createObjectURL: () => "blob:report", revokeObjectURL: () => {}},
    setTimeout: (callback) => callback(),
  };
  vm.runInNewContext(script, context, {filename: "demo/batch.html"});
  await new Promise(resolve => setImmediate(resolve));
  return {get, fetchCalls, blobs, downloads};
}
const sample = (name="normal.png") => ({name, type:"image/png", size:128});

test("discovers trained categories and runs browser batch through correct model routing", async () => {
  const ui = await mount();
  assert.equal(ui.get("category").value, "bottle");
  assert.equal(ui.get("model").value, "padim");
  ui.get("category").value = "cable";
  ui.get("category").handlers.change();
  assert.equal(ui.get("model").value, "patchcore");
  ui.get("images").files = [sample("first.png"), sample("second.png")];
  ui.get("images").handlers.change();
  assert.equal(ui.get("run").disabled, false);
  await ui.get("run").handlers.click();
  assert.equal(ui.get("rows").children.length, 2);
  assert.equal(ui.get("rows").children[0].children[1].textContent, "anomaly");
  assert.equal(ui.get("rows").children[0].children[2].textContent, "0.8500");
  assert.equal(ui.fetchCalls.filter(call => call.url === "/predict?model=patchcore&category=cable").length, 2);
  assert.equal(ui.get("download").disabled, false);
  ui.get("category").value = "bottle";
  ui.get("category").handlers.change();
  assert.equal(ui.get("rows").children.length, 0);
  assert.equal(ui.get("download").disabled, true);
});

test("rejects oversized, empty and excessive batches before sending", async () => {
  const ui = await mount();
  const input = ui.get("images");
  for(const files of [
    [sample("zero.png"), {...sample("empty.png"), size:0}],
    [{...sample("huge.png"), size:11*1024*1024}],
    Array.from({length:21},(_,i)=>sample(i+".png")),
  ]){
    input.files=files;
    input.handlers.change();
    assert.equal(ui.get("run").disabled, true);
  }
  assert.equal(ui.fetchCalls.filter(call => call.url.startsWith("/predict")).length, 0);
});

test("HTTP errors stay errors; they never produce normal outcomes", async () => {
  const ui = await mount({
    predict: async () => ({
      ok:false,status:503,json:async()=>({detail:"Internal checkpoint path must not be shown"}),
    }),
  });
  ui.get("images").files=[sample()];
  ui.get("images").handlers.change();
  await ui.get("run").handlers.click();
  assert.equal(ui.get("rows").children[0].children[1].textContent, "error");
  assert.equal(ui.get("rows").children[0].children[4].textContent, "HTTP 503");
  assert.equal(ui.get("download").disabled, false);
  assert.doesNotMatch(ui.get("progress").textContent,/0 errors/);
});

test("untrusted file names are inserted as text only and formula-safe in CSV", async () => {
  const ui = await mount();
  const name='=HYPERLINK("url","bad").png';
  ui.get("images").files=[sample(name)];
  ui.get("images").handlers.change();
  await ui.get("run").handlers.click();
  assert.equal(ui.get("rows").children[0].children[0].textContent,name);
  ui.get("download").handlers.click();
  assert.equal(ui.downloads.length,1);
  assert.equal(ui.downloads[0].filename,"ihsanvision-batch-inspection.csv");
  const csv=ui.blobs[0].parts[0];
  assert.ok(csv.includes("'=HYPERLINK"));
  assert.ok(!csv.includes("NOT_PERSISTED"));
});

test("rejects category mismatched responses and preserves legacy model discovery", async () => {
  const ui = await mount({
    models: {available_models:["padim"],default_model:"padim"},
    predict: async () => ({
      ok:true,status:200,json:async()=>({
        category:"cable",model:"padim",anomaly_score:0.5,predicted_anomaly:false,
      }),
    }),
  });
  assert.equal(ui.get("category").value,"bottle");
  ui.get("images").files=[sample()];
  ui.get("images").handlers.change();
  await ui.get("run").handlers.click();
  assert.equal(ui.get("rows").children[0].children[1].textContent,"error");
  assert.equal(ui.fetchCalls.find(call=>call.url.startsWith("/predict")).url,"/predict?model=padim");
});

test("accepts opaque tracked lot ID and exports it without altering image decisions", async () => {
  const ui = await mount();
  ui.get("lot-id").value = "LOT-2026-A01";
  ui.get("lot-id").handlers.input();
  ui.get("images").files = [sample("item.png")];
  ui.get("images").handlers.change();
  assert.equal(ui.get("run").disabled, false);
  await ui.get("run").handlers.click();
  const attempts = ui.fetchCalls.filter(x => x.url.startsWith("/predict"));
  assert.equal(attempts.length, 1);
  assert.equal(attempts[0].url, "/predict?model=padim&lot_id=LOT-2026-A01");
  ui.get("download").handlers.click();
  assert.match(ui.blobs[0].parts[0], /lot_id/);
  assert.match(ui.blobs[0].parts[0], /LOT-2026-A01/);
});

test("rejects unsafe lot identifiers and preserves untracked legacy behavior", async () => {
  const ui = await mount();
  ui.get("images").files = [sample("item.png")];
  ui.get("images").handlers.change();
  for (const bad of ["AB", "=HYPERLINK", "../secrets", "has space", "x".repeat(41)]) {
    ui.get("lot-id").value = bad;
    ui.get("lot-id").handlers.input();
    assert.equal(ui.get("run").disabled, true, bad);
  }
  assert.equal(ui.fetchCalls.filter(x => x.url.startsWith("/predict")).length, 0);
  ui.get("lot-id").value = "";
  ui.get("lot-id").handlers.input();
  assert.equal(ui.get("run").disabled, false);
  await ui.get("run").handlers.click();
  assert.equal(ui.fetchCalls.find(x => x.url.startsWith("/predict")).url,
               "/predict?model=padim");
});


test("batch controls keep the operator key fixed during inference", async () => {
  let release;
  const pending = new Promise(resolve => { release = resolve; });
  const ui = await mount({predict:async () => {
    await pending;
    return {ok:true,status:200,json:async()=>({model:"padim",category:"bottle",anomaly_score:0.2,predicted_anomaly:false})};
  }});
  ui.get("images").files=[sample()];
  ui.get("images").handlers.change();
  const request=ui.get("run").handlers.click();
  assert.equal(ui.get("api-key").disabled,true);
  assert.equal(ui.get("clear-selection").disabled,true);
  release();
  await request;
  assert.equal(ui.get("api-key").disabled,false);
  // Native file inputs clear FileList when value is assigned an empty string.
  Object.defineProperty(ui.get("images"), "value", {set(v){if(v === "")this.files=[];}});
  ui.get("clear-selection").handlers.click();
  assert.equal(ui.get("rows").children.length,0);
  assert.equal(ui.get("run").disabled,true);
  assert.equal(ui.get("download").disabled,true);
});
