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

async function mount({response, models, camera} = {}) {
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
  elem("camera-video").play=async()=>{};
  elem("camera-video").videoWidth=1280;
  elem("camera-video").videoHeight=960;
  vm.runInNewContext(script, {
    document: {getElementById: elem, createElement: (tag) => tag === "canvas" && camera ? camera.canvas : elem("option-"+Math.random())},
    navigator: camera ? {mediaDevices:{getUserMedia:camera.getUserMedia}} : {},
    File: class {constructor(parts,name,options){this.name=name;this.type=options.type;this.size=parts[0].size;}},
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


test("clear selection releases the preview and blocks resubmission", async () => {
  const ui = await mount();
  selectFile(ui, validFile);
  await ui.elem("analyze").listeners.click();
  ui.elem("clear-selection").listeners.click();
  assert.equal(ui.elem("file").value, "");
  assert.equal(ui.elem("analyze").disabled, true);
  assert.equal(ui.elem("result").style.display, "none");
  assert.equal(ui.elem("preview").src, undefined);
  assert.deepEqual(ui.revocations, ["blob:test-image"]);
});


function cameraFixture() {
  const tracks = [{stopped:false, stop(){this.stopped=true;}, addEventListener(){}}];
  const stream = {getTracks:()=>tracks};
  return {tracks, stream, getUserMedia:async constraints=>{
    assert.equal(constraints.audio,false);
    return stream;
  }, canvas:{getContext:()=>({drawImage(){}}),toBlob:callback=>callback({size:2048})}};
}

test("camera capture stops tracks and submits the photo through existing model route", async()=>{
  const camera=cameraFixture();
  const ui=await mount({camera});
  await ui.elem("open-camera").listeners.click();
  assert.equal(ui.elem("analyze").disabled,true);
  assert.equal(ui.elem("camera-video").srcObject,camera.stream);
  ui.elem("capture-photo").listeners.click();
  assert.equal(camera.tracks[0].stopped,true);
  assert.equal(ui.elem("camera-panel").hidden,true);
  assert.equal(ui.elem("analyze").disabled,false);
  assert.equal(ui.fetchCalls.length,1, "capture must not upload automatically");
  await ui.elem("analyze").listeners.click();
  const call=ui.fetchCalls.find(x=>x.url.startsWith("/predict"));
  assert.equal(call.options.body.parts[0][1].name,"camera-capture.jpg");
  assert.equal(call.options.body.parts[0][1].type,"image/jpeg");
});

test("cancelled pending permission cannot restart the camera", async()=>{
  const camera=cameraFixture();
  let grant;
  camera.getUserMedia=()=>new Promise(resolve=>{grant=resolve;});
  const ui=await mount({camera});
  const opening=ui.elem("open-camera").listeners.click();
  ui.elem("close-camera").listeners.click();
  grant(camera.stream);
  await opening;
  assert.equal(camera.tracks[0].stopped,true);
  assert.equal(ui.elem("camera-panel").hidden,true);
  assert.equal(ui.elem("camera-video").srcObject,null);
});

test("permission denial and unsupported camera retain upload fallback", async()=>{
  const camera=cameraFixture();
  camera.getUserMedia=async()=>{throw {name:"NotAllowedError"};};
  const ui=await mount({camera});
  selectFile(ui,validFile);
  await ui.elem("open-camera").listeners.click();
  assert.match(ui.elem("camera-status").textContent,/denied/);
  assert.equal(ui.elem("analyze").disabled,false);
  const unavailable=await mount();
  await unavailable.elem("open-camera").listeners.click();
  assert.match(unavailable.elem("camera-status").textContent,/unavailable/);
});

test("late photo encoding cannot restore cleared selection", async()=>{
  const camera=cameraFixture();
  let encode;
  camera.canvas.toBlob=callback=>{encode=callback;};
  const ui=await mount({camera});
  await ui.elem("open-camera").listeners.click();
  ui.elem("capture-photo").listeners.click();
  ui.elem("clear-selection").listeners.click();
  encode({size:2048});
  assert.equal(ui.elem("analyze").disabled,true);
  assert.equal(ui.elem("preview").src,undefined);
  assert.equal(camera.tracks[0].stopped,true);
});

test("native phone photo uses existing validation and inference", async()=>{
  const ui=await mount();
  ui.elem("camera-file").files=[validFile];
  ui.elem("camera-file").listeners.change();
  await ui.elem("analyze").listeners.click();
  assert.equal(ui.fetchCalls.at(-1).options.body.parts[0][1],validFile);
  ui.elem("camera-file").files=[{name:"photo.heic",type:"image/heic",size:2000}];
  ui.elem("camera-file").listeners.change();
  assert.equal(ui.elem("analyze").disabled,true);
});

test("failed capture keeps camera available for retry", async()=>{
  const camera=cameraFixture();
  camera.canvas.toBlob=callback=>callback(null);
  const ui=await mount({camera});
  await ui.elem("open-camera").listeners.click();
  ui.elem("capture-photo").listeners.click();
  assert.match(ui.elem("camera-status").textContent,/Capture failed/);
  assert.equal(ui.elem("capture-photo").disabled,false);
  assert.equal(ui.elem("analyze").disabled,true);
  ui.elem("close-camera").listeners.click();
  assert.equal(camera.tracks[0].stopped,true);
});

test("selecting an upload closes the live camera and uses the upload", async()=>{
  const camera=cameraFixture();
  const ui=await mount({camera});
  await ui.elem("open-camera").listeners.click();
  selectFile(ui,validFile);
  assert.equal(camera.tracks[0].stopped,true);
  assert.equal(ui.elem("camera-video").srcObject,null);
  await ui.elem("analyze").listeners.click();
  assert.equal(ui.fetchCalls.at(-1).options.body.parts[0][1],validFile);
});
