// app.js — browser front-end for the cupystorm DNS solver running in Pyodide.
// Mirrors the PySide6 GUI (turbo_gui.py / turbo_logic.py): mode buttons,
// variable/colormap/N/Re/K0/CFL selectors, status bar, mouse forcing and
// the V/C/N/R/K/L/S/U shortcuts.

const PY_PACKAGE_FILES = ["__init__.py", "turbo_simulator.py", "turbo_wrapper.py"];

// Per-mode Reynolds number (mirrors MODE_RE in turbo_gui.py; web_sim.py
// applies it on the Python side — this copy only syncs the Re selector).
const MODE_RE = {
  pao: 10000, highh: 10000, rain: 10000, circle: 5000, mouse: 10000,
  kolmo: 20000, tg: 10000, merge: 25000, bickley: 8000, vortices: 4000,
};

const STEP_BUDGET_MS = 45; // max solver time per animation frame

const $ = (id) => document.getElementById(id);
const canvas = $("view");
const ctx = canvas.getContext("2d");
const statusEl = $("status");
const loadingEl = $("loading");
const loadingText = $("loading-text");
const appEl = $("app");

let sim = null;
let running = false;
let rafId = 0;
let currentMode = "pao";
let updateSteps = 5;
let maxSteps = 1e5;
let autoReset = false;
let imageData = null;
let lastStatus = null;
let simStartTime = performance.now();
let simStartIter = 0;

// ----------------------------------------------------------------------
// boot
// ----------------------------------------------------------------------
async function boot() {
  loadingText.textContent = "Loading Pyodide…";
  const pyodide = await loadPyodide({
    indexURL: new URL("pyodide/", location.href).href,
  });

  loadingText.textContent = "Loading NumPy + SciPy…";
  await pyodide.loadPackage(["numpy", "scipy"]);

  loadingText.textContent = "Mounting solver sources…";
  pyodide.FS.mkdirTree("/home/pyodide/cupystorm");
  for (const name of PY_PACKAGE_FILES) {
    const text = await (await fetch(`py/cupystorm/${name}`)).text();
    pyodide.FS.writeFile(`/home/pyodide/cupystorm/${name}`, text);
  }
  const glue = await (await fetch("web_sim.py")).text();
  pyodide.FS.writeFile("/home/pyodide/web_sim.py", glue);

  loadingText.textContent = "Initializing DNS state…";
  await nextPaint();
  // Read all initial parameters from the form controls: browsers restore
  // previous form state on reload without firing change events, so the
  // selects are the source of truth.
  const websim = pyodide.pyimport("web_sim");
  sim = websim.WebSim(
    parseInt($("n-select").value, 10),
    currentMode,
    parseFloat($("re-select").value),
    parseFloat($("k0-select").value),
    parseFloat($("cfl-select").value)
  );
  sim.set_variable(parseInt($("var-select").value, 10));
  sim.set_cmap($("cmap-select").value);
  updateSteps = parseInt($("update-select").value, 10);
  maxSteps = parseFloat($("steps-select").value);
  autoReset = $("auto-reset").checked;

  wireUi();
  setActiveModeButton(currentMode);
  loadingEl.hidden = true;
  appEl.hidden = false;
  drawFrame();
  start();
}

function nextPaint() {
  return new Promise((resolve) =>
    requestAnimationFrame(() => requestAnimationFrame(resolve))
  );
}

// ----------------------------------------------------------------------
// run loop
// ----------------------------------------------------------------------
function start() {
  if (running || !sim) return;
  running = true;
  resetFpsCounters();
  updateRunButtons();
  rafId = requestAnimationFrame(tick);
}

function stop() {
  running = false;
  cancelAnimationFrame(rafId);
  updateRunButtons();
}

function tick() {
  if (!running) return;
  sim.step_block(updateSteps, STEP_BUDGET_MS);
  drawFrame();

  const st = lastStatus;
  if (st) {
    if (st.lowSigStop) {
      stop();
      console.log("Field decayed (sigma < 1) — simulation stopped.");
      return;
    }
    if (st.iter >= maxSteps) {
      if (autoReset) {
        sim.reset();
        resetFpsCounters();
      } else {
        stop();
        console.log("Max steps reached — simulation stopped (auto-reset OFF).");
        return;
      }
    }
  }
  rafId = requestAnimationFrame(tick);
}

function resetFpsCounters() {
  simStartTime = performance.now();
  simStartIter = sim ? JSON.parse(sim.status_json()).iter : 0;
}

function updateRunButtons() {
  $("start-btn").disabled = running;
  $("stop-btn").disabled = !running;
}

// ----------------------------------------------------------------------
// rendering / status
// ----------------------------------------------------------------------
function drawFrame() {
  if (!sim) return;
  const frame = sim.render();
  const buf = frame.getBuffer("u8clamped");
  const st = JSON.parse(sim.status_json());
  lastStatus = st;

  if (canvas.width !== st.w || canvas.height !== st.h) {
    canvas.width = st.w;
    canvas.height = st.h;
    imageData = null;
  }
  if (!imageData) imageData = ctx.createImageData(st.w, st.h);
  imageData.data.set(buf.data);
  buf.release();
  frame.destroy();
  ctx.putImageData(imageData, 0, 0);
  updateStatusLine(st);
}

function updateStatusLine(st) {
  const elapsed = (performance.now() - simStartTime) / 1000;
  const steps = st.iter - simStartIter;
  const fps = elapsed > 0 && steps > 0 ? (steps / elapsed).toFixed(2) : "N/A";
  const pal = st.pal != null ? (10000 * st.pal).toFixed(1) : "N/A";
  const cells = [
    `FPS: ${fps}`,
    `pal/Zkmax²: ${pal}`,
    `σ: ${Math.trunc(st.sig)}`,
    `Iter: ${st.iter}`,
    `T: ${st.t.toFixed(3)}`,
    `dt: ${st.dt.toFixed(6)}`,
    `${(elapsed / 60).toFixed(1)} min`,
    `Visc: ${st.visc.toPrecision(6)}`,
  ];
  statusEl.innerHTML = cells.map((c) => `<span>${c}</span>`).join("");
}

// ----------------------------------------------------------------------
// UI wiring
// ----------------------------------------------------------------------
// Stop, show a notice, run the (blocking) re-initialization, then restore
// the previous run state — mirrors TurboLogic.on_reset_clicked.
async function reinit(fn) {
  const wasRunning = running;
  stop();
  statusEl.textContent = "Re-initializing…";
  await nextPaint();
  fn();
  resetFpsCounters();
  drawFrame();
  if (wasRunning) start();
}

function setActiveModeButton(mode) {
  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  canvas.classList.toggle("mouse-mode", mode === "mouse");
}

function syncReSelect(mode) {
  const re = MODE_RE[mode];
  if (re == null) return;
  const sel = $("re-select");
  const txt = String(re);
  if (![...sel.options].some((o) => o.value === txt)) {
    sel.add(new Option(txt, txt));
  }
  sel.value = txt;
}

function wireUi() {
  document.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      currentMode = btn.dataset.mode;
      setActiveModeButton(currentMode);
      syncReSelect(currentMode);
      reinit(() => sim.set_mode(currentMode));
    });
  });

  $("start-btn").addEventListener("click", start);
  $("stop-btn").addEventListener("click", stop);
  $("reset-btn").addEventListener("click", () => reinit(() => sim.reset()));
  $("save-btn").addEventListener("click", saveFrame);

  $("n-select").addEventListener("change", (e) =>
    reinit(() => sim.set_n(parseInt(e.target.value, 10)))
  );
  $("re-select").addEventListener("change", (e) =>
    reinit(() => sim.set_re(parseFloat(e.target.value)))
  );
  $("k0-select").addEventListener("change", (e) =>
    reinit(() => sim.set_k0(parseFloat(e.target.value)))
  );
  $("cfl-select").addEventListener("change", (e) =>
    reinit(() => sim.set_cfl(parseFloat(e.target.value)))
  );
  $("var-select").addEventListener("change", (e) => {
    sim.set_variable(parseInt(e.target.value, 10));
    drawFrame();
  });
  $("cmap-select").addEventListener("change", (e) => {
    sim.set_cmap(e.target.value);
    drawFrame();
  });
  $("update-select").addEventListener("change", (e) => {
    updateSteps = parseInt(e.target.value, 10);
  });
  $("steps-select").addEventListener("change", (e) => {
    maxSteps = parseFloat(e.target.value);
  });
  $("auto-reset").addEventListener("change", (e) => {
    autoReset = e.target.checked;
  });

  wireMouseForce();
  wireShortcuts();
}

function saveFrame() {
  const varName = $("var-select").selectedOptions[0].textContent;
  const cmapName = $("cmap-select").value;
  canvas.toBlob((blob) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `cupystorm_${varName}_${cmapName}.png`;
    a.click();
    URL.revokeObjectURL(a.href);
  }, "image/png");
}

// --- mouse forcing (mode "mouse") --------------------------------------
function wireMouseForce() {
  let dragging = false;
  let last = null;

  const simXY = (e) => {
    const r = canvas.getBoundingClientRect();
    const x = Math.floor(((e.clientX - r.left) * canvas.width) / r.width);
    const y = Math.floor(((e.clientY - r.top) * canvas.height) / r.height);
    return [
      Math.min(Math.max(x, 0), canvas.width - 1),
      Math.min(Math.max(y, 0), canvas.height - 1),
    ];
  };

  canvas.addEventListener("pointerdown", (e) => {
    if (currentMode !== "mouse" || !sim) return;
    dragging = true;
    canvas.setPointerCapture(e.pointerId);
    last = simXY(e);
    sim.apply_mouse_force(last[0], last[1], true);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    last = simXY(e);
    sim.apply_mouse_force(last[0], last[1], true);
  });
  const release = (e) => {
    if (!dragging) return;
    dragging = false;
    if (last) sim.apply_mouse_force(last[0], last[1], false);
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
}

// --- keyboard shortcuts (V/C/N/R/K/L/S/U, as in the Qt GUI) -------------
function wireShortcuts() {
  const keyToSelect = {
    v: "var-select", c: "cmap-select", n: "n-select", r: "re-select",
    k: "k0-select", l: "cfl-select", s: "steps-select", u: "update-select",
  };
  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const tag = e.target.tagName;
    if (tag === "SELECT" || tag === "INPUT" || tag === "TEXTAREA") return;
    const id = keyToSelect[e.key.toLowerCase()];
    if (!id) return;
    const sel = $(id);
    sel.selectedIndex = (sel.selectedIndex + 1) % sel.options.length;
    sel.dispatchEvent(new Event("change"));
    e.preventDefault();
  });
}

boot().catch((err) => {
  console.error(err);
  loadingText.textContent = `Failed to load: ${err.message ?? err}`;
});
