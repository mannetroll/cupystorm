// Headless smoke test: run the DNS solver inside Pyodide under Node.
//   node test_node.mjs
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { loadPyodide } from "./pyodide/pyodide.mjs";

const here = dirname(fileURLToPath(import.meta.url));

const pyodide = await loadPyodide({ indexURL: join(here, "pyodide") });
await pyodide.loadPackage(["numpy", "scipy"]);

pyodide.FS.mkdirTree("/home/pyodide/cupystorm");
const mount = (fsPath, localPath) =>
  pyodide.FS.writeFile(fsPath, readFileSync(join(here, localPath), "utf8"));
mount("/home/pyodide/cupystorm/__init__.py", "py/cupystorm/__init__.py");
mount("/home/pyodide/cupystorm/turbo_simulator.py", "py/cupystorm/turbo_simulator.py");
mount("/home/pyodide/cupystorm/turbo_wrapper.py", "py/cupystorm/turbo_wrapper.py");
mount("/home/pyodide/web_sim.py", "web_sim.py");

const websim = pyodide.pyimport("web_sim");

for (const mode of ["pao", "kolmo", "mouse"]) {
  const t0 = performance.now();
  const sim = websim.WebSim(128, mode);
  if (mode === "mouse") sim.apply_mouse_force(90, 90, true);
  const steps = sim.step_block(10, 60000);
  if (mode === "mouse") sim.apply_mouse_force(90, 90, false);
  const frame = sim.render();
  const buf = frame.getBuffer("u8");
  const st = JSON.parse(sim.status_json());
  const ms = ((performance.now() - t0) / steps).toFixed(1);
  console.log(
    `[${mode}] steps=${steps} t=${st.t.toFixed(4)} dt=${st.dt.toExponential(2)} ` +
      `sig=${st.sig.toFixed(2)} pal=${st.pal?.toFixed(6)} frame=${buf.data.length} ` +
      `(expect ${st.w * st.h * 4}) ~${ms}ms/step`
  );
  if (buf.data.length !== st.w * st.h * 4) throw new Error("frame size mismatch");
  if (!Number.isFinite(st.t) || !Number.isFinite(st.dt)) throw new Error("non-finite state");
  buf.release();
  frame.destroy();
  sim.destroy();
}

// exercise parameter changes on one instance
const sim = websim.WebSim(128, "pao");
sim.set_cmap("Viridis");
sim.set_variable(4); // streamfunction
sim.step_block(3, 60000);
sim.render().destroy();
sim.set_mode("circle");
sim.step_block(3, 60000);
sim.render().destroy();
console.log("after circle:", sim.status_json());
sim.destroy();

console.log("OK");
