# cupystorm — web (Pyodide)

Browser version of the 2D turbulence DNS solver. The unmodified NumPy/SciPy
solver (`cupystorm/turbo_simulator.py` + `turbo_wrapper.py`) runs inside
[Pyodide](https://pyodide.org) (CPython on WebAssembly); the PySide6 GUI is
replaced by an HTML/canvas front-end with the same controls, colormaps,
keyboard shortcuts (V/C/N/R/K/L/S/U) and mouse forcing.

## Setup

```bash
./install-pyodide.sh   # download the Pyodide distribution into web/pyodide/
./sync-sources.sh      # copy the solver sources into web/py/cupystorm/
```

Both target directories are gitignored; re-run `sync-sources.sh` whenever the
solver sources change.

## Run

Serve this directory over HTTP (Pyodide cannot load over `file://`):

```bash
python3 -m http.server 8000
# open http://localhost:8000/
```

## Files

- `index.html` / `styles.css` / `app.js` — front-end; `app.js` boots Pyodide,
  loads NumPy + SciPy, mounts the solver sources into the Pyodide FS and
  drives the run loop (time-budgeted `step_block` per animation frame).
- `web_sim.py` — Qt-free glue executed inside Pyodide: `WebSim` wraps
  `DnsSimulator`, ports the display normalization / colormap LUTs /
  per-mode Re defaults from `turbo_gui.py`+`turbo_logic.py`+`turbo_colors.py`
  (which cannot be imported in the browser since they import PySide6), and
  forces `scipy.fft` to a single worker (no threads under WASM).
- `test_node.mjs` — headless smoke test: `node test_node.mjs` runs the solver
  inside Pyodide under Node and checks stepping, rendering and mode switches.
