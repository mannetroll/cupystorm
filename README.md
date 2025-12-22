# 2D Turbulence Simulation (SciPy / CuPy)

A Direct Numerical Simulation (DNS) code for **2D homogeneous incompressible turbulence**

It supports:

- **SciPy / NumPy** for CPU runs
- **CuPy** (optional) for GPU acceleration on CUDA devices (e.g. RTX 3090)

### DNS solver
The solver includes:

- **PAO-style random-field initialization**
- **3/2 de-aliasing** in spectral space
- **Crank–Nicolson** time integration
- **CFL-based adaptive time stepping** (Δt updated from the current flow state)

### cupystormive GUI (PySide6)
Run an cupystormive window that:

- Displays the flow field as a live image (fast `Indexed8` palette rendering)
- Lets you switch displayed variable:
  - **U**, **V** (velocity components)
  - **K** (kinetic energy)
  - **Ω** (vorticity)
  - **φ** (stream function)
- Lets you switch **colormap** (several built-in palettes)
- Lets you change simulation parameters on the fly:
  - Grid size **N**
  - Reynolds number **Re**
  - Initial spectrum peak **K0**
  - CFL number **CFL**
  - Max steps / auto-reset limit
  - GUI update interval (how often to refresh the display)
- Includes force/initialization modes:
  - **PAO**: PAO spectrum init (no continuous forcing)
  - **Circle**: a forcing point moves on a circular trajectory (continuous forcing)
  - **Rain**: random “kick” forcing events (good for sustained turbulence)
  - **Mouse**: apply forcing by click/drag in the image

### Keyboard shortcuts
Single-key shortcuts (application-wide) for fast control:

- **V**: cycle variable
- **C**: cycle colormap
- **N**: cycle grid size
- **R**: cycle Reynolds number
- **K**: cycle K0
- **L**: cycle CFL
- **S**: cycle max steps
- **U**: cycle update interval

### Saving / exporting
From the GUI you can:

- **Save the current frame** as a PNG image
- **Dump full-resolution fields** to a folder as PGM images:
  - u-velocity, v-velocity, kinetic energy, vorticity

### Display scaling
To keep the GUI responsive for large grids, the displayed image is automatically upscaled/downscaled depending on `N`. The window is resized accordingly when you change `N`.

## Installation

### Using uv

From the project root:

    uv sync
    uv run python -m cupystorm.turbo_main


## The DNS with SciPy (384 x 384) Rain

![SciPy](https://raw.githubusercontent.com/mannetroll/cupystorm/main/window.png)


### Full CLI

    $ python -m cupystorm.turbo_simulator N Re K0 STEPS CFL BACKEND

Where:

- N       — grid size (e.g. 256, 512)
- Re      — Reynolds number (e.g. 10000)
- K0      — peak wavenumber of the energy spectrum
- STEPS   — number of time steps
- CFL     — target CFL number (e.g. 0.75)
- BACKEND — "cpu", "gpu", or "auto"

Examples:

    # CPU run (SciPy with 4 workers)
    $ python -m cupystorm.turbo_simulator 256 10000 10 1001 0.75 cpu

    # Auto-select backend (GPU if CuPy + CUDA are available)
    $ python -m cupystorm.turbo_simulator 256 10000 10 1001 0.75 auto


## Enabling GPU with CuPy (CUDA 13)

On a CUDA machine (e.g. RTX 3090):

1. Check that the driver/CUDA are available:

       $ nvidia-smi | head -n 3

2. Install CuPy into the uv environment:

       $ uv sync
       $ uv pip install cupy

3. Verify that CuPy sees the GPU:

       $ uv run python -c "import cupy as cp; x = cp.arange(5); print(x, x.device)"

4. Run in GPU mode:

       $ uv run python -m cupystorm.turbo_simulator 256 10000 10 1001 0.75 gpu

Or let the backend auto-detect:

       $ uv run python -m cupystorm.turbo_simulator 256 10000 10 1001 0.75 auto


## The DNS with CuPy (4096 x 4096)

![CuPy](https://raw.githubusercontent.com/mannetroll/cupystorm/main/window4096.png)


## Profiling

### cProfile (CPU)

    $ python -m cProfile -o turbo_simulator.prof -m cupystorm.turbo_simulator    

Inspect the results:

    $ python -m pstats turbo_simulator.prof
    # inside pstats:
    turbo_simulator.prof% sort time
    turbo_simulator.prof% stats 20


### GUI profiling with SnakeViz

Install SnakeViz:

    $ uv pip install snakeviz

Visualize the profile:

    $ snakeviz turbo_simulator.prof


### Memory & CPU profiling with Scalene (GUI)

Install Scalene:

    $ uv pip install scalene

Run with GUI report:

    $ scalene -m cupystorm.turbo_simulator 256 10000 10 201 0.75 cpu


### Memory & CPU profiling with Scalene (CLI only)

For a terminal-only summary:

    $ scalene --cli --cpu -m cupystorm.turbo_simulator 256 10000 10 201 0.75 cpu


## Project layout (key modules)

- `cupystorm/turbo_main.py`  
  PyQt6 GUI viewer; displays DNS fields (U, V, ω, kinetic) in real time.

- `cupystorm/turbo_simulator.py`  
  Headless CLI DNS solver:
  - PAO initialization (dns_pao_host_init)
  - FFT helpers (vfft_full_*)
  - STEP2A, STEP2B, STEP3
  - CFL-based time-step control (compute_cflm, next_dt)

- `cupystorm/turbo_wrapper.py`  
  Thin wrapper for programmatic use.

## one-liner

```
$ curl -LsSf https://astral.sh/uv/install.sh | sh
$ uv cache clean mannetroll-cupystorm
$ uv run --python 3.13 --with mannetroll-cupystorm==0.1.0 python -m cupystorm.turbo_main
```

## License

Copyright © Mannetroll
See the project repository for license details.
