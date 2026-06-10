# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**cupystorm** is a 2D homogeneous incompressible turbulence Direct Numerical Simulation (DNS) solver. It uses spectral methods (3/2 de-aliasing, Crank–Nicolson time integration, CFL-based adaptive timestepping) and supports both CPU (NumPy/SciPy) and GPU (CuPy/CUDA) backends. An interactive PySide6 GUI provides real-time visualization and parameter control.

## Commands

```bash
# Install (requires Python >=3.13, <3.14)
uv sync                        # CPU only
uv sync --extra cuda           # With CuPy GPU support

# Run GUI
uv run turbulence

# Run CLI simulator directly (entry-point alias: `uv run sim ...`)
uv run python -m cupystorm.turbo_simulator N Re K0 STEPS CFL BACKEND [MODE]
# Example: uv run python -m cupystorm.turbo_simulator 256 10000 10 1000 0.75 cpu pao
# All args positional & optional; BACKEND ∈ {cpu,gpu,auto}, MODE ∈ {pao,highh,rain,circle,mouse,kolmo,tg,merge,bickley,vortices}

# Build package
python -m build

# Profiling
python -m cProfile -o profile.out -m cupystorm.turbo_simulator 256 10000 10 500 0.75 cpu
python -m snakeviz profile.out
```

No test suite or linter is configured; validation is done by running the simulator.

## Architecture

`turbo_simulator.py` is a faithful structural port of a CUDA reference (`dns_all.cu`); the spectral formulas follow those kernels line-by-line. This explains the naming (`DnsState` mirrors `DnsDeviceState`) and the array layouts below — preserve them when editing.

### Module Roles

- **`turbo_simulator.py`** (~2600 lines): Core DNS solver. Contains `DnsState` dataclass holding all simulation state, and the spectral time-integration pipeline.
- **`turbo_wrapper.py`**: `DnsSimulator` class — thin wrapper around `DnsState` for use by the GUI. Handles field rendering (`make_pixels_component`) and parameter cycling.
- **`turbo_gui.py`**: `MainWindow(QMainWindow, TurboLogic)` — PySide6 window with controls, colormaps, keyboard shortcuts (V/C/N/R/K/L/S/U), and mouse-force interaction via `ClickableLabel`.
- **`turbo_logic.py`**: `TurboLogic` mixin — timer loop, mode switching, diagnostics, PGM file export.
- **`turbo_colors.py`**: 12 built-in colormaps producing Qt color tables for 8-bit indexed rendering.

### DNS Time-Step Pipeline

Each call to `DnsSimulator.step()` runs this pipeline in `turbo_simulator.py` (mirrors `run_dns` in the CUDA source: STEP2B → STEP3 → STEP2A → NEXTDT, then `T += dt_old`):

1. **`dns_step2b()`** — builds non-linear products from `ur_full`, forward FFTs → `uc_full`
2. **`dns_step3()`** — updates vorticity ω in spectral space (Crank–Nicolson + forcing injection)
3. **`dns_step2a()`** — de-aliasing, reshuffle compact↔full, inverse FFT → `ur_full`
4. **`next_dt()`** — CFL-based adaptive timestep from `compute_cflm()`

GPU acceleration uses CuPy `RawKernel` and `ElementwiseKernel` for the bottleneck operations (`STEP2B_MUL3_KERNEL`, `STEP3_UPDATE_KERNEL`, `STEP3_BUILD_UC_KERNEL`, `STEP2A_CROP_KERNEL`) and reuses cuFFT plans across steps.

### Key Spectral Arrays in `DnsState`

| Array | Description |
|-------|-------------|
| `uc`, `uc_full` | Vorticity in spectral space (compact N×N and 3/2-padded) |
| `ur`, `ur_full` | Velocity/vorticity in physical space (compact and 3/2-padded) |
| `alfa`, `gamma` | Wavenumber arrays kx, kz |
| `force_omega_hat` | Spectral body-force pattern |
| `rayleigh_alpha_k`, `highk_omega_hat` | High-k/Ekman drag and spectral forcing |

Memory layouts (kept identical to the CUDA port — important when indexing):
- `ur` (compact, physical): `(NZ, NX, 3)` AoS — `[z, x, comp]`
- `uc` (compact, spectral): `(NZ, NK, 3)` — `[z, kx, comp]`
- `ur_full` (3/2-padded, physical): `(3, NZ_full, NX_full)` SoA — `[comp, z, x]`
- `uc_full` (3/2-padded, spectral): `(3, NZ_full, NK_full)` SoA

### Modes

Controlled by `MODE` parameter or GUI buttons. Selected in two places: the init
dispatch in `create_dns_state` (which field to start from) and `_apply_mode_defaults`
(which forcing knobs to set). Adding a mode means a branch in each, plus registering
the name in the `Literal` hints, `main()` validation, and the GUI buttons/`MODE_RE`.

Sustained-forcing modes (built via `force_omega_hat` / `highk_omega_hat` / `rayleigh_alpha_k`):
- **`pao`**: PAO spectrum init, no sustained forcing (decaying turbulence)
- **`highh`**: High-k spectral forcing + Ekman/Rayleigh drag (sustained turbulence)
- **`rain`**: Random localized impulses at Hz=2.0, amp=0.25
- **`circle`**: Steady tangential forcing on a ring
- **`mouse`**: Interactive click-drag forcing
- **`kolmo`**: Steady Kolmogorov forcing `f_x = A·sin(kf·z)` (kf≈K0) + weak large-scale drag; starts from a smooth low-k random seed so the shear transitions to turbulence

Initial-condition-only modes (analytic field via `_load_vorticity_field`, then free decay):
- **`tg`**: Taylor–Green vortex array `ω = A·cos(kX)·cos(kZ)`
- **`merge`**: two like-sign Gaussian vortices that co-rotate and merge
- **`bickley`**: unstable Bickley jet that rolls up into a periodic row of vortices
- **`vortices`**: random vortex gas that merges into decaying 2D turbulence

Body-force plumbing: `_update_force_omega_hat` builds a Gaussian blob into
`force_u_full`/`force_w_full`; `_force_fields_to_omega_hat` is the FFT+curl tail that
turns any custom real-space force field into `force_omega_hat` (used by `circle`/`kolmo`
so their steady patterns are not overwritten by the blob builder). `next_dt` caps `dt`
growth at 1.1×/step so forcing-from-rest modes don't blow up on the first adaptation.

### Backend Selection

Auto-detection tries CuPy first, falls back to NumPy/SciPy. Pass `backend="cpu"`, `"gpu"`, or `"auto"` to `create_dns_state()`. FFT module is `scipy.fft` (CPU) or `cupyx.scipy.fft` (GPU).

## Key Parameters

| Parameter | Range | Default |
|-----------|-------|---------|
| N (grid size) | 128–32768 | 256 |
| Re (Reynolds number) | 10–1E9 | 10000 |
| K0 (peak wavenumber) | 2–90 | 10 |
| CFL | 0.05–0.95 | 0.75 |
