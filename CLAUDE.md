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

# Run CLI simulator directly
uv run python -m cupystorm.turbo_simulator N Re K0 STEPS CFL BACKEND [MODE]
# Example: uv run python -m cupystorm.turbo_simulator 256 10000 10 1000 0.75 cpu pao

# Build package
python -m build

# Profiling
python -m cProfile -o profile.out -m cupystorm.turbo_simulator 256 10000 10 500 0.75 cpu
python -m snakeviz profile.out
```

No test suite or linter is configured; validation is done by running the simulator.

## Architecture

### Module Roles

- **`turbo_simulator.py`** (~2600 lines): Core DNS solver. Contains `DnsState` dataclass holding all simulation state, and the spectral time-integration pipeline.
- **`turbo_wrapper.py`**: `DnsSimulator` class — thin wrapper around `DnsState` for use by the GUI. Handles field rendering (`make_pixels_component`) and parameter cycling.
- **`turbo_gui.py`**: `MainWindow(QMainWindow, TurboLogic)` — PySide6 window with controls, colormaps, keyboard shortcuts (V/C/N/R/K/L/S/U), and mouse-force interaction via `ClickableLabel`.
- **`turbo_logic.py`**: `TurboLogic` mixin — timer loop, mode switching, diagnostics, PGM file export.
- **`turbo_colors.py`**: 12 built-in colormaps producing Qt color tables for 8-bit indexed rendering.

### DNS Time-Step Pipeline

Each call to `DnsSimulator.step()` runs three stages in `turbo_simulator.py`:

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

### Forcing Modes

Controlled by `MODE` parameter or GUI buttons:
- **`pao`**: PAO spectrum init, no sustained forcing (decaying turbulence)
- **`highh`**: High-k spectral forcing + Ekman/Rayleigh drag (sustained turbulence)
- **`rain`**: Random localized impulses at Hz=2.0, amp=0.25
- **`circle`**: Tangential forcing on a ring
- **`mouse`**: Interactive click-drag forcing

### Backend Selection

Auto-detection tries CuPy first, falls back to NumPy/SciPy. Pass `backend="cpu"`, `"gpu"`, or `"auto"` to `create_dns_state()`. FFT module is `scipy.fft` (CPU) or `cupyx.scipy.fft` (GPU).

## Key Parameters

| Parameter | Range | Default |
|-----------|-------|---------|
| N (grid size) | 128–32768 | 256 |
| Re (Reynolds number) | 10–1E9 | 10000 |
| K0 (peak wavenumber) | 2–90 | 10 |
| CFL | 0.05–0.95 | 0.75 |
