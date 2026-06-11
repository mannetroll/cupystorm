# web_sim.py
# Qt-free glue between the cupystorm DNS solver and the browser UI (Pyodide).
#
# Ports the non-Qt parts of turbo_gui.py / turbo_logic.py / turbo_colors.py:
#   - colormap LUTs (turbo_colors.py imports PySide6, so the stops live here too)
#   - display normalization (mu +/- k*sigma) and the low-sigma auto-stop
#   - circle forcing driven per step, mouse forcing, per-mode Re defaults
#   - palinstrophy/enstrophy grain metric for the status bar
import colorsys
import json
import math
import time
from contextlib import contextmanager

import numpy as np

from cupystorm import turbo_simulator as dns_all
import cupystorm.turbo_wrapper as _tw
from cupystorm.turbo_wrapper import DnsSimulator

# scipy.fft worker threads do not exist under single-threaded WASM:
# pocketfft raises "thread constructor failed" when workers > 1. The solver
# passes workers=S.fft_workers (4) explicitly, so force every scipy.fft
# transform to workers=1 and neutralize the set_workers context manager.
import scipy.fft as _spfft


@contextmanager
def _no_workers(_n):
    yield


_spfft.set_workers = _no_workers


def _single_worker(fn):
    def wrapped(*args, **kwargs):
        if kwargs.get("workers") is not None:
            kwargs["workers"] = 1
        return fn(*args, **kwargs)
    return wrapped


for _name in ("fft", "ifft", "fft2", "ifft2", "fftn", "ifftn",
              "rfft", "irfft", "rfft2", "irfft2", "rfftn", "irfftn"):
    setattr(_spfft, _name, _single_worker(getattr(_spfft, _name)))

DISPLAY_NORM_K_STD = 4.0
DEFAULT_FORCE_AMP = 0.5
DEFAULT_FORCE_SIGMA = 8.0
DEFAULT_CMAP_NAME = "Magma"

# Per-mode Reynolds number (mirrors MODE_RE in turbo_gui.py).
MODE_RE = {
    "pao": 10000,
    "highh": 10000,
    "rain": 10000,
    "circle": 5000,
    "mouse": 10000,
    "kolmo": 1000,
    "tg": 10000,
    "merge": 25000,
    "bickley": 1000,
    "vortices": 4000,
}

# Per-mode default CFL: the sustained-forcing modes run fine at 0.75; the
# initial-condition-only modes need 0.3 to stay accurate through the decay.
MODE_CFL = {
    "pao": 0.75,
    "highh": 0.75,
    "rain": 0.75,
    "circle": 0.75,
    "mouse": 0.75,
    "kolmo": 0.5,
    "tg": 0.5,
    "merge": 0.5,
    "bickley": 0.5,
    "vortices": 0.5,
}

VALID_MODES = tuple(MODE_RE.keys())


# ----------------------------------------------------------------------
# Colormaps (same stops as turbo_colors.py, built with np.interp)
# ----------------------------------------------------------------------
def _lut_from_stops(stops):
    pos = np.array([p for p, _ in stops], dtype=np.float64)
    cols = np.array([c for _, c in stops], dtype=np.float64)
    x = np.linspace(0.0, 1.0, 256)
    lut = np.stack([np.interp(x, pos, cols[:, i]) for i in range(3)], axis=1)
    return lut.round().clip(0, 255).astype(np.uint8)


def _make_gray_lut():
    g = np.arange(256, dtype=np.uint8)
    return np.stack([g, g, g], axis=1)


def _make_fire_lut():
    """Approximate 'fire' palette via HSL ramp: red -> yellow, brightening."""
    lut = np.zeros((256, 3), dtype=np.uint8)
    for x in range(256):
        h = (85.0 * (x / 255.0)) / 360.0
        l = min(1.0, x / 128.0)
        r, g, b = colorsys.hls_to_rgb(h, l, 1.0)
        lut[x] = (int(r * 255), int(g * 255), int(b * 255))
    return lut


def _make_doom_fire_lut():
    key_colors = [
        (0, 0, 0), (7, 7, 7), (31, 7, 7), (47, 15, 7), (71, 15, 7),
        (87, 23, 7), (103, 31, 7), (119, 31, 7), (143, 39, 7), (159, 47, 7),
        (175, 63, 7), (191, 71, 7), (199, 71, 7), (223, 79, 7), (223, 87, 7),
        (223, 87, 7), (215, 95, 7), (215, 95, 7), (215, 103, 15), (207, 111, 15),
        (207, 119, 15), (207, 127, 15), (207, 135, 23), (199, 135, 23), (199, 143, 23),
        (199, 151, 31), (191, 159, 31), (191, 159, 31), (191, 167, 39), (191, 167, 39),
        (191, 175, 47), (183, 175, 47), (183, 183, 47), (183, 183, 55), (207, 207, 111),
        (223, 223, 159), (239, 239, 199), (255, 255, 255),
    ]
    n = len(key_colors)
    stops = [(i / (n - 1), c) for i, c in enumerate(key_colors)]
    return _lut_from_stops(stops)


_CMAP_STOPS = {
    "Inferno": [(0.0, (0, 0, 4)), (0.25, (87, 15, 109)), (0.50, (187, 55, 84)),
                (0.75, (249, 142, 8)), (1.0, (252, 255, 164))],
    "Ocean": [(0.0, (0, 5, 30)), (0.25, (0, 60, 125)), (0.50, (0, 140, 190)),
              (0.75, (0, 200, 175)), (1.0, (180, 245, 240))],
    "Viridis": [(0.0, (68, 1, 84)), (0.25, (59, 82, 139)), (0.50, (33, 145, 140)),
                (0.75, (94, 201, 98)), (1.0, (253, 231, 37))],
    "Plasma": [(0.0, (13, 8, 135)), (0.25, (126, 3, 167)), (0.50, (203, 71, 119)),
               (0.75, (248, 149, 64)), (1.0, (240, 249, 33))],
    "Magma": [(0.0, (0, 0, 4)), (0.25, (73, 18, 99)), (0.50, (150, 50, 98)),
              (0.75, (226, 102, 73)), (1.0, (252, 253, 191))],
    "Turbo": [(0.0, (48, 18, 59)), (0.25, (31, 120, 180)), (0.50, (78, 181, 75)),
              (0.75, (241, 208, 29)), (1.0, (133, 32, 26))],
    "Cividis": [(0.00, (0, 34, 77)), (0.25, (0, 68, 117)), (0.50, (60, 111, 130)),
                (0.75, (147, 147, 95)), (1.00, (250, 231, 33))],
    "Jet": [(0.00, (0, 0, 131)), (0.35, (0, 255, 255)), (0.66, (255, 255, 0)),
            (1.00, (128, 0, 0))],
    "Coolwarm": [(0.00, (59, 76, 192)), (0.25, (127, 150, 203)), (0.50, (217, 217, 217)),
                 (0.75, (203, 132, 123)), (1.00, (180, 4, 38))],
    "RdBu": [(0.00, (103, 0, 31)), (0.25, (178, 24, 43)), (0.50, (247, 247, 247)),
             (0.75, (33, 102, 172)), (1.00, (5, 48, 97))],
}

COLOR_MAPS = {
    "Gray": _make_gray_lut(),
    "Inferno": _lut_from_stops(_CMAP_STOPS["Inferno"]),
    "Ocean": _lut_from_stops(_CMAP_STOPS["Ocean"]),
    "Viridis": _lut_from_stops(_CMAP_STOPS["Viridis"]),
    "Plasma": _lut_from_stops(_CMAP_STOPS["Plasma"]),
    "Magma": _lut_from_stops(_CMAP_STOPS["Magma"]),
    "Turbo": _lut_from_stops(_CMAP_STOPS["Turbo"]),
    "Fire": _make_fire_lut(),
    "Doom": _make_doom_fire_lut(),
    "Cividis": _lut_from_stops(_CMAP_STOPS["Cividis"]),
    "Jet": _lut_from_stops(_CMAP_STOPS["Jet"]),
    "Coolwarm": _lut_from_stops(_CMAP_STOPS["Coolwarm"]),
    "RdBu": _lut_from_stops(_CMAP_STOPS["RdBu"]),
}

# 256x4 RGBA LUTs for direct ImageData filling.
_RGBA_LUTS = {
    name: np.concatenate([lut, np.full((256, 1), 255, dtype=np.uint8)], axis=1)
    for name, lut in COLOR_MAPS.items()
}

CMAP_NAMES = list(COLOR_MAPS.keys())


class WebSim:
    """Browser-facing wrapper: one instance is driven from app.js."""

    def __init__(self, n: int = 256, mode: str = "pao",
                 re: float | None = None, k0: float = 10.0, cfl: float = 0.75):
        self.mode = mode if mode in VALID_MODES else "pao"
        self.cmap_name = DEFAULT_CMAP_NAME
        self.sig = 20.0
        self.mu = 128.0
        self.pal = None
        self.low_sig_stop = False
        self.f_hz = 0.1  # circle stirring frequency
        self.cx = self.cy = self.R = 0.0

        self.sim = DnsSimulator(
            n=int(n),
            re=float(re) if re is not None else float(MODE_RE[self.mode]),
            k0=float(k0),
            cfl=float(cfl),
            backend="cpu",
            mode=self.mode,
        )
        self.sim.set_variable(DnsSimulator.VAR_OMEGA)
        self.reset(self.mode)
        self._frame = None

    # --- geometry helpers -------------------------------------------------
    def _update_circle_geom(self):
        self.cx = 0.5 * (self.sim.px - 1)
        self.cy = 0.5 * (self.sim.py - 1)
        self.R = self.sim.py / 4.0

    def _sigma_full_from_base(self, sigma_base: float) -> float:
        S = self.sim.state
        return float(sigma_base) * (float(S.NX) / float(S.NX_full))

    # --- reset / parameter changes (mirror TurboLogic handlers) -----------
    def _post_init_nextdt(self):
        S = self.sim.state
        dns_all.dns_step2a(S)
        CFLM = dns_all.compute_cflm(S)
        if CFLM == 0.0:
            S.dt = 0.01
        else:
            S.dt = S.cflnum / (CFLM * math.pi)
        S.t = 0.0
        S.cn = 1.0
        S.cnm1 = 0.0
        self.sim.t = float(S.t)
        self.sim.dt = float(S.dt)
        self.sim.cn = float(S.cn)
        self.sim.iteration = 0

    def reset(self, mode: str | None = None):
        if mode is not None:
            if mode not in VALID_MODES:
                mode = "pao"
            self.mode = mode
        self.low_sig_stop = False
        self.sim.reset_field(mode=self.mode)
        if self.mode in ("pao", "highh", "tg", "merge"):
            self._post_init_nextdt()
        self._update_circle_geom()

    def set_mode(self, mode: str):
        re = MODE_RE.get(mode)
        if re is not None:
            self.sim.re = float(re)
        cfl = MODE_CFL.get(mode)
        if cfl is not None:
            self.sim.cfl = float(cfl)
        self.reset(mode)

    def set_n(self, n: int):
        self.sim.set_N(int(n))
        self.reset()

    def set_re(self, re: float):
        self.sim.re = float(re)
        self.reset()

    def set_k0(self, k0: float):
        self.sim.k0 = float(k0)
        self.reset()

    def set_cfl(self, cfl: float):
        self.sim.cfl = float(cfl)
        self.reset()

    def set_variable(self, var: int):
        self.sim.set_variable(int(var))

    def set_cmap(self, name: str):
        if name in _RGBA_LUTS:
            self.cmap_name = name

    # --- stepping ----------------------------------------------------------
    def step_block(self, max_steps: int, budget_ms: float) -> int:
        """Run up to max_steps DNS steps, stopping early after budget_ms."""
        deadline = time.monotonic() + float(budget_ms) / 1000.0
        done = 0
        while done < int(max_steps):
            t0 = self.sim.get_time()
            self.sim.step(1)
            done += 1
            if self.mode == "circle":
                theta = 2.0 * math.pi * self.f_hz * t0
                x = self.cx + self.R * math.cos(theta)
                y = self.cy - self.R * math.sin(theta)
                self.sim.set_body_force(
                    int(x), int(y),
                    amp=DEFAULT_FORCE_AMP,
                    sigma=self._sigma_full_from_base(DEFAULT_FORCE_SIGMA),
                    active=True,
                )
            if time.monotonic() >= deadline:
                break
        return done

    def apply_mouse_force(self, x: int, y: int, active: bool):
        if self.mode != "mouse":
            return
        px_max = int(self.sim.px) - 1
        py_max = int(self.sim.py) - 1
        x = min(max(int(x), 0), px_max)
        y = min(max(int(y), 0), py_max)
        self.sim.set_body_force(
            x, y,
            amp=DEFAULT_FORCE_AMP,
            sigma=self._sigma_full_from_base(DEFAULT_FORCE_SIGMA),
            active=bool(active),
        )

    # --- diagnostics ---------------------------------------------------------
    def _pal_over_ens_kmax2(self) -> float:
        """Palinstrophy/enstrophy metric (CPU port of TurboLogic.pal_over_ens_kmax2)."""
        S = self.sim.state
        band = S.om2
        P = band.real * band.real + band.imag * band.imag
        K2 = S.step3_K2
        NX_half = int(P.shape[1])

        if NX_half == 1:
            total = float(P.sum() - P[0, 0])
            if total <= 0.0:
                return 0.0
            kmax2 = float(K2.max())
            if kmax2 <= 0.0:
                return 0.0
            return float((K2 * P).sum()) / (total * kmax2)

        edge = P[:, 0].sum() + P[:, -1].sum()
        mid = P[:, 1:-1].sum() if NX_half > 2 else 0.0
        total = float(edge + 2.0 * mid - P[0, 0])
        if total <= 0.0:
            return 0.0
        kmax2 = float(K2.max())
        if kmax2 <= 0.0:
            return 0.0
        edge_p = (K2[:, 0] * P[:, 0]).sum() + (K2[:, -1] * P[:, -1]).sum()
        mid_p = (K2[:, 1:-1] * P[:, 1:-1]).sum() if NX_half > 2 else 0.0
        return float(edge_p + 2.0 * mid_p) / (total * kmax2)

    # --- rendering --------------------------------------------------------
    def render(self):
        """Return a flat RGBA uint8 array (h*w*4) for the canvas ImageData."""
        pix = self.sim.get_frame_pixels()
        pf = pix.astype(np.float32, copy=False)

        if self.mode == "rain":
            self.sig = 5.0
        elif self.mode in ("circle", "mouse"):
            self.sig = 10.0
        else:
            self.sig = float(pf.std())

        if self.sig < 1.0 and self.mode in ("pao", "highh"):
            self.low_sig_stop = True

        self.pal = self._pal_over_ens_kmax2()
        self.mu = float(pf.mean())
        k = DISPLAY_NORM_K_STD
        lo = self.mu - k * self.sig
        hi = self.mu + k * self.sig
        inv = 255.0 / (hi - lo) if (hi - lo) != 0.0 else 0.0
        p8 = ((pf - lo) * inv).round().clip(0.0, 255.0).astype(np.uint8)

        rgba = _RGBA_LUTS.get(self.cmap_name, _RGBA_LUTS["Gray"])[p8]
        self._frame = np.ascontiguousarray(rgba).reshape(-1)
        return self._frame

    def status_json(self) -> str:
        S = self.sim.state
        return json.dumps({
            "t": float(self.sim.get_time()),
            "dt": float(S.dt),
            "iter": int(self.sim.get_iteration()),
            "sig": float(self.sig),
            "pal": (float(self.pal) if self.pal is not None else None),
            "visc": float(S.visc),
            "w": int(self.sim.px),
            "h": int(self.sim.py),
            "mode": self.mode,
            "lowSigStop": bool(self.low_sig_stop),
        })
