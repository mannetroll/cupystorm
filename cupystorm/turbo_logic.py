# turbo_logic.py
import math
import os
import time
from typing import Optional

from PySide6.QtCore import QTimer
import numpy as np

from cupystorm import turbo_simulator as dns_all
from cupystorm.turbo_colors import DEFAULT_FORCE_AMP, DEFAULT_FORCE_SIGMA
from cupystorm.turbo_wrapper import DnsSimulator


class TurboLogicMixin:
    # These are provided by the concrete MainWindow class (or another mixin),
    # but we declare them here so type-checkers stop complaining.
    sim: DnsSimulator
    timer: QTimer
    _force_mode: str

    _sim_start_time: float
    _sim_start_iter: int
    _status_update_counter: int
    _update_intervall: int

    _force_dragging: bool
    _force_last_xy: Optional[tuple[int, int]]

    def _logic_init_defaults(self) -> None:
        # Force/init mode selector
        #   "pao"   : PAO spectrum init (no auto force)
        #   "circle": circle stirring (auto force)
        #   "rain"  : random body-force kicks (auto force)
        #   "mouse" : mouse drag force only (no auto force)
        self._force_mode = "rain"

        # --- FPS from simulation start ---
        self._sim_start_time = time.time()
        self._sim_start_iter = self.sim.get_iteration()

        # --- random small-vortex injector ("Rain" mode) ---
        self._inj_enabled = True
        self._inj_f_hz = 2.0
        self._inj_sigma = 4.0
        self._inj_amp0 = DEFAULT_FORCE_AMP
        self._inj_duration_steps = 1
        self._inj_next_t = self.sim.get_time()
        self._inj_off_iter = -1
        self._inj_last = None
        self._rng = np.random.default_rng(1)

        # circle forcing params
        self.f_hz = 0.1
        self.cx = 0.5 * (self.sim.px - 1)
        self.cy = 0.5 * (self.sim.py - 1)
        self.R = self.sim.py / 4.0

        # start in Rain mode: reset injector schedule (no constant force)
        self._injector_reset()
        self.sim.state.force_active = False
        self.sim.state.force_dirty = True

    def _update_run_buttons(self) -> None:
        running = self.timer.isActive()
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def on_start_clicked(self) -> None:
        self._sim_start_time = time.time()
        self._sim_start_iter = self.sim.get_iteration()
        if not self.timer.isActive():
            self.timer.start()
        self._update_run_buttons()

    def on_stop_clicked(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
        self._update_run_buttons()

    def on_step_clicked(self) -> None:
        self.sim.step()
        pixels = self.sim.get_frame_pixels()
        self._update_image(pixels)
        self._update_status(self.sim.get_time(), self.sim.get_iteration(), fps=None)

    def on_reset_clicked(self) -> None:
        was_running = self.timer.isActive()
        if was_running:
            self.on_stop_clicked()

        if self._force_mode == "pao":
            self.sim.reset_field()
            dns_all.dns_pao_host_init(self.sim.state)
            self._post_init_nextdt()

            self.sim.state.force_active = False
            self.sim.state.force_dirty = True

            self._reset_gui_after_init()

        elif self._force_mode == "circle":
            self.sim.reset_field()

            self.cx = 0.5 * (self.sim.px - 1)
            self.cy = 0.5 * (self.sim.py - 1)
            self.R = self.sim.py / 4.0

            x = self.cx + self.R * math.cos(0)
            y = self.cy + self.R * math.sin(0)

            self.sim.set_body_force(
                int(x),
                int(y),
                amp=DEFAULT_FORCE_AMP,
                sigma=DEFAULT_FORCE_SIGMA,
                active=True,
            )

            self._reset_gui_after_init()

        elif self._force_mode == "rain":
            self.sim.reset_field()
            self._injector_reset()
            self._reset_gui_after_init()

        elif self._force_mode == "mouse":
            self.sim.reset_field()
            self.sim.state.force_active = False
            self.sim.state.force_dirty = True
            self._reset_gui_after_init()

        if was_running:
            self.on_start_clicked()

    def _post_init_nextdt(self) -> None:
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

    def _reset_gui_after_init(self) -> None:
        self._sim_start_time = time.time()
        self._sim_start_iter = self.sim.get_iteration()
        self._status_update_counter = 0
        self._update_image(self.sim.get_frame_pixels())
        self._update_status(self.sim.get_time(), self.sim.get_iteration(), None)

    def on_init_pao_clicked(self) -> None:
        was_running = self.timer.isActive()
        self.on_stop_clicked()

        self.sim.reset_field()
        dns_all.dns_pao_host_init(self.sim.state)
        self._post_init_nextdt()

        self.sim.state.force_active = False
        self.sim.state.force_dirty = True

        self._force_mode = "pao"
        self._reset_gui_after_init()
        if was_running:
            self.on_start_clicked()

    def on_init_pao_ekman_clicked(self) -> None:
        # Start PAO + large-scale Rayleigh/Ekman drag in one click.
        # (Keeps existing PAO button behaviour untouched.)
        self.on_stop_clicked()

        # Default large-scale drag params (user can tune in code later)
        self.sim.rayleigh_alpha0 = 0.1
        self.sim.rayleigh_k_cut = 6.0
        self.sim.rayleigh_p = 8.0

        self.sim.reset_field()
        dns_all.dns_pao_host_init(self.sim.state)
        self._post_init_nextdt()

        self.sim.state.force_active = False
        self.sim.state.force_dirty = True

        self._force_mode = "pao"
        self._reset_gui_after_init()
        self.on_start_clicked()

    def on_init_circle_clicked(self) -> None:
        was_running = self.timer.isActive()
        self.on_stop_clicked()

        self.sim.reset_field()

        self.cx = 0.5 * (self.sim.px - 1)
        self.cy = 0.5 * (self.sim.py - 1)
        self.R = self.sim.py / 4.0

        x = self.cx + self.R * math.cos(0)
        y = self.cy + self.R * math.sin(0)

        self.sim.set_body_force(
            int(x),
            int(y),
            amp=DEFAULT_FORCE_AMP,
            sigma=DEFAULT_FORCE_SIGMA,
            active=True,
        )

        self._force_mode = "circle"
        self._reset_gui_after_init()
        if was_running:
            self.on_start_clicked()

    def on_init_rain_clicked(self) -> None:
        was_running = self.timer.isActive()
        self.on_stop_clicked()

        self.sim.reset_field()
        self._injector_reset()

        self._force_mode = "rain"
        self._reset_gui_after_init()
        if was_running:
            self.on_start_clicked()

    def on_init_mouse_clicked(self) -> None:
        was_running = self.timer.isActive()
        self.on_stop_clicked()

        self.sim.reset_field()
        self.sim.state.force_active = False
        self.sim.state.force_dirty = True

        self._force_mode = "mouse"
        self._reset_gui_after_init()
        if was_running:
            self.on_start_clicked()

    @staticmethod
    def sci_no_plus(x, decimals=0):
        x = float(x)
        s = f"{x:.{decimals}E}"
        return s.replace("E+", "E").replace("e+", "e")

    def on_folder_clicked(self) -> None:
        N = self.sim.N
        Re = self.sim.re
        K0 = self.sim.k0
        CFL = self.sim.cfl
        STEPS = self.sim.get_iteration()

        folder = f"cupyturbo_{N}_{self.sci_no_plus(Re)}_{K0}_{CFL}_{STEPS}"

        desktop = self._desktop_path()

        dlg = self._make_folder_dialog(title=f"Case: {folder}", start_dir=desktop)
        if dlg.exec():
            base_dir = dlg.selectedFiles()[0]
        else:
            return

        folder_path = os.path.join(base_dir, folder)
        os.makedirs(folder_path, exist_ok=True)

        print(f"[SAVE] Dumping fields to folder: {folder_path}")
        self._dump_pgm_full(self._get_full_field("u"), os.path.join(folder_path, "u_velocity.pgm"))
        self._dump_pgm_full(self._get_full_field("v"), os.path.join(folder_path, "v_velocity.pgm"))
        self._dump_pgm_full(
            self._get_full_field("kinetic"), os.path.join(folder_path, "kinetic.pgm")
        )
        self._dump_pgm_full(self._get_full_field("omega"), os.path.join(folder_path, "omega.pgm"))
        print("[SAVE] Completed.")

    # These two helpers are implemented in the GUI file (kept there):
    #   - _desktop_path()
    #   - _make_folder_dialog()

    def _get_full_field(self, variable: str) -> np.ndarray:
        S = self.sim.state

        if variable == "u":
            field = S.ur_full[0]
            return (field.get() if S.backend == "gpu" else field).astype(np.float32)

        if variable == "v":
            field = S.ur_full[1]
            return (field.get() if S.backend == "gpu" else field).astype(np.float32)

        if variable == "kinetic":
            dns_all.dns_kinetic(S)
            field = S.ur_full[2]
            return (field.get() if S.backend == "gpu" else field).astype(np.float32)

        if variable == "omega":
            dns_all.dns_om2_phys(S)
            field = S.ur_full[2]
            return (field.get() if S.backend == "gpu" else field).astype(np.float32)

        raise ValueError(f"Unknown variable: {variable}")

    def on_re_changed(self, value: str) -> None:
        self.sim.re = float(value)
        # Re-init the current mode (pao/circle/rain/mouse) instead of a raw reset_field().
        self.on_reset_clicked()

    def on_k0_changed(self, value: str) -> None:
        self.sim.k0 = float(value)
        # Re-init the current mode (pao/circle/rain/mouse) instead of a raw reset_field().
        self.on_reset_clicked()

    def on_cfl_changed(self, value: str) -> None:
        self.sim.cfl = float(value)
        # Re-init the current mode (pao/circle/rain/mouse) instead of a raw reset_field().
        self.on_reset_clicked()

    def on_steps_changed(self, value: str) -> None:
        self.sim.max_steps = int(float(value))

    def on_update_changed(self, value: str) -> None:
        self._update_intervall = int(float(value))

    def _injector_reset(self) -> None:
        self._inj_next_t = self.sim.get_time()
        self._inj_off_iter = -1
        self._inj_last = None

    def _injector_maybe_apply(self) -> None:
        if not self._inj_enabled:
            return

        t = self.sim.get_time()
        it = self.sim.get_iteration()

        if self.sim.state.force_active and it >= self._inj_off_iter and self._inj_last is not None:
            x, y, amp, sigma = self._inj_last
            self.sim.set_body_force(x, y, amp=amp, sigma=sigma, active=False)
            return

        if (not self.sim.state.force_active) and (t >= self._inj_next_t):
            x = int(self._rng.integers(0, self.sim.px))
            y = int(self._rng.integers(0, self.sim.py))

            amp = float(self._inj_amp0) * (1.0 if self._rng.random() < 0.5 else -1.0)
            sigma = float(self._inj_sigma)

            self.sim.set_body_force(x, y, amp=amp, sigma=sigma, active=True)
            self._inj_last = (x, y, amp, sigma)
            self._inj_off_iter = it + int(self._inj_duration_steps)
            self._inj_next_t = t + 1.0 / float(self._inj_f_hz)

    def _on_timer(self) -> None:
        t = self.sim.get_time()

        self.sim.step(self._update_intervall, run_next_dt=True)

        self._status_update_counter += 1

        if self._force_mode == "circle":
            theta = 2.0 * math.pi * self.f_hz * t
            x = self.cx + self.R * math.cos(theta)
            y = self.cy - self.R * math.sin(theta)

            self.sim.set_body_force(
                int(x),
                int(y),
                amp=DEFAULT_FORCE_AMP,
                sigma=DEFAULT_FORCE_SIGMA,
                active=True,
            )
        elif self._force_mode == "rain":
            self._injector_maybe_apply()

        if self._status_update_counter >= self._update_intervall:
            pixels = self.sim.get_frame_pixels()
            self._update_image(pixels)

            now = time.time()
            elapsed = now - self._sim_start_time
            steps = self.sim.get_iteration() - self._sim_start_iter
            fps = None
            if elapsed > 0 and steps > 0:
                fps = steps / elapsed

            self._update_status(self.sim.get_time(), self.sim.get_iteration(), fps)
            self._status_update_counter = 0

        if self.sim.get_iteration() >= self.sim.max_steps:
            if self.auto_reset_checkbox.isChecked():
                self.sim.reset_field()
                self._sim_start_time = time.time()
                self._sim_start_iter = self.sim.get_iteration()
            else:
                self.timer.stop()
                print("Max steps reached — simulation stopped (Auto-Reset OFF).")

    @staticmethod
    def _dump_pgm_full(arr: np.ndarray, filename: str) -> None:
        h, w = arr.shape
        minv = float(arr.min())
        maxv = float(arr.max())
        rng = maxv - minv

        with open(filename, "wb") as f:
            f.write(f"P5\n{w} {h}\n255\n".encode())

            if rng <= 1e-12:
                f.write(bytes([128]) * (w * h))
                return

            norm = (arr - minv) / rng
            pix = (1.0 + norm * 254.0).round().clip(1, 255).astype(np.uint8)
            f.write(pix.tobytes())

    # --- mouse force mode helpers (GUI provides _display_scale) ---

    def _map_label_xy_to_sim_xy(self, lx: int, ly: int) -> Optional[tuple[int, int]]:
        pix = self.image_label.pixmap()
        if pix is None:
            return None

        pw = pix.width()
        ph = pix.height()
        lw = self.image_label.width()
        lh = self.image_label.height()

        ox = (lw - pw) // 2
        oy = (lh - ph) // 2

        x = lx - ox
        y = ly - oy

        if not (0 <= x < pw and 0 <= y < ph):
            return None

        scale = self._display_scale()

        if scale < 1.0:
            up = int(round(1.0 / scale))
            x //= up
            y //= up
        elif scale > 1.0:
            s = int(round(scale))
            x *= s
            y *= s

        sx_max = int(self.sim.px) - 1
        sy_max = int(self.sim.py) - 1
        if sx_max >= 0:
            x = 0 if x < 0 else (sx_max if x > sx_max else x)
        if sy_max >= 0:
            y = 0 if y < 0 else (sy_max if y > sy_max else y)

        return (int(x), int(y))

    def _apply_force_from_label_xy(self, lx: int, ly: int, active: bool) -> None:
        xy = self._map_label_xy_to_sim_xy(lx, ly)
        if xy is None:
            return
        x, y = xy
        self._force_last_xy = (x, y)
        self.sim.set_body_force(
            x,
            y,
            amp=DEFAULT_FORCE_AMP,
            sigma=DEFAULT_FORCE_SIGMA,
            active=active,
        )

    def on_image_pressed(self, lx: int, ly: int) -> None:
        if self._force_mode != "mouse":
            return
        self._force_dragging = True
        self.image_label.grabMouse()
        self._apply_force_from_label_xy(lx, ly, active=True)

    def on_image_moved(self, lx: int, ly: int) -> None:
        if self._force_mode != "mouse":
            return
        if not self._force_dragging:
            return
        self._apply_force_from_label_xy(lx, ly, active=True)

    def on_image_released(self, lx: int, ly: int) -> None:
        if self._force_mode != "mouse":
            return
        if not self._force_dragging:
            return
        self._force_dragging = False
        self.image_label.releaseMouse()

        if self._force_last_xy is not None:
            x, y = self._force_last_xy
            self.sim.set_body_force(
                x,
                y,
                amp=DEFAULT_FORCE_AMP,
                sigma=DEFAULT_FORCE_SIGMA,
                active=False,
            )