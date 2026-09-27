"""
Interactive measuring / calibration tool (Pygame).

Workflow
    1. Click a point in the LEFT image.  The epipolar curve of that pixel is
       drawn in the RIGHT image and an epipolar-guided ZNCC search proposes the
       correspondence (green cross).  With auto-match on and a reliable match,
       the distance is measured immediately.
    2. Clicking the RIGHT image instead sets the correspondence by hand; a
       manual click is always authoritative.
    3. K adds a calibration point: type the true distance, press Enter, then
       click the object in the left and right images.  With ≥ 15 points the
       model is (re)trained automatically and saved.
"""

from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass

import numpy as np

from .config import DEFAULT_SENSOR, MIN_CAL_PTS, TrainConfig
from .data import load_points, save_points
from .matching import clahe, epipolar_match, lk_refine, to_gray
from .model import StereoModel
from .training import train

DISP_W, DISP_H = 800, 600
HUD_H = 132
BG, HUD_BG = (18, 18, 22), (12, 12, 18)
C_TEXT, C_DIM = (225, 225, 230), (140, 140, 150)
C_LEFT, C_CAL, C_MATCH = (50, 255, 90), (90, 150, 255), (60, 255, 60)
C_CURVE, C_OK, C_BAD, C_MEAS = (70, 160, 90), (255, 255, 60), (255, 70, 70), (255, 170, 60)
RINGS = ((1.0, (255, 90, 90)), (2.0, (255, 180, 60)), (5.0, (90, 180, 255)), (10.0, (200, 120, 255)))

HELP = [
    "LEFT CLICK (left img)   pick point → auto-match on epipolar curve",
    "LEFT CLICK (right img)  set correspondence manually (authoritative)",
    "K  add calibration point (type distance, Enter, click L then R)",
    "U  undo last calibration point / cancel K      G  retrain now",
    "C  clear calibration (press twice)              R  clear measurements",
    "A  toggle auto-match      V  toggle Γ-cell coverage overlay",
    "M  3-D length between the last two measurements",
    "L  known length between the last two measurements → calibrates lateral scale",
    "T  swap left/right images (view only — measuring is disabled while swapped)",
    "S  screenshot      E  export CSV (pixels, X Y Z, range, σ)",
    "Wheel zoom · right-drag pan · H help · Esc quit / cancel",
]


@dataclass
class Measurement:
    uL: float
    vL: float
    uR: float
    vR: float
    Z: float  # depth along the left optical axis (m)
    sigma: float
    source: str  # "auto" | "manual"
    P: tuple = (0.0, 0.0, 0.0)  # 3-D point in the left-camera frame (m)
    extrapolated: bool = False  # outside the depth range covered by the calibration

    @property
    def range(self) -> float:
        return float(np.linalg.norm(self.P))


class App:
    def __init__(self, left_path, right_path, calib_path="stereo_calibration.json",
                 points_path="cal_pts.json", left_surface=None, right_surface=None):
        import pygame

        self.pg = pygame
        pygame.init()
        pygame.font.init()
        self.calib_path, self.points_path = calib_path, points_path
        self.model, self.calibrated = self._load_model()
        self.sensor = self.model.sensor
        self.W, self.H = DISP_W * 2, DISP_H + HUD_H
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("Γ-Grid Stereo Distance  ·  H = help")
        self.font = pygame.font.SysFont("consolas,dejavusansmono,menlo,monospace", 13)
        self.big = pygame.font.SysFont("consolas,dejavusansmono,menlo,monospace", 20, bold=True)

        self.imgs = [left_surface if left_surface is not None else self._load_image(left_path),
                     right_surface if right_surface is not None else self._load_image(right_path)]
        self.grays = [self._prep_gray(s) for s in self.imgs]
        self.cal_pts = [tuple(p) for p in load_points(points_path)]
        print(f"[INIT] {len(self.cal_pts)} calibration points from {points_path}")

        self.measurements: list[Measurement] = []
        self.mode = "measure"  # measure | input_Z | calib_L | calib_R
        self.typed = ""
        self.pending: dict = {}
        self.cur_L = self.cur_R = self.match = None
        self.cur_Z = self.cur_sigma = None
        self.cur_err = None
        self.auto_match, self.show_help, self.show_cov = True, False, False
        self.swapped = False  # images swapped: the calibration no longer matches → measuring disabled
        self.length = None  # (metres, σ) between the last two measurements
        self.clear_armed = 0.0
        self.message, self.message_t = "", 0.0
        self.zoom, self.pan = [1.0, 1.0], [[0.0, 0.0], [0.0, 0.0]]
        self.drag, self.drag0, self.drag_pan = [False, False], (0, 0), [[0, 0], [0, 0]]
        self._cache: dict = {}

    # ── loading ──────────────────────────────────────────────────────────────
    def _load_model(self):
        if os.path.exists(self.calib_path):
            try:
                m = StereoModel.load(self.calib_path)
                print(f"[INIT] loaded {self.calib_path}  dx={m.baseline[0]:.4f}  "
                      f"ω=[{', '.join(f'{math.degrees(a):+.2f}°' for a in m.rotation)}]")
                return m, True
            except Exception as e:  # corrupt file → start fresh but say so
                print(f"[INIT] could not load {self.calib_path} ({e}); starting uncalibrated")
        return StereoModel.nominal(DEFAULT_SENSOR), False

    def _load_image(self, path):
        s = self.pg.image.load(path).convert()
        w, h = s.get_size()
        if (w, h) != (self.sensor.width, self.sensor.height):
            if abs(w / h - self.sensor.width / self.sensor.height) > 0.01:
                print(f"[WARN] {path} is {w}×{h}: aspect ratio differs from the "
                      f"{self.sensor.width}×{self.sensor.height} calibration — distances will be wrong")
            s = self.pg.transform.smoothscale(s, (self.sensor.width, self.sensor.height))
        return s

    def _prep_gray(self, surf):
        arr = self.pg.surfarray.array3d(surf).transpose(1, 0, 2)
        return clahe(to_gray(arr))

    # ── coordinates ──────────────────────────────────────────────────────────
    @property
    def sx(self):
        return DISP_W / self.sensor.width

    @property
    def sy(self):
        return DISP_H / self.sensor.height

    def panel_of(self, mx, my):
        return None if my >= DISP_H else (0 if mx < DISP_W else 1)

    def to_img(self, mx, my, p):
        return ((mx - p * DISP_W - self.pan[p][0]) / (self.zoom[p] * self.sx),
                (my - self.pan[p][1]) / (self.zoom[p] * self.sy))

    def to_screen(self, ix, iy, p):
        return (int(ix * self.sx * self.zoom[p] + self.pan[p][0] + p * DISP_W),
                int(iy * self.sy * self.zoom[p] + self.pan[p][1]))

    def clamp_pan(self, p):
        z = self.zoom[p]
        self.pan[p][0] = min(0.0, max(self.pan[p][0], DISP_W - DISP_W * z))
        self.pan[p][1] = min(0.0, max(self.pan[p][1], DISP_H - DISP_H * z))

    # ── actions ──────────────────────────────────────────────────────────────
    def say(self, msg):
        print(msg)
        self.message, self.message_t = msg, time.time()

    def reset_current(self):
        self.cur_extra = None
        self.cur_L = self.cur_R = self.match = None
        self.cur_Z = self.cur_sigma = self.cur_err = None

    def click_left(self, ix, iy):
        self.reset_current()
        self.cur_L = (ix, iy)
        if self.swapped:
            self.say("  images are swapped — press T to restore before measuring or calibrating")
            return
        self.match = epipolar_match(self.model, self.grays[0], self.grays[1], ix, iy)
        if self.match is not None:
            lk = lk_refine(self.grays[0], self.grays[1], ix, iy, self.match.u, self.match.v)
            if lk is not None and math.hypot(lk[0] - self.match.u, lk[1] - self.match.v) < 3:
                self.match.u, self.match.v = float(lk[0]), float(lk[1])
        if self.mode == "calib_L":
            self.pending.update(uL=ix, vL=iy)
            self.mode = "calib_R"
            self.say(f"  L=({ix},{iy}) — now click the same object in the RIGHT image")
        elif (self.mode == "measure" and self.calibrated and self.auto_match
              and self.match is not None and self.match.reliable):
            self.measure(self.match.u, self.match.v, "auto")

    def click_right(self, ix, iy):
        if self.swapped:
            self.say("  images are swapped — press T to restore before measuring or calibrating")
            return
        if self.mode == "calib_R" and "uL" in self.pending:
            p = (self.pending["uL"], self.pending["vL"], ix, iy, self.pending["Z"])
            self.cal_pts.append(p)
            save_points(self.cal_pts, self.points_path)
            self.mode, self.pending = "measure", {}
            self.cur_R = (ix, iy)
            self.say(f"  calibration point {len(self.cal_pts)} saved "
                     f"(L=({p[0]},{p[1]}) R=({ix},{iy}) Z={p[4]:.3f} m)")
            if len(self.cal_pts) >= MIN_CAL_PTS:
                self.retrain()
        elif self.cur_L is not None:
            self.measure(ix, iy, "manual")

    def measure(self, uR, vR, source):
        uL, vL = self.cur_L
        self.cur_R = (uR, vR)
        # an automatic (ZNCC + Lucas–Kanade) match is sub-pixel; a hand click has the calibration's click noise
        s_px = self.model.meta.get("sigma_px", 1.0)
        s_px = min(s_px, 0.5) if source == "auto" else s_px
        Z, sigma, _ = self.model.depth_uncertainty(uL, vL, uR, vR, sigma_px=s_px)
        if not math.isfinite(Z):
            self.cur_Z, self.cur_err = None, "no valid intersection (rays diverge / behind camera)"
            return
        self.cur_Z, self.cur_sigma, self.cur_err = Z, sigma, None
        self.cur_parts = None
        if self.model.companion is not None:
            self.cur_parts = (self.model.triangulate_gamma([uL], [vL], [uR], [vR])[0][0],
                              float(self.model.companion_model().triangulate([uL], [vL], [uR], [vR])[0]))
        P = tuple(float(c) for c in self.model.point_3d([uL], [vL], [uR], [vR])[0])
        default = ([min(p[4] for p in self.cal_pts), max(p[4] for p in self.cal_pts)] if self.cal_pts
                   else [0.0, float("inf")])  # older calibration files lack the field
        lo, hi = self.model.meta.get("depth_range_m", default)
        self.cur_extra = (float(np.linalg.norm(P)), not (0.8 * lo <= Z <= 1.25 * hi))
        if self.calibrated:
            self.measurements.append(Measurement(uL, vL, uR, vR, Z, sigma, source, P, self.cur_extra[1]))
            self.length = None
            z_ols = self.model.triangulate_gamma([uL], [vL], [uR], [vR])[0][0]
            gd, it = self.model.triangulate_gd([uL], [vL], [uR], [vR], max_iter=50000)
            check = f"GD {gd[0]:.4f} m in {it} it" if it < 50000 else "GD not converged"
            parts = f"Γ-OLS {z_ols:.4f} m, {check}"
            if self.model.companion is not None:
                zb = self.model.companion_model().triangulate([uL], [vL], [uR], [vR])[0]
                parts += f", Brown–Conrady {zb:.4f} m"
            print(f"  Z = {Z:.3f} ± {sigma:.3f} m  [{source}]   ({parts})")

    def retrain(self):
        if len(self.cal_pts) < MIN_CAL_PTS:
            self.say(f"  need {MIN_CAL_PTS} calibration points (have {len(self.cal_pts)})")
            return
        self.draw_banner(f"Calibrating on {len(self.cal_pts)} points (≈3 s) …")
        t0 = time.time()
        refs = self.model.meta.get("length_refs", [])
        self.model, _ = train(np.asarray(self.cal_pts), TrainConfig(), self.sensor, log=None)
        if refs:
            self.model.fit_lateral_scale(refs)
        self.model.save(self.calib_path)
        self.calibrated = True
        m = self.model.meta
        self.say(f"[TRAIN] {time.time() - t0:.1f}s  in-sample MAPE {100 * m['in_sample_mape']:.2f}%  "
                 f"→ saved {self.calib_path}")
        if self.cur_L and self.cur_R:
            self.measure(*self.cur_R, "manual")

    def key(self, k, uni=""):
        pg = self.pg
        if self.mode in ("input_Z", "input_L"):
            if k in (pg.K_RETURN, pg.K_KP_ENTER):
                try:
                    Z = float(self.typed.replace(",", "."))
                    if not (0 < Z < 1000):
                        raise ValueError
                    if self.mode == "input_L":
                        self.mode = "measure"
                        self.add_length_reference(Z)
                    else:
                        self.pending = {"Z": Z}
                        self.mode = "calib_L"
                        self.say(f"  Z = {Z:.3f} m — click the object in the LEFT image")
                except ValueError:
                    self.say(f"  '{self.typed}' is not a valid distance")
                    self.mode = "measure"
            elif k == pg.K_ESCAPE:
                self.mode = "measure"
                self.say("  cancelled")
            elif k == pg.K_BACKSPACE:
                self.typed = self.typed[:-1]
            elif uni and uni in "0123456789.,":
                self.typed += uni
            return True
        if k == pg.K_ESCAPE:
            if self.mode != "measure":
                self.mode, self.pending = "measure", {}
                self.say("  cancelled")
                return True
            return False
        if k == pg.K_k:
            if self.swapped:
                self.say("  images are swapped — press T to restore before adding calibration points")
                return True
            self.mode, self.typed = "input_Z", ""
        elif k == pg.K_u:
            if self.mode in ("calib_L", "calib_R"):
                self.mode, self.pending = "measure", {}
                self.say("  calibration point cancelled")
            elif self.cal_pts:
                rm = self.cal_pts.pop()
                save_points(self.cal_pts, self.points_path)
                self.say(f"  removed calibration point {len(self.cal_pts) + 1} (Z={rm[4]:.2f} m)")
        elif k == pg.K_c:
            if time.time() - self.clear_armed < 2.0:
                self.cal_pts.clear()
                save_points(self.cal_pts, self.points_path)
                self.model, self.calibrated = StereoModel.nominal(self.sensor), False
                self.reset_current()
                self.say("  calibration cleared — model reset to nominal pinhole")
            else:
                self.clear_armed = time.time()
                self.say("  press C again within 2 s to delete ALL calibration points")
        elif k == pg.K_r:
            self.measurements.clear()
            self.reset_current()
            self.say("  measurements cleared")
        elif k == pg.K_t:
            self.imgs.reverse()
            self.grays.reverse()
            self._cache.clear()
            self.reset_current()
            self.swapped = not self.swapped
            self.say("  images SWAPPED — view only: the calibration belongs to the original order, so measuring "
                     "and calibrating are disabled (T restores)" if self.swapped else "  images restored")
        elif k == pg.K_m:
            self.measure_length()
        elif k == pg.K_l:
            if len(self.measurements) < 2 or not self.calibrated:
                self.say("  measure the two ends of a known length first, then press L")
            else:
                self.mode, self.typed = "input_L", ""
        elif k == pg.K_g:
            self.retrain()
        elif k == pg.K_a:
            self.auto_match = not self.auto_match
            self.say(f"  auto-match {'ON' if self.auto_match else 'OFF'}")
        elif k == pg.K_v:
            self.show_cov = not self.show_cov
        elif k == pg.K_h:
            self.show_help = not self.show_help
        elif k == pg.K_s:
            name = time.strftime("screenshot_%Y%m%d_%H%M%S.png")
            pg.image.save(self.screen, name)
            self.say(f"  saved {name}")
        elif k == pg.K_e:
            self.export_csv()
        return True

    def measure_length(self):
        """3-D distance between the last two measurements (e.g. an object's width)."""
        if len(self.measurements) < 2:
            self.say("  measure two points first, then press M")
            return
        a, b = self.measurements[-2], self.measurements[-1]
        d = float(np.linalg.norm(np.subtract(a.P, b.P)))
        # first-order propagation: P = Z·ray  ⇒  ∂P/∂Z = ray = P/Z; project on the segment direction u
        u = np.subtract(a.P, b.P) / max(d, 1e-9)
        s = float(np.hypot(a.sigma * np.dot(u, a.P) / a.Z, b.sigma * np.dot(u, b.P) / b.Z))
        self.length = (d, s)
        self.say(f"  length between the last two points: {d:.3f} m ± {s:.3f}")

    def add_length_reference(self, length):
        """Use the last two measurements as a known length → refit the lateral scale and save."""
        a, b = self.measurements[-2], self.measurements[-1]
        refs = list(self.model.meta.get("length_refs", []))
        refs.append([a.uL, a.vL, a.uR, a.vR, b.uL, b.vL, b.uR, b.vR, float(length)])
        try:
            k = self.model.fit_lateral_scale(refs)
        except ValueError as e:
            self.say(f"  {e}")
            return
        self.model.save(self.calib_path)
        self._refresh_points()
        self.say(f"  lateral scale k = {k:.4f} from {len(refs)} known length(s) (≙ focal "
                 f"{self.sensor.f_init / k:.0f} px) — saved")

    def _refresh_points(self):
        for m in self.measurements:
            m.P = tuple(float(c) for c in self.model.point_3d([m.uL], [m.vL], [m.uR], [m.vR])[0])
        if self.length is not None:
            self.measure_length()

    def export_csv(self, path="measurements.csv"):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["uL", "vL", "uR", "vR", "X_m", "Y_m", "Z_depth_m", "range_m", "sigma_m", "source",
                        "extrapolated"])
            for m in self.measurements:
                w.writerow([m.uL, m.vL, f"{m.uR:.2f}", f"{m.vR:.2f}", f"{m.P[0]:.4f}", f"{m.P[1]:.4f}",
                            f"{m.Z:.4f}", f"{m.range:.4f}", f"{m.sigma:.4f}", m.source, int(m.extrapolated)])
        self.say(f"  exported {len(self.measurements)} measurements → {path}")

    # ── event loop ───────────────────────────────────────────────────────────
    def handle(self, ev) -> bool:
        pg = self.pg
        if ev.type == pg.QUIT:
            return False
        if ev.type == pg.KEYDOWN:
            return self.key(ev.key, getattr(ev, "unicode", ""))
        if ev.type == pg.MOUSEWHEEL:
            mx, my = pg.mouse.get_pos()
            p = self.panel_of(mx, my)
            if p is not None:
                old = self.zoom[p]
                self.zoom[p] = max(1.0, min(12.0, old * (1.2 if ev.y > 0 else 1 / 1.2)))
                lx = mx - p * DISP_W
                self.pan[p][0] = lx - (lx - self.pan[p][0]) * self.zoom[p] / old
                self.pan[p][1] = my - (my - self.pan[p][1]) * self.zoom[p] / old
                self.clamp_pan(p)
        elif ev.type == pg.MOUSEBUTTONDOWN:
            mx, my = ev.pos
            p = self.panel_of(mx, my)
            if p is None:
                return True
            if ev.button == 3:
                self.drag[p], self.drag0, self.drag_pan[p] = True, (mx, my), list(self.pan[p])
            elif ev.button == 1 and self.mode not in ("input_Z", "input_L"):
                ix, iy = (int(round(c)) for c in self.to_img(mx, my, p))
                if 0 <= ix < self.sensor.width and 0 <= iy < self.sensor.height:
                    (self.click_left if p == 0 else self.click_right)(ix, iy)
        elif ev.type == pg.MOUSEBUTTONUP and ev.button == 3:
            self.drag = [False, False]
        elif ev.type == pg.MOUSEMOTION:
            for p in (0, 1):
                if self.drag[p]:
                    self.pan[p][0] = self.drag_pan[p][0] + ev.pos[0] - self.drag0[0]
                    self.pan[p][1] = self.drag_pan[p][1] + ev.pos[1] - self.drag0[1]
                    self.clamp_pan(p)
        return True

    def run(self):
        clock = self.pg.time.Clock()
        running = True
        while running:
            for ev in self.pg.event.get():
                running = self.handle(ev) and running
            self.render()
            self.pg.display.flip()
            clock.tick(60)
        self.pg.quit()

    # ── drawing ──────────────────────────────────────────────────────────────
    def draw_banner(self, text):
        self.screen.fill((50, 25, 25), self.pg.Rect(0, DISP_H, self.W, HUD_H))
        self.screen.blit(self.big.render(text, True, (255, 200, 100)), (12, DISP_H + 40))
        self.pg.display.flip()

    def _panel_surface(self, p):
        """Crop the visible full-resolution region and scale only that (sharp + fast)."""
        z, (px, py) = self.zoom[p], self.pan[p]
        key = (p, round(z, 4), round(px, 2), round(py, 2))
        if key in self._cache:
            return self._cache[key]
        s = self.sensor
        kx, ky = self.sx * z, self.sy * z
        x0, y0 = max(0, int(math.floor(-px / kx))), max(0, int(math.floor(-py / ky)))
        x1 = min(s.width, int(math.ceil((DISP_W - px) / kx)) + 1)
        y1 = min(s.height, int(math.ceil((DISP_H - py) / ky)) + 1)
        sub = self.imgs[p].subsurface(self.pg.Rect(x0, y0, x1 - x0, y1 - y0))
        size = (max(1, round((x1 - x0) * kx)), max(1, round((y1 - y0) * ky)))
        scaled = self.pg.transform.smoothscale(sub, size) if z < 4 else self.pg.transform.scale(sub, size)
        out = (scaled, (int(round(x0 * kx + px)) + p * DISP_W, int(round(y0 * ky + py))))
        if len(self._cache) > 8:
            self._cache.clear()
        self._cache[key] = out
        return out

    def cross(self, pos, col, r=12, w=1):
        x, y = pos
        self.pg.draw.line(self.screen, col, (x - r, y), (x + r, y), w)
        self.pg.draw.line(self.screen, col, (x, y - r), (x, y + r), w)

    def label(self, text, pos, col):
        t = self.font.render(text, True, col)
        bg = self.pg.Surface((t.get_width() + 4, t.get_height() + 2), self.pg.SRCALPHA)
        bg.fill((0, 0, 0, 150))
        self.screen.blit(bg, (pos[0] - 2, pos[1] - 1))
        self.screen.blit(t, pos)

    def render(self):
        pg, scr = self.pg, self.screen
        scr.fill(BG)
        for p in (0, 1):
            scr.set_clip(pg.Rect(p * DISP_W, 0, DISP_W, DISP_H))
            surf, pos = self._panel_surface(p)
            scr.blit(surf, pos)
            if self.show_cov:
                self.draw_coverage(p)
            for pt in self.cal_pts:
                pg.draw.circle(scr, C_CAL, self.to_screen(pt[2 * p], pt[2 * p + 1], p), 3)

        # left overlays
        scr.set_clip(pg.Rect(0, 0, DISP_W, DISP_H))
        for m in self.measurements:
            sp = self.to_screen(m.uL, m.vL, 0)
            pg.draw.circle(scr, C_MEAS, sp, 4)
            self.label(f"{m.Z:.2f}±{m.sigma:.2f}m", (sp[0] + 6, sp[1] - 16), C_MEAS)
        if self.cur_L:
            sp = self.to_screen(*self.cur_L, 0)
            pg.draw.circle(scr, C_LEFT, sp, 6, 2)
            self.cross(sp, C_LEFT)

        # right overlays
        scr.set_clip(pg.Rect(DISP_W, 0, DISP_W, DISP_H))
        if self.cur_L:
            cu, cv, _ = self.model.epipolar_curve(*self.cur_L, n=200)
            pts = [self.to_screen(u, v, 1) for u, v in zip(cu, cv)]
            if len(pts) > 1:
                pg.draw.lines(scr, C_CURVE, False, pts, 1)
            if self.calibrated:
                for Zr, col in RINGS:
                    uR, vR, ok = self.model.project_left_point_to_right(
                        *self.cur_L, self.model.raw_depth_for(Zr))
                    if ok[0]:
                        sp = self.to_screen(uR[0], vR[0], 1)
                        pg.draw.circle(scr, col, sp, 7, 2)
                        self.label(f"{Zr:g}m", (sp[0] + 9, sp[1] - 9), col)
        if self.match is not None:
            col = C_MATCH if self.match.reliable else C_DIM
            sp = self.to_screen(self.match.u, self.match.v, 1)
            self.cross(sp, col, 14, 2)
            self.label(f"ZNCC {self.match.score:.2f}", (sp[0] + 10, sp[1] + 6), col)
        if self.cur_R:
            pg.draw.circle(scr, C_BAD if self.cur_err else C_OK, self.to_screen(*self.cur_R, 1), 6, 2)

        scr.set_clip(None)
        pg.draw.line(scr, (60, 60, 70), (DISP_W, 0), (DISP_W, DISP_H), 1)
        self.draw_hud()
        if self.show_help:
            self.draw_help()

    def draw_coverage(self, p):
        """Shade Γ-grid cells by number of calibration points (guides where to add more)."""
        s = self.sensor
        counts = np.zeros((s.grid_rows - 1, s.grid_cols - 1), int)
        for pt in self.cal_pts:
            i = min(int(pt[2 * p + 1] / s.cell_h), s.grid_rows - 2)
            j = min(int(pt[2 * p] / s.cell_w), s.grid_cols - 2)
            counts[i, j] += 1
        for i in range(s.grid_rows - 1):
            for j in range(s.grid_cols - 1):
                x0, y0 = self.to_screen(j * s.cell_w, i * s.cell_h, p)
                x1, y1 = self.to_screen((j + 1) * s.cell_w, (i + 1) * s.cell_h, p)
                n = counts[i, j]
                shade = self.pg.Surface((max(1, x1 - x0), max(1, y1 - y0)), self.pg.SRCALPHA)
                shade.fill((255, 60, 60, 70) if n == 0 else (255, 200, 60, 50) if n < 3 else (60, 255, 120, 35))
                self.screen.blit(shade, (x0, y0))
                self.pg.draw.rect(self.screen, (200, 200, 200), (x0, y0, x1 - x0, y1 - y0), 1)
                self.label(str(n), (x0 + 4, y0 + 4), C_TEXT)

    def draw_hud(self):
        pg, scr = self.pg, self.screen
        scr.fill(HUD_BG, pg.Rect(0, DISP_H, self.W, HUD_H))
        y = DISP_H + 8
        if self.swapped:
            scr.blit(self.big.render("IMAGES SWAPPED — view only (press T to restore)", True, C_BAD), (10, y))
        elif self.cur_Z is not None:
            txt = f"depth Z = {self.cur_Z:.3f} m ± {self.cur_sigma:.3f}"
            if not self.calibrated:
                txt = "UNCALIBRATED estimate: " + txt
            if self.cur_extra:
                txt += f"   ·   straight-line {self.cur_extra[0]:.3f} m"
                if self.cur_extra[1]:
                    txt += "   ·   EXTRAPOLATED (outside calibrated depths)"
            if self.length:
                txt += f"   ·   length {self.length[0]:.3f} m ± {self.length[1]:.3f}"
            scr.blit(self.big.render(txt, True, C_BAD if self.cur_extra and self.cur_extra[1] else C_OK), (10, y))
        elif self.cur_err:
            scr.blit(self.big.render(self.cur_err, True, C_BAD), (10, y))
        elif self.mode in ("input_Z", "input_L"):
            what = "True distance (m)" if self.mode == "input_Z" else "True length between the last two points (m)"
            scr.blit(self.big.render(f"{what}: {self.typed}_   [Enter/Esc]", True, (255, 220, 120)), (10, y))
        else:
            scr.blit(self.big.render("Click a point in the left image", True, C_DIM), (10, y))
        y += 30
        m, d, w = self.model, self.model.baseline, np.degrees(self.model.rotation)
        status = (f"CALIBRATED ({m.meta.get('n_points', '?')} pts)" if self.calibrated
                  else f"UNCALIBRATED ({len(self.cal_pts)}/{MIN_CAL_PTS} pts)")
        lines = [
            f"[{status}]  d=[{d[0]:.4f}, {d[1]:+.4f}]  pitch/yaw/roll=[{w[0]:+.2f}°, {w[1]:+.2f}°, {w[2]:+.2f}°]"
            f"  auto-match={'on' if self.auto_match else 'off'}  zoom L/R {self.zoom[0]:.1f}×/{self.zoom[1]:.1f}×",
        ]
        info = []
        if self.cur_L:
            info.append(f"L=({self.cur_L[0]}, {self.cur_L[1]})")
        if self.cur_R:
            info.append(f"R=({self.cur_R[0]:.1f}, {self.cur_R[1]:.1f})")
        if self.match:
            info.append(f"match ZNCC={self.match.score:.2f} uniq={self.match.uniqueness:.2f}"
                        f"{'' if self.match.reliable else ' (unreliable — click right image)'}")
        if getattr(self, "cur_parts", None) and self.cur_Z is not None:
            info.append(f"members: Γ {self.cur_parts[0]:.3f} · Brown–Conrady {self.cur_parts[1]:.3f} m")
        info.append(f"{len(self.measurements)} measurements")
        lines.append("   ".join(info))
        if self.mode == "calib_L":
            lines.append(f"CALIBRATION  Z={self.pending['Z']:.3f} m → click the object in the LEFT image")
        elif self.mode == "calib_R":
            lines.append(f"CALIBRATION  Z={self.pending['Z']:.3f} m → click the same object in the RIGHT image")
        elif self.message and time.time() - self.message_t < 6:
            lines.append(self.message.strip())
        lines.append("K add-cal · U undo · G train · C clear · R clear-meas · M length · L known length · "
                     "A auto · V coverage · T swap · S shot · E csv · H help · Esc quit")
        for i, line in enumerate(lines):
            scr.blit(self.font.render(line, True, C_TEXT if i < len(lines) - 1 else C_DIM), (10, y + 18 * i))

    def draw_help(self):
        pg = self.pg
        box = pg.Surface((620, 24 + 20 * len(HELP)), pg.SRCALPHA)
        box.fill((0, 0, 0, 200))
        for i, line in enumerate(HELP):
            box.blit(self.font.render(line, True, C_TEXT), (14, 12 + 20 * i))
        self.screen.blit(box, ((self.W - box.get_width()) // 2, 60))


def run(left_path, right_path, calib_path="stereo_calibration.json", points_path="cal_pts.json"):
    App(left_path, right_path, calib_path, points_path).run()
