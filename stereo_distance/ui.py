"""Interactive Pygame tool: click a point in both images, read the distance.

Controls::

    left click   pick left point, then right point        wheel   zoom panel
    K            add calibration point (asks Z in terminal) right-drag  pan
    U            undo last calibration point / cancel K    T       swap L/R
    C            clear calibration + reset model           R       clear measurements
    Esc          quit

After ``rig.min_cal_pts`` calibration points the trainer runs automatically
and writes the calibration file.  The raw right click is always authoritative
for depth; ZNCC / SSD / LK results are shown as hints only.
"""

from __future__ import annotations

import math
import os

import numpy as np

from .calibration import TrainConfig, train
from .config import DEFAULT_RIG, RigConfig
from .io import (
    CAL_PTS_PATH,
    CALIB_PATH,
    load_cal_pts,
    load_calibration,
    save_cal_pts,
    save_calibration,
)
from .lens import default_params, predict_right
from .matching import clahe, lk_refine, ncc_match, ssd_subpixel, to_gray
from .triangulation import triangulate

DISP_W, DISP_H = 800, 600
HUD_H = 110


def surf_to_gray(surf) -> np.ndarray:
    """Pygame surface -> ``float32`` grayscale ``(H, W)``."""
    import pygame

    arr = pygame.surfarray.array3d(surf).transpose(1, 0, 2)
    return to_gray(arr)


def run(
    left_path: str,
    right_path: str,
    calib_path: str = CALIB_PATH,
    cal_pts_path: str = CAL_PTS_PATH,
    rig: RigConfig = DEFAULT_RIG,
) -> None:  # pragma: no cover - interactive
    """Open the two images and start the measurement loop."""
    import pygame

    pygame.init()
    pygame.font.init()

    gamma_l, gamma_r, d = default_params(rig)
    calibrated = False
    if os.path.exists(calib_path):
        try:
            gamma_l, gamma_r, d = load_calibration(calib_path, rig)
            calibrated = True
            print(f"[INIT] loaded {calib_path}  dx={d[0]:.4f}m")
        except Exception as e:  # noqa: BLE001 - any broken file -> start uncalibrated
            print(f"[INIT] load failed ({e}); starting uncalibrated")

    win_w, win_h = 2 * DISP_W, DISP_H + HUD_H
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(
        "5x5 Pure-Gamma Stereo  |  K=add  U=undo  R=clear  C=reset  T=swap  Esc=quit"
    )

    def _load(p: str):
        s = pygame.image.load(p).convert()
        if s.get_size() != (rig.img_w, rig.img_h):
            s = pygame.transform.scale(s, (rig.img_w, rig.img_h))
        return s

    left_img, right_img = _load(left_path), _load(right_path)
    left_gray = clahe(surf_to_gray(left_img))
    right_gray = clahe(surf_to_gray(right_img))
    print("[INIT] CLAHE applied to both images")

    font = pygame.font.SysFont("consolas", 13)
    bigfont = pygame.font.SysFont("consolas", 17, bold=True)

    cal_pts = load_cal_pts(cal_pts_path)
    print(f"[INIT] loaded {len(cal_pts)} cal_pts")
    measurements: list[tuple] = []
    pending: dict = {"Z": None, "uL": None, "vL": None}
    mode = "measure"  # "measure" | "calib_wait_L" | "calib_wait_R"
    current_l = current_r = current_z = current_err = ncc_hint = None

    zooms = [1.0, 1.0]
    pans = [[0, 0], [0, 0]]
    drag = [False, False]
    drag_0 = (0, 0)
    drag_p = [[0, 0], [0, 0]]
    sx_scale, sy_scale = DISP_W / rig.img_w, DISP_H / rig.img_h

    def panel_of(mx, my):
        if my >= DISP_H:
            return None
        return 0 if mx < DISP_W else 1

    def screen_to_img(mx, my, p):
        return (
            ((mx - p * DISP_W) - pans[p][0]) / (zooms[p] * sx_scale),
            (my - pans[p][1]) / (zooms[p] * sy_scale),
        )

    def img_to_screen(ix, iy, p):
        return (
            ix * sx_scale * zooms[p] + pans[p][0] + p * DISP_W,
            iy * sy_scale * zooms[p] + pans[p][1],
        )

    def clamp_pan(p):
        sw, sh = int(DISP_W * zooms[p]), int(DISP_H * zooms[p])
        pans[p][0] = min(0, max(pans[p][0], DISP_W - sw))
        pans[p][1] = min(0, max(pans[p][1], DISP_H - sh))

    def draw_panel(surf, p):
        s = pygame.transform.scale(surf, (int(DISP_W * zooms[p]), int(DISP_H * zooms[p])))
        screen.set_clip(pygame.Rect(p * DISP_W, 0, DISP_W, DISP_H))
        screen.blit(s, (p * DISP_W + pans[p][0], pans[p][1]))

    def recompute_depth():
        nonlocal current_z, current_err
        if current_l is None or current_r is None:
            current_z = current_err = None
            return
        z, _, _, err = triangulate(*current_l, *current_r, gamma_l, gamma_r, d, rig)
        current_z = None if math.isnan(z) else z
        current_err = err

    def maybe_auto_train():
        nonlocal gamma_l, gamma_r, d, calibrated
        if len(cal_pts) < rig.min_cal_pts:
            return
        screen.fill((50, 25, 25), pygame.Rect(0, DISP_H, win_w, HUD_H))
        screen.blit(
            bigfont.render(
                f"  TRAINING 5x5 grid on {len(cal_pts)} pts - see terminal ...",
                True, (255, 200, 100),
            ),
            (10, DISP_H + 30),
        )
        pygame.display.flip()
        res = train(cal_pts, TrainConfig(), rig)
        gamma_l, gamma_r, d = res.params
        save_calibration(gamma_l, gamma_r, d, calib_path)
        print(f"[SAVE] {calib_path}  dx={d[0]:.4f}m  dy={d[1]:.4f}m")
        calibrated = True

    def ask_z():
        print(f"\n[K-MODE] calibration pt {len(cal_pts) + 1}  (need {rig.min_cal_pts})")
        try:
            z = float(input("  Enter Real Distance (m): ").strip())
            if z <= 0:
                raise ValueError
            return z
        except Exception:  # noqa: BLE001
            print("  invalid - cancelled")
            return None

    clock = pygame.time.Clock()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_k:
                    z = ask_z()
                    if z is not None:
                        pending.update({"Z": z, "uL": None, "vL": None})
                        mode = "calib_wait_L"
                        print(f"  -> click LEFT for Z={z:.2f}m")
                elif ev.key == pygame.K_c:
                    cal_pts.clear()
                    save_cal_pts(cal_pts, cal_pts_path)
                    gamma_l, gamma_r, d = default_params(rig)
                    calibrated = False
                    current_l = current_r = current_z = current_err = ncc_hint = None
                    mode = "measure"
                    pending["Z"] = None
                    print("[C] reset to defaults")
                elif ev.key == pygame.K_r:
                    measurements.clear()
                    current_l = current_r = current_z = current_err = ncc_hint = None
                    print("[R] measurements cleared")
                elif ev.key == pygame.K_u:
                    if mode in ("calib_wait_L", "calib_wait_R"):
                        mode = "measure"
                        pending["Z"] = None
                        print("[U] K-mode cancelled")
                    elif cal_pts:
                        rm = cal_pts.pop()
                        save_cal_pts(cal_pts, cal_pts_path)
                        print(f"[U] removed pt {len(cal_pts) + 1}  Z={rm[4]:.2f}m")
                elif ev.key == pygame.K_t:
                    left_img, right_img = right_img, left_img
                    left_gray, right_gray = right_gray, left_gray
                    current_l = current_r = current_z = current_err = ncc_hint = None
                    print("[T] swapped L/R")

            elif ev.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                p = panel_of(mx, my)
                if p is not None:
                    old = zooms[p]
                    zooms[p] = max(1.0, min(8.0, old * (1.15 if ev.y > 0 else 1 / 1.15)))
                    lx = mx - p * DISP_W
                    pans[p][0] = int(lx - (lx - pans[p][0]) * zooms[p] / old)
                    pans[p][1] = int(my - (my - pans[p][1]) * zooms[p] / old)
                    clamp_pan(p)

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos
                p = panel_of(mx, my)
                if p is None:
                    continue
                if ev.button == 3:
                    drag[p] = True
                    drag_0 = (mx, my)
                    drag_p[p] = pans[p].copy()
                elif ev.button == 1:
                    ix, iy = screen_to_img(mx, my, p)
                    ix, iy = int(round(ix)), int(round(iy))
                    if not (0 <= ix < rig.img_w and 0 <= iy < rig.img_h):
                        continue
                    if p == 0:
                        current_l = (ix, iy)
                        current_r = current_z = current_err = None
                        ncc_hint = ncc_match(left_gray, right_gray, ix, iy)
                        if mode == "calib_wait_L" and pending["Z"] is not None:
                            pending["uL"], pending["vL"] = ix, iy
                            mode = "calib_wait_R"
                            print(f"  L=({ix},{iy}) - click RIGHT")
                    else:
                        current_r = (ix, iy)
                        if mode == "calib_wait_R" and pending["uL"] is not None:
                            cal_pts.append((pending["uL"], pending["vL"], ix, iy, pending["Z"]))
                            save_cal_pts(cal_pts, cal_pts_path)
                            print(f"  R=({ix},{iy})  cal_pts={len(cal_pts)}/{rig.min_cal_pts}")
                            mode = "measure"
                            pending["Z"] = None
                            maybe_auto_train()
                        elif calibrated and current_l is not None:
                            ul_, vl_ = current_l
                            ssd_u = ssd_subpixel(left_gray, right_gray, ul_, vl_, ix)
                            lk = lk_refine(left_gray, right_gray, ul_, vl_, ix, iy)
                            ssd_s = "--" if ssd_u is None else f"{ssd_u:.2f}"
                            lk_s = "--" if lk is None else f"({lk[0]:.2f},{lk[1]:.2f})"
                            print(f"  [INFO-ONLY] click=({ix},{iy})  SSD={ssd_s}  LK={lk_s}")
                            recompute_depth()
                            if current_z is not None:
                                measurements.append((ul_, vl_, ix, iy, current_z))

            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 3:
                    drag[0] = drag[1] = False

            elif ev.type == pygame.MOUSEMOTION:
                mx, my = ev.pos
                for p in (0, 1):
                    if drag[p]:
                        pans[p][0] = drag_p[p][0] + (mx - drag_0[0])
                        pans[p][1] = drag_p[p][1] + (my - drag_0[1])
                        clamp_pan(p)

        # ---- draw -----------------------------------------------------------
        screen.fill((18, 18, 22))
        draw_panel(left_img, 0)
        draw_panel(right_img, 1)

        screen.set_clip(pygame.Rect(0, 0, DISP_W, DISP_H))
        for ul, vl, *_ in cal_pts:
            sx, sy = img_to_screen(ul, vl, 0)
            pygame.draw.circle(screen, (90, 150, 255), (int(sx), int(sy)), 3)
        if current_l is not None:
            sx, sy = img_to_screen(current_l[0], current_l[1], 0)
            pygame.draw.circle(screen, (50, 255, 90), (int(sx), int(sy)), 6, 2)
            pygame.draw.line(screen, (50, 255, 90), (int(sx) - 12, int(sy)), (int(sx) + 12, int(sy)))
            pygame.draw.line(screen, (50, 255, 90), (int(sx), int(sy) - 12), (int(sx), int(sy) + 12))

        screen.set_clip(pygame.Rect(DISP_W, 0, DISP_W, DISP_H))
        for _, _, ur, vr, _ in cal_pts:
            sx, sy = img_to_screen(ur, vr, 1)
            pygame.draw.circle(screen, (90, 150, 255), (int(sx), int(sy)), 3)

        if calibrated and current_l is not None:
            curve = []
            for z_test in np.linspace(0.5, 50.0, 60):
                pr = predict_right(current_l[0], current_l[1], z_test, gamma_l, gamma_r, d, rig)
                if pr is not None:
                    sx, sy = img_to_screen(pr[0], pr[1], 1)
                    if DISP_W <= sx <= 2 * DISP_W:
                        curve.append((int(sx), int(sy)))
            if len(curve) > 1:
                pygame.draw.lines(screen, (70, 160, 90), False, curve, 1)
        elif current_l is not None:
            _, sy = img_to_screen(0, current_l[1], 1)
            pygame.draw.line(screen, (70, 160, 90), (DISP_W, int(sy)), (2 * DISP_W, int(sy)), 1)

        if current_l is not None and ncc_hint is not None:
            sx, sy = img_to_screen(ncc_hint, current_l[1], 1)
            if DISP_W <= sx < 2 * DISP_W:
                pygame.draw.line(screen, (60, 255, 60), (int(sx) - 14, int(sy)), (int(sx) + 14, int(sy)), 2)
                pygame.draw.line(screen, (60, 255, 60), (int(sx), int(sy) - 14), (int(sx), int(sy) + 14), 2)

        if calibrated and current_l is not None:
            for zh, col in ((1.0, (255, 90, 90)), (2.0, (255, 180, 60)), (5.0, (90, 180, 255))):
                pr = predict_right(current_l[0], current_l[1], zh, gamma_l, gamma_r, d, rig)
                if pr is None:
                    continue
                ur, vr = pr
                if 0 <= ur < rig.img_w and 0 <= vr < rig.img_h:
                    sx, sy = img_to_screen(ur, vr, 1)
                    pygame.draw.circle(screen, col, (int(sx), int(sy)), 7, 2)
                    screen.blit(font.render(f"{zh:.0f}m", True, col), (int(sx) + 9, int(sy) - 9))

        if current_r is not None:
            sx, sy = img_to_screen(current_r[0], current_r[1], 1)
            col = (255, 60, 60) if current_err else (255, 255, 60)
            pygame.draw.circle(screen, col, (int(sx), int(sy)), 6, 2)

        screen.set_clip(None)
        pygame.draw.line(screen, (60, 60, 70), (DISP_W, 0), (DISP_W, DISP_H), 1)

        screen.fill((12, 12, 18), pygame.Rect(0, DISP_H, win_w, HUD_H))
        status = "CALIBRATED (5x5)" if calibrated else f"UNCALIBRATED ({len(cal_pts)}/{rig.min_cal_pts})"
        hud = [
            f"[{status}]  mode={mode}   d=[{d[0]:+.4f},{d[1]:+.4f},0]   "
            f"G0={rig.g0:.4f}   zL={zooms[0]:.2f}x zR={zooms[1]:.2f}x"
        ]
        if current_l:
            hud.append(f"L=({current_l[0]:4d},{current_l[1]:4d})")
        if current_r:
            hud.append(f"R=({current_r[0]:4d},{current_r[1]:4d})")
        if current_z is not None:
            hud.append(f">>> Z = {current_z:.3f} m   ({len(measurements)} stored)")
        elif current_err:
            hud.append(f"triangulate: {current_err}")
        if mode == "calib_wait_L":
            hud.append(f"K-mode -> click LEFT for Z={pending['Z']:.2f}m")
        elif mode == "calib_wait_R":
            hud.append(f"K-mode -> click RIGHT  (L=({pending['uL']},{pending['vL']}); raw click is authoritative)")
        hud.append("K=add-cal  U=undo-cal  C=clear+reset  R=clear-meas  T=swap  wheel=zoom  rclick-drag=pan  Esc=quit")
        for i, line in enumerate(hud[:6]):
            screen.blit(font.render(line, True, (220, 220, 225)), (8, DISP_H + 6 + i * 16))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
