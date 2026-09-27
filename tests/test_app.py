import numpy as np
import pytest

pygame = pytest.importorskip("pygame")


@pytest.fixture
def app(tmp_path, small_scene):
    from stereo_gamma.app import App

    rig, left, right, depth, model = small_scene
    calib = tmp_path / "cal.json"
    model.save(calib)
    pygame.init()
    surf = [pygame.surfarray.make_surface(np.repeat(im[..., None], 3, 2).transpose(1, 0, 2))
            for im in (left, right)]
    a = App(None, None, str(calib), str(tmp_path / "pts.json"), surf[0], surf[1])
    yield a, depth
    pygame.quit()


def _key(a, k, uni=""):
    return a.handle(pygame.event.Event(pygame.KEYDOWN, key=k, unicode=uni))


def test_auto_match_measures_depth(app):
    a, depth = app
    assert a.calibrated
    h, w = depth.shape
    hits = 0
    for u, v in [(w // 3, h // 2), (w // 2, h // 2), (2 * w // 3, h // 3), (w // 4, h // 3)]:
        a.click_left(u, v)
        a.render()
        if a.cur_Z is not None:
            hits += 1
            assert abs(a.cur_Z / depth[v, u] - 1) < 0.08
            assert a.cur_sigma > 0
    assert hits >= 2 and len(a.measurements) == hits


def test_calibration_point_flow_and_keys(app, tmp_path):
    a, _ = app
    _key(a, pygame.K_k)
    for ch in "2.5":
        _key(a, 0, ch)
    _key(a, pygame.K_RETURN)
    assert a.mode == "calib_L" and a.pending["Z"] == 2.5
    a.click_left(100, 100)
    a.click_right(80, 101)
    assert a.mode == "measure" and a.cal_pts[-1] == (100, 100, 80, 101, 2.5)
    _key(a, pygame.K_u)
    assert a.cal_pts == []
    for k in (pygame.K_h, pygame.K_v, pygame.K_a, pygame.K_t, pygame.K_r):
        _key(a, k)
        a.render()
    _key(a, pygame.K_c)
    _key(a, pygame.K_c)
    assert not a.calibrated
    assert _key(a, pygame.K_ESCAPE) is False


def test_swap_disables_measuring_and_calibration(app):
    a, depth = app
    h, w = depth.shape
    _key(a, pygame.K_t)
    assert a.swapped
    a.click_left(w // 2, h // 2)
    assert a.cur_Z is None and a.match is None
    _key(a, pygame.K_k)
    assert a.mode == "measure"  # K refused while swapped
    a.render()
    _key(a, pygame.K_t)
    assert not a.swapped


def test_length_between_points_and_csv(app, tmp_path):
    import csv

    a, depth = app
    h, w = depth.shape
    for u, v in [(w // 3, h // 2), (w // 2, h // 2), (2 * w // 3, h // 3), (w // 4, h // 3)]:
        a.click_left(u, v)
    assert len(a.measurements) >= 2
    _key(a, pygame.K_m)
    m1, m2 = a.measurements[-2:]
    assert a.length[0] == pytest.approx(np.linalg.norm(np.subtract(m1.P, m2.P)))
    assert a.length[1] > 0 and m1.range >= m1.Z
    out = tmp_path / "m.csv"
    a.export_csv(str(out))
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == len(a.measurements) and float(rows[0]["range_m"]) >= float(rows[0]["Z_depth_m"])


def test_known_length_key_calibrates_lateral_scale(app, small_scene, tmp_path):
    a, depth = app
    rig = small_scene[0]
    h, w = depth.shape
    pts = [(w // 3, h // 2), (2 * w // 3, h // 3), (w // 2, h // 2), (w // 4, h // 3)]
    for u, v in pts:
        a.click_left(u, v)
    assert len(a.measurements) >= 2
    m1, m2 = a.measurements[-2:]
    T = [rig.left.unproject(np.array([m.uL]), np.array([m.vL]))[0] * depth[int(m.vL), int(m.uL)] for m in (m1, m2)]
    true_len = float(np.linalg.norm(T[0] - T[1]))
    _key(a, pygame.K_l)
    assert a.mode == "input_L"
    for ch in f"{true_len:.4f}":
        _key(a, 0, ch)
    _key(a, pygame.K_RETURN)
    assert a.model.lateral_scale != 1.0 and a.model.meta["length_refs"]
    _key(a, pygame.K_m)
    assert a.length[0] == pytest.approx(true_len, rel=0.02)
