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
