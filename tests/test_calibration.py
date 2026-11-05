"""Gradient-descent calibration on synthetic data and the shipped point set."""

import numpy as np
import pytest

from stereo_distance import (
    SyntheticRig,
    TrainConfig,
    depth_mae,
    load_cal_pts,
    train,
    warm_start_baseline,
)


@pytest.fixture(scope="module")
def noisy_pinhole_pts():
    rig = SyntheticRig.pinhole(dx=0.25)
    return rig, rig.make_cal_pts(40, z_range=(0.8, 5.0), noise_px=0.5, seed=3)


def test_warm_start_is_in_the_right_ballpark(noisy_pinhole_pts):
    rig, pts = noisy_pinhole_pts
    dx0 = warm_start_baseline(pts)
    assert 0.15 < dx0 < 0.40  # truth 0.25; heuristic uses mean(disp)*mean(Z)


def test_sgd_recovers_baseline_on_pinhole_data(noisy_pinhole_pts):
    rig, pts = noisy_pinhole_pts
    res = train(pts, TrainConfig(epochs=4000, verbose_every=0), log=None)
    # plain SGD closes the dx/Gamma near-degeneracy slowly: ~5 % after 4000
    # epochs, ~2 % after the default 9000 (see benchmarks/)
    assert res.d[0] == pytest.approx(0.25, rel=0.06)
    assert abs(res.d[1]) < 0.01
    mae = depth_mae(pts, *res.params)
    assert mae < 0.02  # 0.5 px noise at up to 5 m
    assert res.best_loss < 0.002
    assert 0 <= res.best_epoch < 4000


def test_more_epochs_tighten_the_baseline(noisy_pinhole_pts):
    rig, pts = noisy_pinhole_pts
    short = train(pts, TrainConfig(epochs=800, verbose_every=0), log=None)
    long = train(pts, TrainConfig(epochs=3000, verbose_every=0), log=None)
    assert abs(long.d[0] - 0.25) < abs(short.d[0] - 0.25)
    assert long.best_loss < short.best_loss


def test_training_reduces_loss_and_mae_over_epochs(noisy_pinhole_pts):
    rig, pts = noisy_pinhole_pts
    res = train(pts, TrainConfig(epochs=600, verbose_every=100), log=None)
    losses = [h[1] for h in res.history]
    maes = [h[2] for h in res.history]
    assert len(res.history) == 7  # 0,100,...,500 and the last epoch
    assert losses[-1] < 0.2 * losses[0]
    assert maes[-1] < maes[0]


def test_centre_node_frozen_and_clamps_respected(noisy_pinhole_pts):
    rig, pts = noisy_pinhole_pts
    res = train(pts, TrainConfig(epochs=200, verbose_every=0), log=None)
    g0 = rig.rig.g0
    assert res.gamma_l[2, 2] == g0 and res.gamma_r[2, 2] == g0
    assert res.gamma_l.min() >= rig.rig.g_min and res.gamma_l.max() <= rig.rig.g_max
    assert res.d[0] >= rig.rig.dx_min and res.d[2] == 0.0


def test_adam_converges_too(noisy_pinhole_pts):
    rig, pts = noisy_pinhole_pts
    cfg = TrainConfig(epochs=800, optimizer="adam", lr_gamma=2e-3, lr_d=2e-3, verbose_every=0)
    res = train(pts, cfg, log=None)
    assert res.d[0] == pytest.approx(0.25, rel=0.05)
    assert depth_mae(pts, *res.params) < 0.08


def test_huber_resists_an_outlier():
    rig = SyntheticRig.pinhole(dx=0.25)
    pts = rig.make_cal_pts(40, z_range=(0.8, 4.0), noise_px=0.3, seed=8)
    clean = pts.copy()
    pts[0, 4] *= 2.5  # one grossly mislabelled distance
    res_sq = train(pts, TrainConfig(epochs=1200, verbose_every=0), log=None)
    res_hb = train(pts, TrainConfig(epochs=1200, huber_delta=0.05, verbose_every=0), log=None)
    mae_sq = depth_mae(clean[1:], *res_sq.params)
    mae_hb = depth_mae(clean[1:], *res_hb.params)
    assert mae_hb < mae_sq


def test_distorted_rig_is_learned_better_than_pinhole_prior():
    truth = SyntheticRig.distorted(amplitude=0.08, dx=0.25, seed=2)
    pts = truth.make_cal_pts(60, z_range=(0.8, 5.0), noise_px=0.0, seed=4)
    prior = SyntheticRig.pinhole(dx=0.25)
    mae_prior = depth_mae(pts, prior.gamma_l, prior.gamma_r, prior.d)
    res = train(pts, TrainConfig(epochs=3000, verbose_every=0), log=None)
    mae_fit = depth_mae(pts, *res.params)
    assert mae_fit < 0.3 * mae_prior
    assert res.d[0] == pytest.approx(0.25, rel=0.08)


def test_train_rejects_empty_input_and_unknown_optimizer(noisy_pinhole_pts):
    with pytest.raises(ValueError):
        train([], TrainConfig(epochs=1), log=None)
    with pytest.raises(ValueError):
        train(noisy_pinhole_pts[1], TrainConfig(epochs=1, optimizer="rmsprop"), log=None)


def test_init_override_and_log_callback(noisy_pinhole_pts):
    rig, pts = noisy_pinhole_pts
    lines = []
    res = train(pts, TrainConfig(epochs=50, verbose_every=25), log=lines.append,
                init=(rig.gamma_l, rig.gamma_r, rig.d))
    assert lines and lines[0].startswith("[TRAIN]")
    # started at the truth -> stays near it
    assert res.d[0] == pytest.approx(0.25, abs=0.01)
    assert not np.shares_memory(res.gamma_l, rig.gamma_l)


def test_shipped_real_points_fit_under_10cm(shipped_cal_pts_path):
    """27 hand-labelled real points (0.4-4.7 m) from the author's rig."""
    pts = load_cal_pts(shipped_cal_pts_path)
    assert len(pts) == 27
    res = train(pts, TrainConfig(epochs=3000, verbose_every=0), log=None)
    assert 0.2 < res.d[0] < 0.35
    assert depth_mae(pts, *res.params) < 0.12
