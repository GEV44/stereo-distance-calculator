import numpy as np

from stereo_gamma.legacy import train_legacy

from . import v1_reference as v1


def test_legacy_port_reproduces_v1_exactly(real_points, capsys):
    pts = real_points[:20]
    GL, GR, d = v1.train([tuple(p) for p in pts], epochs=200, verbose=10 ** 9)
    capsys.readouterr()
    m = train_legacy(pts, epochs=200)
    assert np.allclose(m.gamma_L[..., 0], GL, atol=1e-12)
    assert np.allclose(m.gamma_R[..., 0], GR, atol=1e-12)
    assert np.allclose(m.baseline, d, atol=1e-12)
    z_v1 = [v1.triangulate(*p[:4], GL, GR, d)[0] for p in real_points]
    assert np.allclose(m.triangulate(*real_points[:, :4].T)[0], z_v1, atol=1e-9)
