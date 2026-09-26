import json

import numpy as np
import pytest

from stereo_gamma.data import load_points, parse_points, save_points


def test_both_formats(tmp_path):
    a = [[1, 2, 3, 4, 2.5]]
    b = [{"left_pt": [1, 2], "right_pt": [3, 4], "distance": 2.5, "left_file": "x"}]
    assert np.array_equal(parse_points(a), parse_points(b))


def test_invalid_rows_raise():
    with pytest.raises(ValueError):
        parse_points([[1, 2, 3, 4, -1.0]])
    with pytest.raises(ValueError):
        parse_points([[1, 2, 3, 4]])


def test_roundtrip_and_missing(tmp_path):
    p = tmp_path / "p.json"
    assert load_points(p).shape == (0, 5)
    save_points([[1, 2, 3.5, 4, 2.25]], p)
    assert json.load(open(p)) == [[1, 2, 3.5, 4, 2.25]]
    assert np.allclose(load_points(p), [[1, 2, 3.5, 4, 2.25]])


def test_sample_dataset_is_subset_of_cal_pts(root):
    import os

    a = load_points(os.path.join(root, "cal_pts.json"))
    b = load_points(os.path.join(root, "data", "sample_training_pairs.json"))
    assert np.array_equal(a[: len(b)], b)
