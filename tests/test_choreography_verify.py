"""Tests for PROMPT §3.2 choreography verification at T/n."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fairy_orbit.design.seeds import build_free_polygon_seed
from fairy_orbit.observe.choreography_verify import (
    accept_seed_choreography,
    cyclic_role_perm,
    is_regular_equal_ngon,
    rotation_axis_angle,
    verify_seed_choreography,
)


def test_cyclic_role_perm_n4():
    assert cyclic_role_perm(4, 1) == (1, 2, 3, 0)
    assert cyclic_role_perm(4, 0) == (0, 1, 2, 3)


def test_rotation_axis_angle_identity():
    R = np.eye(3)
    axis, angle = rotation_axis_angle(R)
    assert angle == pytest.approx(0.0, abs=1e-12)


def test_rotation_axis_angle_z90():
    th = math.pi / 2
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    axis, angle = rotation_axis_angle(R)
    assert abs(axis[2]) == pytest.approx(1.0, abs=1e-9)
    assert angle == pytest.approx(th, rel=1e-9)


def test_free4_square_passes_Tn_but_accept_rejects_maintained():
    seed = build_free_polygon_seed(4, seed_id="free_4_square_re", family="free_4")
    out = verify_seed_choreography(seed, shift=1, atol_rel=1e-8)
    assert out.ok
    acc = accept_seed_choreography(seed, shift=1, atol_rel=1e-8)
    assert not acc.ok
    assert acc.maintains_regular_ngon
    assert acc.reason == "rejected_maintained_regular_ngon"


def test_asymmetric_not_regular():
    pos = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.2, 1.5, 0.0],
            [-1.2, 0.3, 0.0],
            [0.1, -0.8, 0.0],
        ]
    )
    assert not is_regular_equal_ngon(pos, rtol=0.05)
