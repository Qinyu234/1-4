"""Tests for inertial COM frame helpers."""

from __future__ import annotations

import numpy as np

from fairy_orbit.core.body import (
    Body,
    System,
    com_position,
    com_velocity,
    from_com_inertial_frame,
    to_com_inertial_frame,
)


def test_to_com_inertial_frame_zeros_com():
    sys = System(
        bodies=[
            Body(1.0, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), name="a"),
            Body(1.0, np.array([-1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), name="b"),
            Body(2.0, np.array([0.0, 2.0, 0.0]), np.array([1.0, 0.0, 0.0]), name="c"),
        ]
    )
    to_com_inertial_frame(sys)
    assert np.linalg.norm(com_position(sys)) < 1e-14
    assert np.linalg.norm(com_velocity(sys)) < 1e-14


def test_com_roundtrip():
    sys = System(
        bodies=[
            Body(1.0, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), name="a"),
            Body(3.0, np.array([0.0, 2.0, 0.0]), np.array([0.5, 0.0, 0.0]), name="b"),
        ]
    )
    r0 = sys.positions().copy()
    v0 = sys.velocities().copy()
    shift = to_com_inertial_frame(sys)
    assert np.linalg.norm(com_position(sys)) < 1e-14
    from_com_inertial_frame(sys, shift)
    assert np.allclose(sys.positions(), r0)
    assert np.allclose(sys.velocities(), v0)
