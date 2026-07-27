"""Tests for 8-channel representation errors and σ normalize."""

from __future__ import annotations

import numpy as np
import pytest

from fairy_orbit.observe.rep_error import (
    CHANNELS,
    RepErrorSnapshot,
    RepSigmas,
    apply_sigmas,
    frobenius,
    rep_error_for_perm,
    wrap_angle,
)


def test_wrap_angle():
    assert wrap_angle(0.0) == pytest.approx(0.0)
    assert wrap_angle(np.pi + 0.1) == pytest.approx(-np.pi + 0.1, abs=1e-12)
    assert wrap_angle(-np.pi - 0.1) == pytest.approx(np.pi - 0.1, abs=1e-12)


def test_relative_Er_Ev_zero_on_rotated_perm():
    rng = np.random.default_rng(0)
    r0 = rng.normal(size=(4, 3))
    v0 = rng.normal(size=(4, 3))
    perm = (1, 2, 3, 0)
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    r = np.stack([R @ r0[perm[i]] for i in range(4)])
    v = np.stack([R @ v0[perm[i]] for i in range(4)])
    snap = rep_error_for_perm(r, v, r0, v0, perm, mu=1.0)
    assert snap.E_r == pytest.approx(0.0, abs=1e-12)
    assert snap.E_v == pytest.approx(0.0, abs=1e-12)
    assert snap.E_a == pytest.approx(0.0, abs=1e-10)
    assert snap.E_e == pytest.approx(0.0, abs=1e-10)
    assert snap.E_M == pytest.approx(0.0, abs=1e-8)


def test_channels_are_eight():
    assert len(CHANNELS) == 8


def test_apply_sigmas():
    snap = RepErrorSnapshot(
        E_r=0.2,
        E_v=0.4,
        E_a=0.01,
        E_e=0.02,
        E_i=0.03,
        E_Omega=0.04,
        E_omega=0.05,
        E_M=0.06,
        E_energy=1e-12,
        R=np.eye(3),
        perm=(0, 1, 2, 3),
    )
    sig = RepSigmas(
        E_r=0.1,
        E_v=0.2,
        E_a=0.01,
        E_e=0.01,
        E_i=0.01,
        E_Omega=0.01,
        E_omega=0.01,
        E_M=0.01,
    )
    tilde = apply_sigmas(snap, sig)
    assert tilde["E_r"] == pytest.approx(2.0)
    assert tilde["E_v"] == pytest.approx(2.0)


def test_frobenius():
    X = np.array([[3.0, 0.0], [0.0, 4.0]])
    assert frobenius(X) == pytest.approx(5.0)
