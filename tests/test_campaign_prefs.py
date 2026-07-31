"""Campaign preference defaults (RESPONSE §6–§7 + thread 7:2:6:1)."""

from __future__ import annotations

from fairy_orbit.observe.campaign_prefs import (
    BRANCH2_IN_DEFAULT_CAMPAIGN,
    BRANCH2_PROBE_DEFAULT_DIVERSE,
    CHOREO_N5_IN_DEFAULT_CAMPAIGN,
    FLOQUET_GATE_DEFAULT,
    PATH_A_AUTO_FLOQUET_SWEEP,
    ALLOW_UNSTABLE_PATH_A_SEED,
    SLOT_WEIGHTS,
    WEIGHT_BRANCH2,
    WEIGHT_CHOREO_N4,
    WEIGHT_CHOREO_N5,
    WEIGHT_PATH_A,
    campaign_priority_blurb,
    wall_hours_for_slots,
)


def test_branch2_and_n5_dark_by_default() -> None:
    assert BRANCH2_IN_DEFAULT_CAMPAIGN is False
    assert CHOREO_N5_IN_DEFAULT_CAMPAIGN is False
    assert BRANCH2_PROBE_DEFAULT_DIVERSE >= 2


def test_thread_weights_are_7_2_6_1() -> None:
    assert (WEIGHT_CHOREO_N4, WEIGHT_CHOREO_N5, WEIGHT_PATH_A, WEIGHT_BRANCH2) == (
        7,
        2,
        6,
        1,
    )
    assert sum(SLOT_WEIGHTS.values()) == 16


def test_wall_split_active_n4_and_path_a() -> None:
    walls = wall_hours_for_slots(13.0, ["choreo_n4", "path_a"])
    assert walls["choreo_n4"] == 7.0
    assert walls["path_a"] == 6.0
    assert "choreo_n5" not in walls


def test_floquet_and_path_a_prefs() -> None:
    assert FLOQUET_GATE_DEFAULT is True
    assert PATH_A_AUTO_FLOQUET_SWEEP is True
    assert ALLOW_UNSTABLE_PATH_A_SEED is True


def test_priority_blurb_mentions_threads() -> None:
    text = campaign_priority_blurb()
    assert "7:2:6:1" in text
    assert "n5=off" in text
    assert "branch2=off" in text
