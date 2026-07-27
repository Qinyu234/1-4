"""Score floors + significance tests."""

from fairy_orbit.observe.score import ScoreFloors, score_summary


def test_static_below_floor_scores_neg100():
    floors = ScoreFloors(energy_drift=1e-12, amd_total_ptp=1e-8, a_ptp_mean=1e-4, k_sigma=10)
    s = score_summary(
        {
            "status": "success",
            "a_delta_rms": 1e-6,
            "a_ptp_mean": 1e-6,
            "n_encounters": 0,
            "e_max": [0.1],
            "amd_total_ptp": 1e-10,
            "energy_drift": 1e-14,
            "a_order_changed": False,
            "megno": None,
        },
        floors,
    )
    assert s.total == -100.0
    assert s.flags["static"] is True


def test_swap_above_floor_is_interesting():
    floors = ScoreFloors(energy_drift=1e-12, amd_total_ptp=1e-8, a_ptp_mean=1e-4, k_sigma=10)
    s = score_summary(
        {
            "status": "success",
            "a_delta_rms": 0.2,
            "a_ptp_mean": 0.15,
            "n_encounters": 4,
            "e_max": [0.2],
            "amd_total_ptp": 1e-3,
            "energy_drift": 1e-14,
            "a_order_changed": True,
            "megno": 2.5,
        },
        floors,
    )
    assert s.total > 0
    assert s.flags["sig_swap"] and s.flags["sig_a"]
    assert s.flags["integrator_ok"]


def test_escape_hard_fail():
    s = score_summary({"status": "escape"}, ScoreFloors())
    assert s.total <= -1e8
