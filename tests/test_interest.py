"""Interest score prefers secular change over static MEGNO≈2 orbits."""

from fairy_orbit.observe.interest import a_order_changed, interestingness


def test_static_is_penalized():
    score = interestingness(
        {
            "status": "success",
            "a_delta_rms": 1e-6,
            "n_encounters": 0,
            "a_order_changed": False,
            "e_max": [0.1, 0.1],
            "megno": 2.0,
        }
    )
    assert score < 0


def test_migration_and_swap_rewarded():
    score = interestingness(
        {
            "status": "success",
            "a_delta_rms": 0.08,
            "n_encounters": 5,
            "a_order_changed": True,
            "e_max": [0.2, 0.25],
            "megno": 2.5,
        }
    )
    assert score > 5


def test_a_order_changed():
    assert a_order_changed([1.0, 1.3, 1.8, 2.2], [1.4, 1.1, 1.8, 2.2])
    assert not a_order_changed([1.0, 1.3, 1.8], [1.01, 1.31, 1.81])
