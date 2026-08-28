import numpy as np
import pytest

from probeing.probing import raised_cosine_ramp


def test_raised_cosine_ramp_has_exact_bounds_and_endpoint_conditions():
    trajectory = raised_cosine_ramp(0.005, 1.0, 0.25, 0.001)

    assert len(trajectory.time_s) == 1251
    assert trajectory.time_s[0] == pytest.approx(0.0)
    assert trajectory.time_s[-1] == pytest.approx(1.25)
    assert trajectory.displacement_m[0] == pytest.approx(0.0)
    assert trajectory.displacement_m[-1] == pytest.approx(0.005)
    assert trajectory.velocity_m_per_s[0] == pytest.approx(0.0)
    assert trajectory.velocity_m_per_s[1000] == pytest.approx(0.0, abs=1.0e-16)
    assert np.min(trajectory.displacement_m) >= 0.0
    assert np.max(trajectory.displacement_m) <= 0.005
    assert np.all(np.diff(trajectory.displacement_m) >= -1.0e-15)
    assert np.all(trajectory.velocity_m_per_s[1001:] == 0.0)


def test_analytical_velocity_agrees_with_central_difference_in_rise_interior():
    trajectory = raised_cosine_ramp(0.005, 1.0, 0.0, 0.0005)
    numerical_velocity = np.gradient(
        trajectory.displacement_m, trajectory.time_s, edge_order=2
    )

    np.testing.assert_allclose(
        numerical_velocity[2:-2],
        trajectory.velocity_m_per_s[2:-2],
        rtol=5.0e-7,
        atol=1.0e-12,
    )


def test_non_integral_sample_count_is_rejected():
    with pytest.raises(ValueError, match="integer multiple"):
        raised_cosine_ramp(0.005, 1.0, 0.2, 0.003)

