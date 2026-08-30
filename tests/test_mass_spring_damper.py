import numpy as np
import pytest

from probeing.metrics import cumulative_trapezoid
from probeing.models import InteractionParameters, MassSpringDamperModel


@pytest.mark.parametrize(
    "stiffness,damping,mass",
    [
        (0.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 1.0, -1.0),
        (np.inf, 1.0, 1.0),
        (1.0, np.nan, 1.0),
    ],
)
def test_invalid_interaction_parameters_are_rejected(stiffness, damping, mass):
    with pytest.raises(ValueError):
        InteractionParameters(stiffness, damping, mass)


@pytest.mark.parametrize(
    "parameters",
    [
        InteractionParameters(100.0, 2.0, 1.0),  # underdamped
        InteractionParameters(100.0, 20.0, 1.0),  # critically damped
        InteractionParameters(100.0, 35.0, 1.0),  # overdamped
    ],
)
def test_numerical_free_response_matches_analytical_solution(parameters):
    model = MassSpringDamperModel(parameters)
    time = np.linspace(0.0, 1.0, 5001)
    numerical = model.simulate(
        time,
        np.zeros_like(time),
        initial_displacement_m=0.012,
        initial_velocity_m_per_s=-0.03,
    )
    analytical = model.analytical_free_response(
        time,
        initial_displacement_m=0.012,
        initial_velocity_m_per_s=-0.03,
    )

    np.testing.assert_allclose(
        numerical.displacement_m, analytical.displacement_m, rtol=2.0e-10, atol=2.0e-12
    )
    np.testing.assert_allclose(
        numerical.velocity_m_per_s,
        analytical.velocity_m_per_s,
        rtol=2.0e-9,
        atol=2.0e-11,
    )


def test_numerical_forced_step_response_matches_analytical_solution():
    model = MassSpringDamperModel(InteractionParameters(250.0, 8.0, 0.75))
    time = np.linspace(0.0, 1.5, 7501)
    force = np.full_like(time, 1.2)

    numerical = model.simulate(time, force)
    analytical = model.analytical_step_response(time, force_n=1.2)

    np.testing.assert_allclose(
        numerical.displacement_m, analytical.displacement_m, rtol=2.0e-9, atol=2.0e-12
    )
    np.testing.assert_allclose(
        numerical.velocity_m_per_s,
        analytical.velocity_m_per_s,
        rtol=2.0e-8,
        atol=2.0e-11,
    )


def test_equilibrium_remains_exactly_stationary():
    parameters = InteractionParameters(400.0, 12.0, 2.0)
    model = MassSpringDamperModel(parameters)
    time = np.linspace(0.0, 2.0, 2001)
    force = np.full_like(time, 3.0)
    equilibrium = 3.0 / parameters.stiffness_n_per_m

    result = model.simulate(time, force, initial_displacement_m=equilibrium)

    np.testing.assert_allclose(result.displacement_m, equilibrium, atol=1.0e-15)
    np.testing.assert_allclose(result.velocity_m_per_s, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(result.acceleration_m_per_s2, 0.0, atol=1.0e-13)


def test_damping_removes_mechanical_energy_at_the_expected_rate():
    parameters = InteractionParameters(120.0, 4.0, 0.8)
    model = MassSpringDamperModel(parameters)
    time = np.linspace(0.0, 2.0, 10001)
    result = model.simulate(
        time, np.zeros_like(time), initial_displacement_m=0.02
    )
    dissipated = cumulative_trapezoid(
        parameters.damping_n_s_per_m * result.velocity_m_per_s**2, time
    )

    assert np.max(np.diff(result.mechanical_energy_j)) <= 2.0e-13
    np.testing.assert_allclose(
        result.mechanical_energy_j + dissipated,
        result.mechanical_energy_j[0],
        rtol=0.0,
        atol=7.0e-9,
    )


def test_zero_damping_conserves_energy_and_remains_periodic():
    parameters = InteractionParameters(144.0, 0.0, 1.0)
    model = MassSpringDamperModel(parameters)
    period = 2.0 * np.pi / 12.0
    time = np.linspace(0.0, 4.0 * period, 8001)
    result = model.simulate(
        time, np.zeros_like(time), initial_displacement_m=0.01
    )

    np.testing.assert_allclose(
        result.mechanical_energy_j,
        result.mechanical_energy_j[0],
        rtol=3.0e-12,
        atol=1.0e-14,
    )
    assert result.displacement_m[-1] == pytest.approx(0.01, abs=2.0e-12)
    assert result.velocity_m_per_s[-1] == pytest.approx(0.0, abs=2.0e-11)


def test_very_high_stiffness_has_small_finite_static_deflection():
    stiffness = 100_000.0
    mass = 1.0
    damping = 2.0 * np.sqrt(stiffness * mass)
    model = MassSpringDamperModel(InteractionParameters(stiffness, damping, mass))
    time = np.linspace(0.0, 0.08, 8001)
    result = model.simulate(time, np.ones_like(time))

    assert np.all(np.isfinite(result.displacement_m))
    assert result.displacement_m[-1] == pytest.approx(1.0 / stiffness, rel=1.0e-8)
    assert np.max(result.displacement_m) <= 1.0 / stiffness * (1.0 + 1.0e-9)
