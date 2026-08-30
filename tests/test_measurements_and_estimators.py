import numpy as np
import pytest

from probeing.estimators import (
    batch_least_squares,
    recursive_least_squares,
    regression_diagnostics,
)
from probeing.measurements import MeasurementNoise, generate_measurements
from probeing.models import InteractionParameters, MassSpringDamperModel
from probeing.probing import chirp


def simulated_case():
    parameters = InteractionParameters(320.0, 7.5, 1.4)
    probe = chirp(1.0, 0.5, 9.0, 3.0, 0.002)
    truth = MassSpringDamperModel(parameters).simulate(probe.time_s, probe.force_n)
    return parameters, truth


def test_perfect_measurements_are_exact_copies():
    _, truth = simulated_case()
    measured = generate_measurements(truth, MeasurementNoise(), random_seed=101)

    np.testing.assert_array_equal(measured.displacement_m, truth.displacement_m)
    np.testing.assert_array_equal(measured.velocity_m_per_s, truth.velocity_m_per_s)
    np.testing.assert_array_equal(
        measured.acceleration_m_per_s2, truth.acceleration_m_per_s2
    )
    np.testing.assert_array_equal(measured.contact_force_n, truth.applied_force_n)


def test_gaussian_measurements_are_seeded_and_configurable():
    _, truth = simulated_case()
    noise = MeasurementNoise(1.0e-5, 2.0e-4, 3.0e-3, 4.0e-4)
    first = generate_measurements(truth, noise, random_seed=102)
    second = generate_measurements(truth, noise, random_seed=102)
    different = generate_measurements(truth, noise, random_seed=103)

    np.testing.assert_array_equal(first.displacement_m, second.displacement_m)
    assert not np.array_equal(first.displacement_m, different.displacement_m)
    assert np.std(first.displacement_m - truth.displacement_m) == pytest.approx(
        noise.displacement_std_m, rel=0.08
    )


def test_batch_least_squares_identifies_all_three_parameters_exactly():
    parameters, truth = simulated_case()
    measured = generate_measurements(truth, MeasurementNoise(), random_seed=104)

    result = batch_least_squares(measured)

    np.testing.assert_allclose(
        result.parameters,
        [
            parameters.stiffness_n_per_m,
            parameters.damping_n_s_per_m,
            parameters.effective_mass_kg,
        ],
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert result.diagnostics.rank == 3


def test_rls_converges_to_the_exact_parameters():
    parameters, truth = simulated_case()
    measured = generate_measurements(truth, MeasurementNoise(), random_seed=105)

    result = recursive_least_squares(measured, initial_covariance=1.0e12)

    np.testing.assert_allclose(
        result.parameters,
        [
            parameters.stiffness_n_per_m,
            parameters.damping_n_s_per_m,
            parameters.effective_mass_kg,
        ],
        rtol=2.0e-8,
        atol=2.0e-8,
    )


def test_single_steady_sinusoid_exposes_mass_stiffness_rank_deficiency():
    time = np.linspace(0.0, 2.0, 2001)
    frequency = 2.0 * np.pi * 3.0
    displacement = np.sin(frequency * time)
    velocity = frequency * np.cos(frequency * time)
    acceleration = -(frequency**2) * displacement
    design = np.column_stack((displacement, velocity, acceleration))

    diagnostics = regression_diagnostics(design)

    assert diagnostics.rank == 2
    assert diagnostics.normalized_condition_number > 1.0e12
