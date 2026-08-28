import numpy as np
import pytest

from probeing.models import KelvinVoigtModel, KelvinVoigtParameters


@pytest.mark.parametrize(
    "stiffness,damping",
    [(0.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (np.inf, 1.0), (1.0, np.nan)],
)
def test_invalid_parameters_are_rejected(stiffness, damping):
    with pytest.raises(ValueError):
        KelvinVoigtParameters(stiffness, damping)


def test_bilateral_response_matches_constitutive_equation_and_components():
    model = KelvinVoigtModel(KelvinVoigtParameters(400.0, 12.0))
    indentation = np.array([0.0, 0.001, 0.004, 0.002])
    velocity = np.array([0.0, 0.02, 0.0, -0.01])

    response = model.response(indentation, velocity)

    np.testing.assert_allclose(response.elastic_force_n, 400.0 * indentation)
    np.testing.assert_allclose(response.damping_force_n, 12.0 * velocity)
    np.testing.assert_allclose(
        response.force_n, 400.0 * indentation + 12.0 * velocity
    )
    assert np.all(response.contact_active)


def test_unilateral_response_never_generates_tension_or_force_outside_contact():
    model = KelvinVoigtModel(
        KelvinVoigtParameters(100.0, 20.0), contact_mode="unilateral"
    )
    indentation = np.array([-0.001, 0.0, 0.001, 0.001])
    velocity = np.array([0.1, 0.1, -0.1, 0.1])

    response = model.response(indentation, velocity)

    np.testing.assert_allclose(response.force_n, [0.0, 2.0, 0.0, 2.1])
    assert np.all(response.force_n >= 0.0)
    np.testing.assert_array_equal(response.contact_active, [False, True, False, True])


def test_non_finite_trajectory_samples_are_rejected():
    model = KelvinVoigtModel(KelvinVoigtParameters(100.0, 1.0))
    with pytest.raises(ValueError):
        model.response([0.0, np.nan], [0.0, 0.0])

