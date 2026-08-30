import numpy as np
import pytest

from probeing.probing import chirp, force_ramp, half_sine_pulse, multisine, sinusoid


@pytest.mark.parametrize(
    "probe",
    [
        force_ramp(1.0, 0.5, 2.0, 0.001),
        half_sine_pulse(1.0, 0.75, 2.0, 0.001),
        sinusoid(1.0, 2.0, 2.0, 0.001),
        chirp(1.0, 0.5, 8.0, 2.0, 0.001),
        multisine(1.0, [0.5, 1.5, 4.0], 2.0, 0.001),
    ],
)
def test_all_force_probes_respect_declared_bound(probe):
    assert probe.time_s[0] == pytest.approx(0.0)
    assert probe.time_s[-1] == pytest.approx(2.0)
    assert np.max(np.abs(probe.force_n)) <= probe.amplitude_bound_n + 1.0e-14


def test_ramp_and_half_sine_have_expected_end_conditions():
    ramp = force_ramp(2.0, 0.5, 1.0, 0.001)
    pulse = half_sine_pulse(2.0, 0.5, 1.0, 0.001)

    assert ramp.force_n[0] == 0.0
    assert ramp.force_n[-1] == pytest.approx(2.0)
    assert pulse.force_n[0] == 0.0
    assert pulse.force_n[500] == 0.0
    assert np.all(pulse.force_n[501:] == 0.0)


def test_multisine_rejects_an_empty_frequency_set():
    with pytest.raises(ValueError, match="non-empty"):
        multisine(1.0, [], 1.0, 0.001)
