"""Independent MATLAB oracle at fixed states, without testing solver parity."""
from pathlib import Path
import numpy as np
import pytest
from sparksimple.features.robust import robust_r2,bisquare_weights


@pytest.mark.parametrize('i',range(5))
def test_fixed_matlab_robust_state(i):
    with np.load(Path(__file__).parents[1]/'fixtures/robust_state_controls.npz') as m:
        actual=robust_r2(m[f'y_{i}'],m[f'residuals_{i}'],m[f'weights_{i}'],m[f'h_{i}'],
                         float(m[f'scale_{i}']),float(m[f'initial_sse_{i}']),int(m[f'p_{i}']))
        assert actual==pytest.approx(float(m[f'r2_{i}']),abs=2e-12,rel=0)
        weights=bisquare_weights(m[f'pre_residuals_{i}']/np.sqrt(1-m[f'h_{i}'])/(4.685*float(m[f'scale_{i}'])))
        np.testing.assert_allclose(weights,m[f'expected_weights_{i}'],atol=1e-14,rtol=0)
