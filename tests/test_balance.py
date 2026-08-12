

def test_stabilogram_warns_on_a_missing_open_loop_region():
    """H_short below 0.5 is usually the resampling; the caller should be told to check."""
    import numpy as np
    import pytest
    import micromotion as mm

    # a series with no short-lag persistence at all: independent samples
    rng = np.random.default_rng(11)
    xy = rng.normal(size=(4000, 2))
    with pytest.warns(RuntimeWarning, match="open-loop"):
        mm.stabilogram_diffusion(xy, 50.0)


def test_stabilogram_is_silent_on_a_normal_postural_series():
    import numpy as np
    import warnings
    import micromotion as mm

    rng = np.random.default_rng(12)
    xy = np.cumsum(rng.normal(size=(4000, 2)), axis=0)   # persistent at short lags
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mm.stabilogram_diffusion(xy, 50.0)
