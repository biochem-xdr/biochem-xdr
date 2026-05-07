import sys
sys.path.insert(0, ".")

from evaluation.compute_stats import wilson_ci, fmt_ci
import numpy as np


def test_wilson_perfect():
    acc, lo, hi = wilson_ci(pd_series(10, 10))
    assert acc == 100.0
    assert lo < 100.0
    assert hi == 100.0


def test_wilson_zero():
    acc, lo, hi = wilson_ci(pd_series(0, 10))
    assert acc == 0.0
    assert lo == 0.0
    assert hi > 0.0


def test_wilson_half():
    acc, lo, hi = wilson_ci(pd_series(5, 10))
    assert abs(acc - 50.0) < 0.01
    assert lo < 50.0
    assert hi > 50.0


def test_wilson_empty():
    acc, lo, hi = wilson_ci(pd_series(0, 0))
    assert np.isnan(acc)


def test_fmt_ci():
    assert fmt_ci(50.0, 30.0, 70.0) == "50.0% (30.0–70.0)"
    assert fmt_ci(float("nan"), 0, 0) == "N/A"


def pd_series(correct, total):
    import pandas as pd
    vals = [True] * correct + [False] * (total - correct)
    return pd.Series(vals)


if __name__ == "__main__":
    test_wilson_perfect()
    test_wilson_zero()
    test_wilson_half()
    test_wilson_empty()
    test_fmt_ci()
    print("All stats tests passed.")