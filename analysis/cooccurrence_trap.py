import numpy as np


def wilson_ci(correct_series, z=1.96):
    n = len(correct_series)
    k = int(correct_series.sum())
    if n == 0:
        return np.nan, np.nan, np.nan
    p      = k / n
    denom  = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half   = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    acc    = round(p * 100, 2)
    lo     = round(max(0, centre - half) * 100, 2)
    hi     = round(min(1, centre + half) * 100, 2)
    return acc, lo, hi


def fmt_ci(acc, lo, hi):
    if np.isnan(acc):
        return "N/A"
    return f"{acc:.1f}% ({lo:.1f}–{hi:.1f})"