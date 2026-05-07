import sys
sys.path.insert(0, ".")

import pandas as pd
from tiers.tier_analysis import assign_tier_aggregate


def test_t1():
    assert assign_tier_aggregate(0.70) == "T1"
    assert assign_tier_aggregate(1.00) == "T1"
    assert assign_tier_aggregate(0.85) == "T1"


def test_t2():
    assert assign_tier_aggregate(0.40) == "T2"
    assert assign_tier_aggregate(0.55) == "T2"
    assert assign_tier_aggregate(0.699) == "T2"


def test_t3():
    assert assign_tier_aggregate(0.10) == "T3"
    assert assign_tier_aggregate(0.25) == "T3"
    assert assign_tier_aggregate(0.399) == "T3"


def test_t4():
    assert assign_tier_aggregate(0.00) == "T4"
    assert assign_tier_aggregate(0.05) == "T4"
    assert assign_tier_aggregate(0.099) == "T4"


def test_boundary_exactly_at_threshold():
    assert assign_tier_aggregate(0.70) == "T1"
    assert assign_tier_aggregate(0.40) == "T2"
    assert assign_tier_aggregate(0.10) == "T3"


if __name__ == "__main__":
    test_t1()
    test_t2()
    test_t3()
    test_t4()
    test_boundary_exactly_at_threshold()
    print("All tier calibration tests passed.")