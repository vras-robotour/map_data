from scripts.bench.mapload import _stats


def test_stats_reports_median_and_median_absolute_deviation():
    assert _stats([1.0, 2.0, 3.0]) == (2.0, 1.0)
