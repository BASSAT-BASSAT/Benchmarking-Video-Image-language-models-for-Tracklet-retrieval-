from shawaf_vlm.bootstrap import bootstrap_mean, bootstrap_metric_from_ranks, format_ci


def test_bootstrap_mean_ci_contains_empirical_mean() -> None:
    values = [1, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    result = bootstrap_mean(values, n_boot=500, seed=0)
    assert result["low"] <= result["mean"] <= result["high"]
    assert "51" in format_ci(result) or "50" in format_ci(result) or "60" in format_ci(result)


def test_bootstrap_from_ranks() -> None:
    ranks = [1, 2, 1, 5, 1, 3, 1, 8]
    result = bootstrap_metric_from_ranks(ranks, rank_k=1, n_boot=400, seed=1)
    assert result["mean"] == 0.5
    assert result["low"] <= 0.5 <= result["high"]
