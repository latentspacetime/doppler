"""The claim this library makes, checked against populations of known size."""

import json

import pytest

from doppler import Capture, Estimator, estimate_recall, simulate_captures
from doppler.recall import Refusal

RATES = {"dense": 0.55, "bm25": 0.40}


def test_recovers_a_known_recall_when_the_assumptions_hold():
    simulation = simulate_captures(RATES, queries=300, relevant_per_query=12, heterogeneity=0.0)
    report = estimate_recall(simulation.captures, "dense", resamples=400)
    assert report.recall == pytest.approx(simulation.recall["dense"], abs=0.05)


def test_the_interval_covers_the_truth_when_the_assumptions_hold():
    covered = 0
    for seed in range(10):
        simulation = simulate_captures(
            RATES, queries=200, relevant_per_query=12, heterogeneity=0.0, seed=seed
        )
        report = estimate_recall(simulation.captures, "dense", resamples=300, seed=seed)
        low, high = report.interval
        covered += low <= simulation.recall["dense"] <= high
    assert covered >= 8


def test_heterogeneous_catchability_biases_recall_upward_as_documented():
    truths, estimates = [], []
    for heterogeneity in (0.0, 1.6):
        simulation = simulate_captures(
            RATES, queries=300, relevant_per_query=12, heterogeneity=heterogeneity
        )
        truths.append(simulation.recall["dense"])
        estimates.append(estimate_recall(simulation.captures, "dense", resamples=200).recall)
    assert truths[1] - truths[0] < 0.1, "the simulated truth barely moves"
    assert estimates[1] - estimates[0] > 0.15, "the estimate should drift upward, not stay put"


def test_a_third_source_holds_the_estimate_under_heterogeneity():
    rates = {**RATES, "splade": 0.45}
    simulation = simulate_captures(rates, queries=300, relevant_per_query=12, heterogeneity=0.6)
    report = estimate_recall(simulation.captures, "dense", resamples=300)
    assert report.estimator is Estimator.CHAO
    assert report.recall == pytest.approx(simulation.recall["dense"], abs=0.06)


def test_a_weaker_source_is_estimated_at_a_lower_recall():
    simulation = simulate_captures(RATES, queries=300, relevant_per_query=12)
    dense = estimate_recall(simulation.captures, "dense", resamples=200)
    bm25 = estimate_recall(simulation.captures, "bm25", resamples=200)
    assert dense.recall > bm25.recall
    assert dense.population == pytest.approx(bm25.population)


def test_unmarked_captures_estimate_pool_coverage_not_relevance():
    simulation = simulate_captures(RATES, queries=300, relevant_per_query=12, mark_relevance=False)
    report = estimate_recall(simulation.captures, "dense", resamples=200)
    assert report.estimand.value == "pool"
    assert report.recall == pytest.approx(simulation.recall["dense"], abs=0.06)


def test_a_rank_cutoff_lowers_recall_and_is_recorded():
    simulation = simulate_captures(RATES, queries=300, relevant_per_query=12)
    full = estimate_recall(simulation.captures, "dense", resamples=200)
    shallow = estimate_recall(simulation.captures, "dense", depth=2, resamples=200)
    assert shallow.recall < full.recall
    assert shallow.depth == 2


def test_a_target_outside_the_sample_is_a_caller_error_not_a_refusal():
    simulation = simulate_captures(RATES, queries=40, relevant_per_query=12)
    with pytest.raises(ValueError, match="not one of the sources"):
        estimate_recall(simulation.captures, "splade")


def test_a_depth_below_one_is_rejected():
    simulation = simulate_captures(RATES, queries=40, relevant_per_query=12)
    with pytest.raises(ValueError, match="depth must be at least 1"):
        estimate_recall(simulation.captures, "dense", depth=0)


def test_an_estimate_serialises_to_json():
    simulation = simulate_captures(RATES, queries=60, relevant_per_query=12)
    payload = json.loads(
        json.dumps(estimate_recall(simulation.captures, "dense", resamples=100).to_dict())
    )
    assert payload["verdict"] == "estimated"
    assert payload["estimand"] == "relevant"
    assert len(payload["interval"]) == 2
    assert payload["repairs"]["recoveries"][0]["source"] == "bm25"


def test_a_refusal_serialises_to_json_with_its_reason():
    simulation = simulate_captures(RATES, queries=4, relevant_per_query=12)
    report = estimate_recall(simulation.captures, "dense")
    assert isinstance(report, Refusal)
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["verdict"] == "refused"
    assert payload["reason"] == "too_few_queries"


def test_chapman_is_refused_before_any_work_when_there_are_three_sources():
    rates = {**RATES, "splade": 0.45}
    simulation = simulate_captures(rates, queries=40, relevant_per_query=12)
    with pytest.raises(ValueError, match="takes exactly two sources"):
        estimate_recall(simulation.captures, "dense", estimator=Estimator.CHAPMAN)


def test_a_stratum_where_nothing_was_retrieved_reports_zero_rather_than_dividing_by_zero():
    captures = []
    for index in range(60):
        captures.append(Capture(f"r{index}", "dense", [f"a{index}", f"b{index}"], stratum="rich"))
        captures.append(Capture(f"r{index}", "bm25", [f"a{index}", f"c{index}"], stratum="rich"))
    for index in range(10):
        captures.append(Capture(f"e{index}", "dense", [], stratum="empty"))
        captures.append(Capture(f"e{index}", "bm25", [], stratum="empty"))
    report = estimate_recall(captures, "dense", resamples=50)
    empty = next(item for item in report.strata if item.stratum == "empty")
    assert empty.population == 0
    assert empty.recall == 0.0
    assert report.recall > 0
