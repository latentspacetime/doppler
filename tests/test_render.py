from doppler import estimate_recall, render, simulate_captures

RATES = {"dense": 0.55, "bm25": 0.40}


def test_an_estimate_renders_the_number_the_interval_and_the_estimand():
    simulation = simulate_captures(RATES, queries=200, relevant_per_query=12, strata=2)
    text = render(estimate_recall(simulation.captures, "dense", resamples=100))
    assert "recall of 'dense'" in text
    assert "relevance recall" in text
    assert "95% interval" in text
    assert "by stratum" in text
    assert "missed set" in text
    assert "recall against rank cutoff" in text


def test_pool_coverage_is_never_rendered_as_recall_over_relevant_documents():
    simulation = simulate_captures(RATES, queries=200, relevant_per_query=12, mark_relevance=False)
    text = str(estimate_recall(simulation.captures, "dense", resamples=100))
    assert "coverage of the retrievable pool" in text


def test_a_refusal_renders_the_reason_and_no_recall():
    simulation = simulate_captures(RATES, queries=4, relevant_per_query=12)
    text = str(estimate_recall(simulation.captures, "dense"))
    assert "no estimate" in text
    assert "too_few_queries" in text
    assert "recall  " not in text
