from doppler import Capture, estimate_recall, render, simulate_captures

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


def test_a_long_stratum_name_is_cut_to_the_column():
    captures = []
    long_name = "product-area/" + "x" * 60
    for index in range(60):
        for stratum in (long_name, "short"):
            captures.append(
                Capture(f"{stratum}{index}", "dense", [f"a{index}", f"b{index}"], stratum=stratum)
            )
            captures.append(
                Capture(f"{stratum}{index}", "bm25", [f"a{index}", f"c{index}"], stratum=stratum)
            )
    text = str(estimate_recall(captures, "dense", resamples=50))
    assert long_name not in text
    assert max(len(line) for line in text.splitlines()) < 100


def test_a_small_recall_still_draws_a_visible_bar():
    simulation = simulate_captures(RATES, queries=200, relevant_per_query=12)
    text = str(estimate_recall(simulation.captures, "dense", resamples=50))
    bars = [line for line in text.splitlines() if line.strip().startswith("k=")]
    assert bars
    assert all(line.split()[1].strip() for line in bars)
