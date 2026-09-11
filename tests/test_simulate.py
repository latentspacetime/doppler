import pytest

from doppler import simulate_captures


def test_the_trace_is_reproducible_from_its_seed():
    first = simulate_captures({"a": 0.5, "b": 0.4}, queries=20, seed=3)
    second = simulate_captures({"a": 0.5, "b": 0.4}, queries=20, seed=3)
    assert first.captures == second.captures
    assert first.recall == second.recall


def test_a_higher_rate_finds_more_of_the_population():
    simulation = simulate_captures({"strong": 0.7, "weak": 0.2}, queries=200)
    assert simulation.recall["strong"] > simulation.recall["weak"]


def test_reported_recall_matches_the_captures_it_generated():
    simulation = simulate_captures({"a": 0.5, "b": 0.4}, queries=50, relevant_per_query=10)
    found = {
        doc
        for capture in simulation.captures
        if capture.source == "a"
        for doc in capture.captured()
    }
    assert simulation.recall["a"] == pytest.approx(len(found) / simulation.population)


def test_unmarked_traces_carry_distractors_in_the_population():
    marked = simulate_captures({"a": 0.5, "b": 0.4}, queries=10, relevant_per_query=10)
    unmarked = simulate_captures(
        {"a": 0.5, "b": 0.4}, queries=10, relevant_per_query=10, mark_relevance=False
    )
    assert marked.population == 100
    assert unmarked.population > marked.population
    assert all(capture.relevant_doc_ids is None for capture in unmarked.captures)


def test_strata_shift_capture_rates_apart():
    simulation = simulate_captures({"a": 0.5, "b": 0.4}, queries=200, strata=3)
    assert len({capture.stratum for capture in simulation.captures}) == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rates": {"a": 1.5, "b": 0.4}}, "must lie in"),
        ({"rates": {"a": 0.5}}, "at least two sources"),
        ({"rates": {"a": 0.5, "b": 0.4}, "queries": 0}, "must all be positive"),
        ({"rates": {"a": 0.5, "b": 0.4}, "heterogeneity": -1.0}, "must not be negative"),
    ],
)
def test_impossible_simulations_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        simulate_captures(**kwargs)
