import pytest

from doppler import Capture, Estimator, build_sample
from doppler.estimators import (
    _chao,
    _chapman,
    _percentile,
    bootstrap_interval,
    default_estimator,
    estimate,
    population,
    tabulate,
)


def varied_sample(count=40):
    """Queries whose overlap differs, so resampling them actually moves the estimate."""
    pairs = []
    for index in range(count):
        dense = [f"a{index}", f"b{index}"] if index % 3 else [f"a{index}"]
        bm25 = [f"a{index}", f"c{index}"] if index % 2 else [f"d{index}", f"e{index}"]
        pairs.append((f"q{index}", {"dense": dense, "bm25": bm25}))
    return sample_of(*pairs)


def sample_of(*pairs, stratum="all"):
    captures = []
    for query_id, lists in pairs:
        for source, docs in lists.items():
            captures.append(Capture(query_id, source, docs, stratum=stratum))
    return build_sample(captures)


def test_chapman_matches_the_petersen_ratio_it_corrects():
    # 100 caught, 100 recaught, 50 of them marked: the textbook answer is 200.
    assert _chapman(100, 100, 50) == pytest.approx(200.0, abs=2.0)


def test_chapman_is_defined_when_the_captures_never_overlap():
    assert _chapman(4, 4, 0) == 24.0


def test_chao_adds_nothing_when_everything_was_caught_twice():
    assert _chao(observed=10, singletons=0, doubletons=10) == 10.0


def test_chao_grows_with_the_documents_only_one_source_found():
    assert _chao(observed=10, singletons=6, doubletons=2) > _chao(
        observed=10, singletons=2, doubletons=2
    )


def test_population_is_never_below_what_was_observed():
    sample = sample_of(("q1", {"dense": ["a", "b"], "bm25": ["a", "b"]}))
    table = tabulate(sample.queries, sample.sources)
    assert population(table, Estimator.CHAPMAN) >= table.observed


def test_chapman_refuses_more_than_two_sources():
    sample = sample_of(("q1", {"a": ["x"], "b": ["x"], "c": ["x"]}))
    table = tabulate(sample.queries, sample.sources)
    with pytest.raises(ValueError, match="exactly two sources"):
        population(table, Estimator.CHAPMAN)


def test_default_estimator_follows_the_source_count():
    assert default_estimator(2) is Estimator.CHAPMAN
    assert default_estimator(3) is Estimator.CHAO


def test_tabulate_counts_capture_frequencies_per_query():
    sample = sample_of(
        ("q1", {"dense": ["a", "b"], "bm25": ["a", "c"]}),
        ("q2", {"dense": ["d"], "bm25": ["e"]}),
    )
    table = tabulate(sample.queries, sample.sources)
    assert table.observed == 5
    assert table.doubletons == 1
    assert table.singletons == 4
    assert table.per_source == {"dense": 3, "bm25": 3}


def test_tabulate_respects_the_rank_cutoff():
    sample = sample_of(("q1", {"dense": ["a", "b", "c"], "bm25": ["a"]}))
    assert tabulate(sample.queries, sample.sources, depth=1).per_source["dense"] == 1


def test_documents_shared_across_queries_are_counted_once_per_query():
    sample = sample_of(
        ("q1", {"dense": ["shared"], "bm25": ["shared"]}),
        ("q2", {"dense": ["shared"], "bm25": ["shared"]}),
    )
    assert tabulate(sample.queries, sample.sources).observed == 2


def test_strata_are_estimated_separately_then_summed():
    captures = []
    for index in range(10):
        captures.append(Capture(f"easy{index}", "dense", ["a", "b"], stratum="easy"))
        captures.append(Capture(f"easy{index}", "bm25", ["a", "b"], stratum="easy"))
        captures.append(Capture(f"hard{index}", "dense", ["c"], stratum="hard"))
        captures.append(Capture(f"hard{index}", "bm25", ["d"], stratum="hard"))
    sample = build_sample(captures)
    point = estimate(sample.queries, sample.sources, "dense", Estimator.CHAPMAN)
    by_name = {item.stratum: item for item in point.strata}
    assert by_name["easy"].recall > by_name["hard"].recall
    assert point.population == pytest.approx(
        by_name["easy"].population + by_name["hard"].population
    )


def test_estimate_rejects_a_target_that_is_not_a_source():
    sample = sample_of(("q1", {"dense": ["a"], "bm25": ["a"]}))
    with pytest.raises(ValueError, match="not one of the sources"):
        estimate(sample.queries, sample.sources, "splade", Estimator.CHAPMAN)


def test_the_interval_is_reproducible_from_its_seed():
    sample = varied_sample()
    first = bootstrap_interval(sample.queries, sample.sources, "dense", Estimator.CHAPMAN, seed=7)
    second = bootstrap_interval(sample.queries, sample.sources, "dense", Estimator.CHAPMAN, seed=7)
    other = bootstrap_interval(sample.queries, sample.sources, "dense", Estimator.CHAPMAN, seed=8)
    assert first == second
    assert first != other


def test_a_wider_confidence_gives_a_wider_interval():
    sample = varied_sample()
    narrow = bootstrap_interval(
        sample.queries, sample.sources, "dense", Estimator.CHAPMAN, confidence=0.5
    )
    wide = bootstrap_interval(
        sample.queries, sample.sources, "dense", Estimator.CHAPMAN, confidence=0.99
    )
    assert wide[1] - wide[0] >= narrow[1] - narrow[0]


def test_a_confidence_outside_zero_and_one_is_rejected():
    sample = sample_of(("q1", {"dense": ["a"], "bm25": ["a"]}))
    with pytest.raises(ValueError, match="confidence must lie"):
        bootstrap_interval(
            sample.queries, sample.sources, "dense", Estimator.CHAPMAN, confidence=1.0
        )


def test_percentile_interpolates_between_neighbours():
    assert _percentile([0.0, 1.0], 0.5) == pytest.approx(0.5)
    assert _percentile([0.0, 1.0, 2.0], 1.0) == 2.0
