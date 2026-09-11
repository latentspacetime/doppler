import pytest

from doppler import Capture, RefusalReason, Thresholds, estimate_recall
from doppler.recall import Refusal


def trace(queries, dense, bm25, stratum="all"):
    captures = []
    for index in range(queries):
        captures.append(
            Capture(f"q{index}", "dense", [f"{tag}{index}" for tag in dense], stratum=stratum)
        )
        captures.append(
            Capture(f"q{index}", "bm25", [f"{tag}{index}" for tag in bm25], stratum=stratum)
        )
    return captures


def test_a_small_query_sample_is_refused():
    report = estimate_recall(trace(5, ["a", "b"], ["a", "c"]), "dense")
    assert isinstance(report, Refusal)
    assert report.reason is RefusalReason.TOO_FEW_QUERIES
    assert "below the minimum of 30" in report.detail


def test_sources_that_never_overlap_are_refused():
    report = estimate_recall(trace(60, ["a", "b"], ["c", "d"]), "dense")
    assert isinstance(report, Refusal)
    assert report.reason is RefusalReason.INSUFFICIENT_RECAPTURE


def test_near_identical_sources_are_refused_rather_than_reported_as_perfect_recall():
    shared = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    report = estimate_recall(trace(60, [*shared, "j"], [*shared, "k"]), "dense")
    assert isinstance(report, Refusal)
    assert report.reason is RefusalReason.SOURCES_TOO_SIMILAR
    assert "Jaccard 0.82" in report.detail


def test_empty_retrievals_are_refused_rather_than_scored():
    captures = []
    for index in range(60):
        captures.append(Capture(f"q{index}", "dense", []))
        captures.append(Capture(f"q{index}", "bm25", []))
    report = estimate_recall(captures, "dense")
    assert isinstance(report, Refusal)
    assert report.reason is RefusalReason.NO_CAPTURES


def test_thresholds_can_be_relaxed_deliberately():
    captures = trace(5, ["a", "b"], ["a", "c"])
    assert isinstance(estimate_recall(captures, "dense"), Refusal)
    relaxed = estimate_recall(
        captures, "dense", thresholds=Thresholds(min_queries=3, min_recaptures=3)
    )
    assert relaxed.verdict == "estimated"


def test_two_sources_always_carry_the_unidentifiable_dependence_note():
    report = estimate_recall(trace(60, ["a", "b"], ["a", "c"]), "dense")
    assert any("cannot be identified" in note for note in report.diagnostics.notes)


def test_three_sources_identify_dependence_and_drop_the_note():
    captures = []
    for index in range(60):
        captures.append(Capture(f"q{index}", "dense", [f"a{index}", f"b{index}"]))
        captures.append(Capture(f"q{index}", "bm25", [f"a{index}", f"c{index}"]))
        captures.append(Capture(f"q{index}", "splade", [f"a{index}", f"d{index}"]))
    report = estimate_recall(captures, "dense")
    assert report.diagnostics.dependence_identifiable
    assert not any("cannot be identified" in note for note in report.diagnostics.notes)


def test_excluded_queries_are_reported_not_hidden():
    captures = trace(60, ["a", "b"], ["a", "c"])
    captures.append(Capture("lonely", "dense", ["z"]))
    report = estimate_recall(captures, "dense")
    assert report.diagnostics.excluded_queries == 1
    assert any("excluded" in note for note in report.diagnostics.notes)


def test_overlap_is_pooled_over_queries():
    report = estimate_recall(trace(60, ["a", "b"], ["a", "c"]), "dense")
    overlap = report.diagnostics.max_overlap
    assert overlap.sources == ("dense", "bm25")
    assert overlap.jaccard == pytest.approx(1 / 3)


def test_every_report_says_the_interval_does_not_cover_estimator_bias():
    report = estimate_recall(trace(60, ["a", "b"], ["a", "c"]), "dense")
    assert any("does not cover estimator bias" in note for note in report.diagnostics.notes)


def test_a_stratum_too_small_to_resample_is_named():
    captures = trace(60, ["a", "b"], ["a", "c"], stratum="main")
    captures.append(Capture("lonely", "dense", ["z1", "z2"], stratum="lonely"))
    captures.append(Capture("lonely", "bm25", ["z1", "z3"], stratum="lonely"))
    report = estimate_recall(captures, "dense", resamples=50)
    note = next(note for note in report.diagnostics.notes if "fewer than 5 queries" in note)
    assert "lonely" in note
    assert "adds no spread" in note
