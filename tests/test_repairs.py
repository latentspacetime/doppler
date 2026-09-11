import pytest

from doppler import Capture, estimate_recall, simulate_captures


def test_a_recovery_counts_what_the_other_source_already_found():
    captures = []
    for index in range(60):
        captures.append(Capture(f"q{index}", "dense", [f"a{index}", f"b{index}"]))
        captures.append(Capture(f"q{index}", "bm25", [f"a{index}", f"c{index}"]))
    report = estimate_recall(captures, "dense", resamples=100)
    recovery = report.repairs.recoveries[0]
    assert recovery.source == "bm25"
    assert recovery.recovered == 60
    assert 0 < recovery.share_of_missed <= 1


def test_the_depth_sweep_rises_with_the_cutoff_and_ends_at_the_longest_list():
    simulation = simulate_captures(
        {"dense": 0.55, "bm25": 0.40}, queries=200, relevant_per_query=12
    )
    report = estimate_recall(simulation.captures, "dense", resamples=100)
    sweep = report.repairs.depth_sweep
    assert [point.recall for point in sweep] == sorted(point.recall for point in sweep)
    assert sweep[-1].depth == max(len(c.doc_ids) for c in simulation.captures)
    assert sweep[-1].recall == pytest.approx(report.recall, abs=1e-9)


def test_the_missed_set_is_grouped_by_stratum_largest_first():
    captures = []
    for index in range(60):
        wide = index % 2 == 0
        captures.append(
            Capture(f"q{index}", "dense", [f"a{index}"], stratum="wide" if wide else "narrow")
        )
        extra = [f"b{index}", f"c{index}"] if wide else [f"b{index}"]
        captures.append(
            Capture(
                f"q{index}", "bm25", [f"a{index}", *extra], stratum="wide" if wide else "narrow"
            )
        )
    groups = estimate_recall(captures, "dense", resamples=100).repairs.missed_by_stratum
    assert [group.stratum for group in groups] == ["wide", "narrow"]
    assert groups[0].missed == 60
    assert len(groups[0].examples) <= 5


def test_estimated_missed_is_the_population_the_target_did_not_capture():
    simulation = simulate_captures(
        {"dense": 0.55, "bm25": 0.40}, queries=200, relevant_per_query=12
    )
    report = estimate_recall(simulation.captures, "dense", resamples=100)
    assert report.repairs.estimated_missed == pytest.approx(report.population - report.captured)


def test_the_depth_sweep_stops_at_the_depth_the_estimate_was_made_at():
    """A sweep past the analysed depth divides captures taken at one cutoff by a
    population estimated at another, which reported recall far above 1."""
    captures = []
    for index in range(40):
        deep = [f"d{index}-{rank}" for rank in range(100)]
        captures.append(Capture(f"q{index}", "dense", deep))
        captures.append(Capture(f"q{index}", "bm25", [f"b{index}", f"d{index}-0", f"d{index}-1"]))
    report = estimate_recall(captures, "dense", depth=2, resamples=50)
    sweep = report.repairs.depth_sweep
    assert sweep[-1].depth == 2
    assert all(point.recall <= 1.0 for point in sweep)
    assert sweep[-1].recall == pytest.approx(report.recall, abs=1e-9)
