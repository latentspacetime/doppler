import pytest

from doppler import Capture, Estimand, build_sample


def test_repeated_doc_is_one_capture_and_keeps_rank_order():
    capture = Capture("q1", "dense", ["b", "a", "b", "c"])
    assert capture.doc_ids == ("b", "a", "c")


def test_relevance_marks_outside_the_retrieved_list_are_rejected():
    with pytest.raises(ValueError, match="not all present in doc_ids"):
        Capture("q1", "dense", ["a"], relevant_doc_ids=["a", "z"])


def test_capture_at_depth_keeps_only_marked_documents_above_the_cutoff():
    capture = Capture("q1", "dense", ["a", "b", "c"], relevant_doc_ids=["a", "c"])
    assert capture.captured(depth=2) == {"a"}
    assert capture.captured() == {"a", "c"}


def test_unmarked_capture_at_depth_keeps_every_document_above_the_cutoff():
    capture = Capture("q1", "dense", ["a", "b", "c"])
    assert capture.captured(depth=2) == {"a", "b"}


def test_empty_identifiers_are_rejected():
    with pytest.raises(ValueError, match="source must be a non-empty string"):
        Capture("q1", "  ", ["a"])


def test_estimand_is_relevance_when_every_capture_is_marked():
    sample = build_sample(
        [
            Capture("q1", "dense", ["a"], relevant_doc_ids=["a"]),
            Capture("q1", "bm25", ["a", "b"], relevant_doc_ids=["a"]),
        ]
    )
    assert sample.estimand is Estimand.RELEVANT


def test_estimand_is_pool_when_no_capture_is_marked():
    sample = build_sample([Capture("q1", "dense", ["a"]), Capture("q1", "bm25", ["a", "b"])])
    assert sample.estimand is Estimand.POOL


def test_mixing_marked_and_unmarked_captures_is_rejected():
    with pytest.raises(ValueError, match="some captures and absent on others"):
        build_sample(
            [
                Capture("q1", "dense", ["a"], relevant_doc_ids=["a"]),
                Capture("q1", "bm25", ["a"]),
            ]
        )


def test_a_source_answering_a_query_twice_is_rejected():
    with pytest.raises(ValueError, match="duplicate capture"):
        build_sample(
            [
                Capture("q1", "dense", ["a"]),
                Capture("q1", "dense", ["b"]),
                Capture("q1", "bm25", ["a"]),
            ]
        )


def test_a_query_in_two_strata_is_rejected():
    with pytest.raises(ValueError, match="belongs to one stratum"):
        build_sample(
            [
                Capture("q1", "dense", ["a"], stratum="how-to"),
                Capture("q1", "bm25", ["a"], stratum="policy"),
            ]
        )


def test_queries_missing_a_source_are_excluded_and_counted():
    sample = build_sample(
        [
            Capture("q1", "dense", ["a"]),
            Capture("q1", "bm25", ["a"]),
            Capture("q2", "dense", ["b"]),
        ]
    )
    assert [query.query_id for query in sample.queries] == ["q1"]
    assert sample.excluded_query_ids == ("q2",)


def test_one_source_cannot_recapture_itself():
    with pytest.raises(ValueError, match="at least two sources"):
        build_sample([Capture("q1", "dense", ["a"]), Capture("q2", "dense", ["b"])])


def test_no_captures_at_all_is_rejected():
    with pytest.raises(ValueError, match="no captures given"):
        build_sample([])
