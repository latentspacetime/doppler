import json

import pytest

from doppler.cli import main, parse_jsonl

LINES = [
    '{"query_id": "q1", "source": "dense", "doc_ids": ["a", "b"], "relevant_doc_ids": ["a"]}',
    '{"query_id": "q1", "source": "bm25", "doc_ids": ["a", "c"],'
    ' "relevant_doc_ids": ["a", "c"], "stratum": "how-to"}',
]


def test_parses_captures_and_their_optional_fields():
    captures = parse_jsonl(("\n".join(LINES) + "\n\n").splitlines())
    assert len(captures) == 2
    assert captures[0].relevant_doc_ids == {"a"}
    assert captures[1].stratum == "how-to"
    assert captures[0].stratum == "all"


def test_names_the_line_that_is_malformed():
    with pytest.raises(ValueError, match="line 2"):
        parse_jsonl((LINES[0] + "\nnot json\n").splitlines())


def test_names_the_line_that_is_missing_a_field():
    with pytest.raises(ValueError, match="line 1: missing doc_ids"):
        parse_jsonl(['{"query_id": "q1", "source": "dense"}'])


def test_names_the_line_whose_relevance_marks_are_not_retrieved():
    with pytest.raises(ValueError, match=r"line 1:.*not all present"):
        parse_jsonl(
            ['{"query_id": "q1", "source": "d", "doc_ids": ["a"], "relevant_doc_ids": ["z"]}']
        )


def test_rejects_input_that_holds_no_captures():
    with pytest.raises(ValueError, match="no captures found"):
        parse_jsonl(["", " "])


def test_demo_runs_and_reports_an_estimate(capsys):
    assert main(["--demo", "--target", "dense"]) == 0
    assert "recall" in capsys.readouterr().out


def test_json_output_is_machine_readable(capsys):
    assert main(["--demo", "--target", "dense", "--resamples", "50", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "estimated"


def test_a_refusal_exits_two_rather_than_zero(capsys):
    assert main(["--demo", "--target", "dense", "--min-queries", "100000"]) == 2
    assert "refused" in capsys.readouterr().out


def test_a_missing_file_is_reported_without_a_traceback(capsys):
    assert main(["nowhere.jsonl", "--target", "dense"]) == 1
    assert "doppler:" in capsys.readouterr().err


def test_a_bad_target_is_reported_without_a_traceback(capsys):
    assert main(["--demo", "--target", "nothing"]) == 1
    assert "not one of the sources" in capsys.readouterr().err


def test_a_file_and_demo_together_are_refused(capsys):
    assert main(["some.jsonl", "--demo", "--target", "dense"]) == 1
    assert "not both" in capsys.readouterr().err


def test_captures_are_read_from_a_file(tmp_path, capsys):
    path = tmp_path / "captures.jsonl"
    rows = []
    for index in range(60):
        rows.append(
            json.dumps(
                {"query_id": f"q{index}", "source": "dense", "doc_ids": [f"a{index}", f"b{index}"]}
            )
        )
        rows.append(
            json.dumps(
                {"query_id": f"q{index}", "source": "bm25", "doc_ids": [f"a{index}", f"c{index}"]}
            )
        )
    path.write_text("\n".join(rows))
    assert main([str(path), "--target", "dense", "--resamples", "50"]) == 0
    assert "coverage of the retrievable pool" in capsys.readouterr().out


def test_a_string_of_lines_is_rejected_because_iterating_one_yields_characters():
    with pytest.raises(ValueError, match="takes lines"):
        parse_jsonl(LINES[0])


def test_a_doc_ids_string_is_named_by_line_rather_than_silently_split():
    with pytest.raises(ValueError, match=r"line 1: doc_ids must be a list"):
        parse_jsonl(['{"query_id": "q1", "source": "dense", "doc_ids": "d12"}'])
