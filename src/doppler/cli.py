"""Command line entry point.

Reads captures as JSON Lines, one capture per line:

    {"query_id": "q1", "source": "dense", "doc_ids": ["a", "b"],
     "relevant_doc_ids": ["a"], "stratum": "how-to"}

``relevant_doc_ids`` and ``stratum`` are optional. Supply relevance marks on
every line or on none of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .captures import DEFAULT_STRATUM, Capture
from .diagnostics import Thresholds
from .estimators import Estimator
from .recall import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, Refusal, estimate_recall
from .simulate import Simulation, simulate_captures

REFUSED = 2
"""Exit code for a refusal: not a crash, and not an answer either."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        captures, simulation = _load(args)
    except (OSError, ValueError) as error:
        print(f"doppler: {error}", file=sys.stderr)
        return 1

    try:
        report = estimate_recall(
            captures,
            args.target,
            depth=args.depth,
            estimator=Estimator(args.estimator) if args.estimator else None,
            confidence=args.confidence,
            resamples=args.resamples,
            seed=args.seed,
            thresholds=Thresholds(
                min_queries=args.min_queries,
                min_recaptures=args.min_recaptures,
                max_overlap=args.max_overlap,
            ),
        )
    except ValueError as error:
        print(f"doppler: {error}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report)
        if simulation is not None:
            # The demo is the one case where the answer is known, so it is
            # printed next to the estimate rather than described in prose.
            print(
                f"\n  true recall of {args.target!r} in this simulated trace: "
                f"{simulation.recall[args.target]:.2f}"
            )
    return REFUSED if isinstance(report, Refusal) else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="doppler-recall",
        description="Estimate retrieval recall from two or more retrievers, without labels.",
    )
    parser.add_argument("--version", action="version", version=f"doppler {__version__}")
    parser.add_argument(
        "captures",
        nargs="?",
        help="JSON Lines file of captures, or - for standard input. "
        "Omit it with --demo to run on a simulated trace.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="estimate over a simulated trace whose true recall is known",
    )
    parser.add_argument("--target", required=True, help="the source being measured")
    parser.add_argument("--depth", type=int, help="rank cutoff applied to every list")
    parser.add_argument(
        "--estimator",
        choices=[member.value for member in Estimator],
        help="population estimator (default: chapman at two sources, chao above two)",
    )
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="emit the full report as JSON")

    limits = parser.add_argument_group("refusal thresholds")
    defaults = Thresholds()
    limits.add_argument("--min-queries", type=int, default=defaults.min_queries)
    limits.add_argument("--min-recaptures", type=int, default=defaults.min_recaptures)
    limits.add_argument("--max-overlap", type=float, default=defaults.max_overlap)
    return parser


def _load(args: argparse.Namespace) -> tuple[list[Capture], Simulation | None]:
    """Load captures, and the simulation they came from when there is one."""
    if args.demo:
        if args.captures:
            raise ValueError("give a captures file or --demo, not both")
        simulation = simulate_captures(
            {"dense": 0.55, "bm25": 0.40},
            queries=300,
            relevant_per_query=12,
            heterogeneity=0.6,
            strata=3,
        )
        return list(simulation.captures), simulation
    if not args.captures:
        raise ValueError("no captures given; pass a JSON Lines file, - for stdin, or --demo")
    text = sys.stdin.read() if args.captures == "-" else Path(args.captures).read_text()
    return parse_jsonl(text), None


def parse_jsonl(text: str) -> list[Capture]:
    """Parse capture records, naming the line that is wrong when one is.

    Raises:
        ValueError: On malformed JSON, a missing field, or a field of the wrong
            type. Line numbers are 1-based so they match an editor.
    """
    captures = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"line {number}: {error.msg}") from error
        if not isinstance(record, dict):
            raise ValueError(f"line {number}: expected a JSON object, got {type(record).__name__}")
        missing = {"query_id", "source", "doc_ids"} - record.keys()
        if missing:
            raise ValueError(f"line {number}: missing {', '.join(sorted(missing))}")
        try:
            captures.append(
                Capture(
                    query_id=record["query_id"],
                    source=record["source"],
                    doc_ids=record["doc_ids"],
                    relevant_doc_ids=record.get("relevant_doc_ids"),
                    stratum=record.get("stratum", DEFAULT_STRATUM),
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"line {number}: {error}") from error
    if not captures:
        raise ValueError("no captures found in the input")
    return captures


if __name__ == "__main__":
    raise SystemExit(main())
