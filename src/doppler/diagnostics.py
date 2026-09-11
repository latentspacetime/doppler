"""What the sample can and cannot support, decided before any recall is reported.

Capture-recapture fails quietly. Two sources that return nearly the same
documents produce a tight, confident, wrong interval, because the estimator
reads their agreement as evidence that little was missed. Every check that
would make the estimate fiction lives here, and the answer to a failed check is
a refusal rather than a number with a caveat attached.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations

from .captures import QueryCaptures
from .estimators import CaptureTable


class RefusalReason(str, Enum):
    """Why no recall estimate was produced."""

    NO_CAPTURES = "no_captures"
    """No source captured anything, so there is no population to estimate."""

    TOO_FEW_QUERIES = "too_few_queries"
    """The query sample is too small for the interval to mean anything."""

    INSUFFICIENT_RECAPTURE = "insufficient_recapture"
    """Almost nothing was captured twice, so the overlap carries no signal
    about what was missed."""

    SOURCES_TOO_SIMILAR = "sources_too_similar"
    """The sources return nearly the same documents. Their agreement is
    evidence about each other, not about the population."""


@dataclass(frozen=True)
class Thresholds:
    """The bar a sample has to clear before recall is estimated at all.

    Defaults are deliberately conservative: a refusal costs a larger query
    sample, and a bad estimate costs a team weeks spent rebuilding the wrong
    half of their system.
    """

    min_queries: int = 30
    min_recaptures: int = 10
    max_overlap: float = 0.8


@dataclass(frozen=True)
class PairOverlap:
    """Pooled Jaccard overlap between two sources."""

    sources: tuple[str, str]
    jaccard: float


@dataclass(frozen=True)
class Diagnostics:
    """Everything measured about the sample that is not the estimate itself."""

    queries: int
    excluded_queries: int
    observed: int
    recaptured: int
    capture_frequency: tuple[int, ...]
    per_source: dict[str, int]
    overlaps: tuple[PairOverlap, ...]
    dependence_identifiable: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def max_overlap(self) -> PairOverlap | None:
        return max(self.overlaps, key=lambda pair: pair.jaccard, default=None)

    def to_dict(self) -> dict:
        return {
            "queries": self.queries,
            "excluded_queries": self.excluded_queries,
            "observed": self.observed,
            "recaptured": self.recaptured,
            "capture_frequency": list(self.capture_frequency),
            "per_source": dict(self.per_source),
            "overlaps": [
                {"sources": list(pair.sources), "jaccard": pair.jaccard} for pair in self.overlaps
            ],
            "dependence_identifiable": self.dependence_identifiable,
            "notes": list(self.notes),
        }


def measure(
    queries: Sequence[QueryCaptures],
    sources: Sequence[str],
    table: CaptureTable,
    excluded_queries: int,
    depth: int | None = None,
) -> Diagnostics:
    """Measure sample health, including the dependence the estimator assumes away."""
    recaptured = sum(table.frequency[1:])
    identifiable = len(sources) >= 3
    notes = []
    if not identifiable:
        notes.append(
            "Dependence between two sources cannot be identified from two sources. "
            "If they tend to find the same documents for reasons other than "
            "relevance, the population is underestimated and this recall is an "
            "upper bound. Add a third, differently built source to identify it."
        )
    if excluded_queries:
        notes.append(
            f"{excluded_queries} queries were excluded because at least one source "
            "did not answer them; a query only carries recapture information when "
            "every source attempted it."
        )
    return Diagnostics(
        queries=len(queries),
        excluded_queries=excluded_queries,
        observed=table.observed,
        recaptured=recaptured,
        capture_frequency=table.frequency,
        per_source=dict(table.per_source),
        overlaps=_overlaps(queries, sources, depth),
        dependence_identifiable=identifiable,
        notes=tuple(notes),
    )


def refuse(diagnostics: Diagnostics, thresholds: Thresholds) -> tuple[RefusalReason, str] | None:
    """Return the reason this sample cannot support an estimate, or None."""
    if diagnostics.observed == 0:
        return (
            RefusalReason.NO_CAPTURES,
            "No source captured a document across the whole sample.",
        )
    if diagnostics.queries < thresholds.min_queries:
        return (
            RefusalReason.TOO_FEW_QUERIES,
            f"{diagnostics.queries} queries is below the minimum of "
            f"{thresholds.min_queries}; sample more queries.",
        )
    if diagnostics.recaptured < thresholds.min_recaptures:
        return (
            RefusalReason.INSUFFICIENT_RECAPTURE,
            f"only {diagnostics.recaptured} documents were captured by more than one "
            f"source, below the minimum of {thresholds.min_recaptures}; with this "
            "little overlap the estimate is driven by the correction term rather "
            "than the data. Retrieve deeper, or sample more queries.",
        )
    worst = diagnostics.max_overlap
    if worst is not None and worst.jaccard >= thresholds.max_overlap:
        first, second = worst.sources
        return (
            RefusalReason.SOURCES_TOO_SIMILAR,
            f"{first} and {second} agree on {worst.jaccard:.0%} of what they return "
            f"(Jaccard {worst.jaccard:.2f}, at or above the limit of "
            f"{thresholds.max_overlap:.2f}). Near-identical sources make the missed "
            "set look empty. Compare sources built on different principles, such as "
            "a dense retriever against a lexical one.",
        )
    return None


def _overlaps(
    queries: Sequence[QueryCaptures],
    sources: Sequence[str],
    depth: int | None,
) -> tuple[PairOverlap, ...]:
    """Jaccard overlap per source pair, pooled over queries.

    Pooled rather than averaged per query so that queries with large retrieved
    sets weigh more than queries with almost nothing to compare.
    """
    intersections: dict[tuple[str, str], int] = {}
    unions: dict[tuple[str, str], int] = {}
    for query in queries:
        marked = query.marked(depth)
        for pair in combinations(sources, 2):
            first = marked.get(pair[0], frozenset())
            second = marked.get(pair[1], frozenset())
            intersections[pair] = intersections.get(pair, 0) + len(first & second)
            unions[pair] = unions.get(pair, 0) + len(first | second)
    return tuple(
        PairOverlap(
            sources=pair, jaccard=intersections[pair] / unions[pair] if unions[pair] else 0.0
        )
        for pair in combinations(sources, 2)
    )
