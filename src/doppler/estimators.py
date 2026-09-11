"""Capture-recapture estimators and the resampled interval around them.

Two sources retrieving for the same query are two attempts to capture the same
unknown population. What each source caught, and how much the catches overlap,
is enough to estimate how large the population was, including the part that
neither source ever returned.

Each query is summarised once into a `CaptureTable` of plain counts. Everything
after that, including a thousand bootstrap resamples, adds those counts instead
of walking the document sets again.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from .captures import QueryCaptures

DEFAULT_CONFIDENCE = 0.95
DEFAULT_RESAMPLES = 1000


class Estimator(str, Enum):
    """Which population estimator to use."""

    CHAPMAN = "chapman"
    """Bias-corrected Lincoln-Petersen. Two sources only. Assumes every
    document is equally catchable. Measured against known populations, it
    underestimates the population as catchability grows uneven, so recall
    reads high and the error has one direction."""

    CHAO = "chao"
    """Chao's bias-corrected estimator from capture frequencies. Works for any
    number of sources and tolerates unequal catchability. Measured against
    known populations, it overestimates the population when catchability is
    even and underestimates it when catchability is strongly uneven, so its
    error changes sign across that range. See the accuracy table in the
    README for the measured amounts."""


def default_estimator(source_count: int) -> Estimator:
    """Chapman is the sharper choice at two sources; Chao is the only choice above two."""
    return Estimator.CHAPMAN if source_count == 2 else Estimator.CHAO


@dataclass(frozen=True)
class CaptureTable:
    """Capture counts for one query, or for a pool of queries.

    Attributes:
        queries: How many queries these counts cover.
        observed: Documents captured by at least one source, summed per query.
        frequency: ``frequency[j - 1]`` is the number of captured documents
            that exactly ``j`` sources returned.
        captured: Documents captured, in the order of the sources the table was
            built from.
    """

    queries: int
    observed: int
    frequency: tuple[int, ...]
    captured: tuple[int, ...]

    @property
    def singletons(self) -> int:
        return self.frequency[0] if self.frequency else 0

    @property
    def doubletons(self) -> int:
        return self.frequency[1] if len(self.frequency) > 1 else 0

    def per_source(self, sources: Sequence[str]) -> dict[str, int]:
        return dict(zip(sources, self.captured, strict=True))


def summarise(
    query: QueryCaptures, sources: Sequence[str], depth: int | None = None
) -> CaptureTable:
    """Reduce one query to the counts every estimator needs."""
    marked = query.marked(depth)
    seen = Counter[str]()
    captured = []
    for source in sources:
        documents = marked.get(source, frozenset())
        captured.append(len(documents))
        seen.update(documents)
    frequency = [0] * len(sources)
    for times in seen.values():
        frequency[times - 1] += 1
    return CaptureTable(
        queries=1,
        observed=len(seen),
        frequency=tuple(frequency),
        captured=tuple(captured),
    )


def pool(tables: Sequence[CaptureTable], width: int) -> CaptureTable:
    """Add up per-query tables into one.

    Pooling assumes the pooled queries have similar capture rates, which is
    what strata are for: pool within a stratum, not across dissimilar ones.
    """
    queries = observed = 0
    frequency = [0] * width
    captured = [0] * width
    for table in tables:
        queries += table.queries
        observed += table.observed
        for index in range(width):
            frequency[index] += table.frequency[index]
            captured[index] += table.captured[index]
    return CaptureTable(
        queries=queries,
        observed=observed,
        frequency=tuple(frequency),
        captured=tuple(captured),
    )


def tabulate(
    queries: Sequence[QueryCaptures],
    sources: Sequence[str],
    depth: int | None = None,
) -> CaptureTable:
    """Summarise every query and pool the result."""
    return pool([summarise(query, sources, depth) for query in queries], len(sources))


def group_by_stratum(
    queries: Sequence[QueryCaptures],
    sources: Sequence[str],
    depth: int | None = None,
) -> dict[str, list[CaptureTable]]:
    """Summarise every query once, kept in its stratum."""
    grouped: dict[str, list[CaptureTable]] = {}
    for query in queries:
        grouped.setdefault(query.stratum, []).append(summarise(query, sources, depth))
    return grouped


def population(table: CaptureTable, estimator: Estimator) -> float:
    """Estimate how many documents the population held, captured or not.

    The result is never below the number actually observed.
    """
    if estimator is Estimator.CHAPMAN:
        if len(table.captured) != 2:
            raise ValueError(
                f"the Chapman estimator takes exactly two sources, got {len(table.captured)}"
            )
        first, second = table.captured
        estimate = _chapman(first, second, table.doubletons)
    else:
        estimate = _chao(table.observed, table.singletons, table.doubletons)
    return max(estimate, float(table.observed))


def stratum_population(
    tables: Sequence[CaptureTable], pooled: CaptureTable, estimator: Estimator
) -> float:
    """Estimate the population of one stratum, aggregated as the estimator requires.

    Args:
        tables: One summary per query in the stratum.
        pooled: Those summaries already added together, which the caller has.
        estimator: Which estimator's aggregation rule to follow.

    The two estimators aggregate differently, and swapping the rules breaks
    them. Chapman is a ratio of capture counts, so pooling a stratum's queries
    into one table is both standard and far steadier than estimating a query
    whose two sources happened to overlap on nothing. Chao's correction is
    quadratic in the number of documents seen exactly once, so pooling queries
    inflates it by roughly the number of queries pooled; it is applied to each
    query and summed instead.
    """
    if estimator is Estimator.CHAPMAN:
        return population(pooled, estimator)
    return sum(population(table, estimator) for table in tables)


def _chapman(first: int, second: int, both: int) -> float:
    """Chapman's bias correction of ``n1 * n2 / m``, which is defined at ``m = 0``."""
    return (first + 1) * (second + 1) / (both + 1) - 1


def _chao(observed: int, singletons: int, doubletons: int) -> float:
    """Chao's bias-corrected estimator of population size."""
    return observed + singletons * (singletons - 1) / (2 * (doubletons + 1))


@dataclass(frozen=True)
class StratumEstimate:
    """The estimate for one stratum of queries."""

    stratum: str
    queries: int
    observed: int
    captured: int
    population: float

    @property
    def recall(self) -> float:
        return self.captured / self.population if self.population else 0.0


@dataclass(frozen=True)
class RecallPoint:
    """A recall estimate aggregated over every stratum."""

    recall: float
    population: float
    captured: int
    observed: int
    strata: tuple[StratumEstimate, ...]


def estimate(
    grouped: Mapping[str, Sequence[CaptureTable]],
    target_index: int,
    estimator: Estimator,
    width: int,
) -> RecallPoint:
    """Estimate what fraction of the population the target source captured.

    Each stratum is estimated on its own and the strata are then summed, so a
    stratum whose queries are easy to retrieve for cannot carry a stratum whose
    queries are not.
    """
    estimates = []
    for stratum, tables in grouped.items():
        pooled = pool(tables, width)
        estimates.append(
            StratumEstimate(
                stratum=stratum,
                queries=pooled.queries,
                observed=pooled.observed,
                captured=pooled.captured[target_index],
                population=stratum_population(tables, pooled, estimator),
            )
        )

    total_population = sum(item.population for item in estimates)
    total_captured = sum(item.captured for item in estimates)
    return RecallPoint(
        recall=total_captured / total_population if total_population else 0.0,
        population=total_population,
        captured=total_captured,
        observed=sum(item.observed for item in estimates),
        strata=tuple(estimates),
    )


def bootstrap_interval(
    grouped: Mapping[str, Sequence[CaptureTable]],
    target_index: int,
    estimator: Estimator,
    width: int,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> tuple[float, float]:
    """A percentile interval for recall, resampling whole queries.

    Queries are the independent unit, not documents: two documents retrieved
    for the same query are not independent observations, and resampling
    documents would report an interval several times too narrow. Resampling
    happens within each stratum so that stratum sizes stay fixed.
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    if resamples < 1:
        raise ValueError(f"resamples must be positive, got {resamples}")

    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        resampled = {
            stratum: rng.choices(tables, k=len(tables)) for stratum, tables in grouped.items()
        }
        draws.append(estimate(resampled, target_index, estimator, width).recall)

    draws.sort()
    tail = (1 - confidence) / 2
    return (_percentile(draws, tail), _percentile(draws, 1 - tail))


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile of an already sorted sequence."""
    if not sorted_values:
        raise ValueError("no values to take a percentile of")
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
