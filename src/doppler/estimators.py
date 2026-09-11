"""Capture-recapture estimators and the resampled interval around them.

Two sources retrieving for the same query are two attempts to capture the same
unknown population. What each source caught, and how much the catches overlap,
is enough to estimate how large the population was -- including the part that
neither source ever returned.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from .captures import QueryCaptures


class Estimator(str, Enum):
    """Which population estimator to use."""

    CHAPMAN = "chapman"
    """Bias-corrected Lincoln-Petersen. Two sources only. Assumes every
    document is equally catchable; positive dependence between the sources
    makes it a lower bound on the population, and so an upper bound on
    recall."""

    CHAO = "chao"
    """Chao's bias-corrected lower bound from capture frequencies. Works for
    any number of sources and tolerates unequal catchability, at the cost of
    estimating a lower bound rather than a point."""


def default_estimator(source_count: int) -> Estimator:
    """Chapman is the sharper choice at two sources; Chao is the only choice above two."""
    return Estimator.CHAPMAN if source_count == 2 else Estimator.CHAO


@dataclass(frozen=True)
class CaptureTable:
    """Capture counts pooled over a set of queries.

    Attributes:
        queries: How many queries were pooled.
        observed: Documents captured by at least one source, summed per query.
        frequency: ``frequency[j - 1]`` is the number of captured documents
            that exactly ``j`` sources returned.
        per_source: Documents captured, by source.
    """

    queries: int
    observed: int
    frequency: tuple[int, ...]
    per_source: dict[str, int]

    @property
    def singletons(self) -> int:
        return self.frequency[0] if self.frequency else 0

    @property
    def doubletons(self) -> int:
        return self.frequency[1] if len(self.frequency) > 1 else 0


def tabulate(
    queries: Sequence[QueryCaptures],
    sources: Sequence[str],
    depth: int | None = None,
) -> CaptureTable:
    """Pool per-query capture counts into one table.

    Pooling assumes queries in the pool have similar capture rates, which is
    what strata are for: pool within a stratum, not across dissimilar ones.
    """
    observed = 0
    frequency = Counter[int]()
    per_source = dict.fromkeys(sources, 0)
    for query in queries:
        marked = query.marked(depth)
        seen = Counter[str]()
        for source in sources:
            captured = marked.get(source, frozenset())
            per_source[source] += len(captured)
            seen.update(captured)
        observed += len(seen)
        frequency.update(seen.values())
    counts = tuple(frequency.get(j, 0) for j in range(1, len(sources) + 1))
    return CaptureTable(
        queries=len(queries), observed=observed, frequency=counts, per_source=per_source
    )


def population(table: CaptureTable, estimator: Estimator) -> float:
    """Estimate how many documents the population held, captured or not.

    The result is never below the number actually observed.
    """
    if estimator is Estimator.CHAPMAN:
        if len(table.per_source) != 2:
            raise ValueError(
                f"the Chapman estimator takes exactly two sources, got {len(table.per_source)}"
            )
        first, second = table.per_source.values()
        estimate = _chapman(first, second, table.doubletons)
    else:
        estimate = _chao(table.observed, table.singletons, table.doubletons)
    return max(estimate, float(table.observed))


def _chapman(first: int, second: int, both: int) -> float:
    """Chapman's bias correction of ``n1 * n2 / m``, which is defined at ``m = 0``."""
    return (first + 1) * (second + 1) / (both + 1) - 1


def _chao(observed: int, singletons: int, doubletons: int) -> float:
    """Chao's bias-corrected lower bound on population size."""
    return observed + singletons * (singletons - 1) / (2 * (doubletons + 1))


def stratum_population(
    queries: Sequence[QueryCaptures],
    sources: Sequence[str],
    estimator: Estimator,
    depth: int | None = None,
) -> float:
    """Estimate the population of a group of queries, aggregated as the estimator requires.

    The two estimators aggregate differently, and swapping the rules breaks
    them. Chapman is a ratio of capture counts, so pooling a stratum's queries
    into one table is both standard and far steadier than estimating a query
    whose two sources happened to overlap on nothing. Chao's correction is
    quadratic in the number of documents seen exactly once, so pooling queries
    inflates it by roughly the number of queries pooled; it is applied to each
    query and summed instead.
    """
    if estimator is Estimator.CHAPMAN:
        return population(tabulate(queries, sources, depth), estimator)
    return sum(population(tabulate([query], sources, depth), estimator) for query in queries)


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
    queries: Sequence[QueryCaptures],
    sources: Sequence[str],
    target: str,
    estimator: Estimator,
    depth: int | None = None,
) -> RecallPoint:
    """Estimate what fraction of the population ``target`` captured.

    Each stratum is estimated on its own and the strata are then summed, so a
    stratum whose queries are easy to retrieve for cannot carry a stratum whose
    queries are not.
    """
    if target not in sources:
        raise ValueError(f"target {target!r} is not one of the sources {list(sources)}")

    by_stratum: dict[str, list[QueryCaptures]] = {}
    for query in queries:
        by_stratum.setdefault(query.stratum, []).append(query)

    estimates = []
    for stratum, group in by_stratum.items():
        table = tabulate(group, sources, depth)
        estimates.append(
            StratumEstimate(
                stratum=stratum,
                queries=table.queries,
                observed=table.observed,
                captured=table.per_source[target],
                population=stratum_population(group, sources, estimator, depth),
            )
        )

    total_population = sum(item.population for item in estimates)
    total_captured = sum(item.captured for item in estimates)
    total_observed = sum(item.observed for item in estimates)
    return RecallPoint(
        recall=total_captured / total_population if total_population else 0.0,
        population=total_population,
        captured=total_captured,
        observed=total_observed,
        strata=tuple(estimates),
    )


def bootstrap_interval(
    queries: Sequence[QueryCaptures],
    sources: Sequence[str],
    target: str,
    estimator: Estimator,
    depth: int | None = None,
    confidence: float = 0.95,
    resamples: int = 1000,
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

    by_stratum: dict[str, list[QueryCaptures]] = {}
    for query in queries:
        by_stratum.setdefault(query.stratum, []).append(query)

    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        resampled: list[QueryCaptures] = []
        for group in by_stratum.values():
            resampled.extend(rng.choices(group, k=len(group)))
        draws.append(estimate(resampled, sources, target, estimator, depth).recall)

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
