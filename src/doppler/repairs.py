"""What to do about a low recall, measured rather than guessed.

Every repair reported here is a quantity read off the sample. Nothing is
suggested that the data cannot price, because a suggestion without a measured
gain is the thing a team already has.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .captures import QueryCaptures
from .estimators import Estimator, tabulate

EXAMPLES_PER_STRATUM = 5
DEPTH_LADDER = (1, 2, 3, 5, 10, 20, 50, 100, 200)


@dataclass(frozen=True)
class SourceRecovery:
    """How much of what the target missed another source already found."""

    source: str
    recovered: int
    share_of_missed: float


@dataclass(frozen=True)
class DepthPoint:
    """Recall the target reaches when its list is cut at this rank."""

    depth: int
    recall: float


@dataclass(frozen=True)
class MissedGroup:
    """Documents another source found and the target did not, within one stratum."""

    stratum: str
    missed: int
    examples: tuple[str, ...]


@dataclass(frozen=True)
class Repairs:
    """The priced options for closing the gap."""

    estimated_missed: float
    recoveries: tuple[SourceRecovery, ...]
    depth_sweep: tuple[DepthPoint, ...]
    missed_by_stratum: tuple[MissedGroup, ...]

    def to_dict(self) -> dict:
        return {
            "estimated_missed": self.estimated_missed,
            "recoveries": [
                {
                    "source": item.source,
                    "recovered": item.recovered,
                    "share_of_missed": item.share_of_missed,
                }
                for item in self.recoveries
            ],
            "depth_sweep": [
                {"depth": point.depth, "recall": point.recall} for point in self.depth_sweep
            ],
            "missed_by_stratum": [
                {
                    "stratum": group.stratum,
                    "missed": group.missed,
                    "examples": list(group.examples),
                }
                for group in self.missed_by_stratum
            ],
        }


def price(
    queries: Sequence[QueryCaptures],
    sources: Sequence[str],
    target: str,
    estimator: Estimator,
    population: float,
    captured: int,
    depth: int | None = None,
    max_depth: int = 0,
) -> Repairs:
    """Price each way of recovering what the target missed.

    Args:
        population: Estimated population size at the analysed depth.
        captured: Documents the target captured at the analysed depth.
        depth: Rank cutoff the estimate used, or None for the full lists.
        max_depth: Longest list in the sample.
    """
    missed = max(population - captured, 0.0)
    ceiling = max_depth if depth is None else min(depth, max_depth)
    return Repairs(
        estimated_missed=missed,
        recoveries=_recoveries(queries, sources, target, missed, depth),
        depth_sweep=_depth_sweep(queries, sources, target, population, ceiling),
        missed_by_stratum=_missed_by_stratum(queries, sources, target, depth),
    )


def _recoveries(
    queries: Sequence[QueryCaptures],
    sources: Sequence[str],
    target: str,
    missed: float,
    depth: int | None,
) -> tuple[SourceRecovery, ...]:
    counts = dict.fromkeys((name for name in sources if name != target), 0)
    for query in queries:
        marked = query.marked(depth)
        target_set = marked.get(target, frozenset())
        for name in counts:
            counts[name] += len(marked.get(name, frozenset()) - target_set)
    return tuple(
        SourceRecovery(
            source=name,
            recovered=found,
            share_of_missed=found / missed if missed else 0.0,
        )
        for name, found in sorted(counts.items(), key=lambda item: -item[1])
    )


def _depth_sweep(
    queries: Sequence[QueryCaptures],
    sources: Sequence[str],
    target: str,
    population: float,
    ceiling: int,
) -> tuple[DepthPoint, ...]:
    """Recall against a fixed population as the target's list is cut shorter.

    The denominator is the population estimated at the analysed depth, so the
    ladder stops at that same depth. Sweeping past it would divide captures
    taken at one cutoff by a population estimated at another and report a
    recall above 1. A curve that has already flattened is the evidence that
    raising the cutoff buys nothing.
    """
    if not population or ceiling <= 0:
        return ()
    ladder = [step for step in DEPTH_LADDER if step < ceiling]
    ladder.append(ceiling)
    index = list(sources).index(target)
    return tuple(
        DepthPoint(
            depth=step,
            recall=tabulate(queries, sources, step).captured[index] / population,
        )
        for step in ladder
    )


def _missed_by_stratum(
    queries: Sequence[QueryCaptures],
    sources: Sequence[str],
    target: str,
    depth: int | None,
) -> tuple[MissedGroup, ...]:
    missed: dict[str, list[str]] = {}
    counts: dict[str, int] = {}
    for query in queries:
        marked = query.marked(depth)
        target_set = marked.get(target, frozenset())
        found_elsewhere: set[str] = set()
        for name in sources:
            if name != target:
                found_elsewhere |= marked.get(name, frozenset())
        gap = sorted(found_elsewhere - target_set)
        counts[query.stratum] = counts.get(query.stratum, 0) + len(gap)
        examples = missed.setdefault(query.stratum, [])
        examples.extend(gap[: EXAMPLES_PER_STRATUM - len(examples)])
    return tuple(
        MissedGroup(stratum=stratum, missed=count, examples=tuple(missed[stratum]))
        for stratum, count in sorted(counts.items(), key=lambda item: -item[1])
    )
