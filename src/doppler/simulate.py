"""A synthetic retrieval trace with a known answer.

Capture-recapture is only as good as its assumptions, and the honest way to
find out whether it holds on your data is to simulate the conditions you think
you have and see whether the estimator recovers a recall you already know.
This module builds that trace. It is also what Doppler's own tests measure
against, because a recall estimator whose accuracy is never checked against a
known population is an opinion.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass

from .captures import Capture

DISTRACTORS_PER_QUERY = 20


@dataclass(frozen=True)
class Simulation:
    """A generated trace together with the truth it was generated from."""

    captures: tuple[Capture, ...]
    population: int
    recall: dict[str, float]
    """True recall per source, the quantity an estimate should recover."""


def simulate_captures(
    rates: Mapping[str, float],
    *,
    queries: int = 200,
    relevant_per_query: int = 20,
    heterogeneity: float = 0.0,
    strata: int = 1,
    mark_relevance: bool = True,
    seed: int = 0,
) -> Simulation:
    """Generate captures from a population of known size.

    Args:
        rates: Per-source probability of capturing an average relevant
            document.
        queries: Number of queries to generate.
        relevant_per_query: Size of each query's relevant population.
        heterogeneity: Spread of per-document catchability, on the log-odds
            scale. Zero makes every document equally catchable and every source
            independent, which is exactly the condition the estimators assume.
            Above zero, prominent documents are found by every source and
            obscure ones by none, which both biases the population downwards
            and correlates the sources.
        strata: Number of query groups, each with its own capture rates.
        mark_relevance: Whether captures carry relevance marks. When False the
            lists also carry distractors and the estimand becomes pool
            coverage.
        seed: Seed, so a simulation is reproducible.

    Raises:
        ValueError: If a rate is not a probability, or the shape arguments are
            not positive.
    """
    for source, rate in rates.items():
        if not 0 < rate < 1:
            raise ValueError(f"rate for {source!r} must lie in (0, 1), got {rate}")
    if len(rates) < 2:
        raise ValueError("simulate at least two sources")
    if queries < 1 or relevant_per_query < 1 or strata < 1:
        raise ValueError("queries, relevant_per_query and strata must all be positive")
    if heterogeneity < 0:
        raise ValueError(f"heterogeneity must not be negative, got {heterogeneity}")

    rng = random.Random(seed)
    captures: list[Capture] = []
    found: dict[str, set[str]] = {source: set() for source in rates}
    population = 0

    for index in range(queries):
        query_id = f"q{index:04d}"
        stratum = f"s{index % strata}"
        # A stratum shifts every source's rate together, which is the pattern
        # strata exist to absorb: some query types are simply harder.
        stratum_shift = (index % strata) * -0.4
        relevant = [f"{query_id}-d{doc:03d}" for doc in range(relevant_per_query)]
        distractors = (
            []
            if mark_relevance
            else [f"{query_id}-x{noise:03d}" for noise in range(DISTRACTORS_PER_QUERY)]
        )
        # Unmarked captures make every returned document part of the captured
        # population, so the distractors join the truth being recovered.
        prominence = {doc: rng.gauss(0, 1) for doc in relevant + distractors}
        population += len(relevant) + len(distractors)

        for source, rate in rates.items():
            base = _logit(rate) + stratum_shift
            scored = [
                (doc, rng.random())
                for doc in relevant + distractors
                if rng.random() < _expit(base + heterogeneity * prominence[doc])
            ]
            found[source].update(doc for doc, _ in scored)
            scored.sort(key=lambda item: -item[1])
            ranked = [doc for doc, _ in scored]
            captures.append(
                Capture(
                    query_id=query_id,
                    source=source,
                    doc_ids=ranked,
                    relevant_doc_ids=set(ranked) & set(relevant) if mark_relevance else None,
                    stratum=stratum,
                )
            )

    return Simulation(
        captures=tuple(captures),
        population=population,
        recall={source: len(docs) / population for source, docs in found.items()},
    )


def _logit(probability: float) -> float:
    return math.log(probability / (1 - probability))


def _expit(value: float) -> float:
    return 1 / (1 + math.exp(-value))
