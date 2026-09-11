"""The one call Doppler exists for, and the two answers it can give."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .captures import Capture, Estimand, build_sample
from .diagnostics import Diagnostics, RefusalReason, Thresholds, measure, refuse
from .estimators import (
    Estimator,
    StratumEstimate,
    bootstrap_interval,
    default_estimator,
    estimate,
    tabulate,
)
from .repairs import Repairs, price

DEFAULT_CONFIDENCE = 0.95
DEFAULT_RESAMPLES = 1000


@dataclass(frozen=True)
class RecallEstimate:
    """What fraction of the population the target source found.

    Attributes:
        recall: The estimate. Read it together with ``interval``; the point
            alone is not the finding.
        interval: Percentile bootstrap interval at ``confidence``, resampling
            queries.
        population: Estimated size of the population, including the part no
            source returned.
        captured: Documents the target actually returned.
        observed: Documents any source returned.
        estimand: Whether this is recall over relevant documents or coverage of
            the retrievable pool. They are not the same claim.
    """

    verdict: Literal["estimated"]
    target: str
    estimand: Estimand
    estimator: Estimator
    depth: int | None
    recall: float
    interval: tuple[float, float]
    confidence: float
    population: float
    captured: int
    observed: int
    strata: tuple[StratumEstimate, ...]
    diagnostics: Diagnostics
    repairs: Repairs

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "target": self.target,
            "estimand": self.estimand.value,
            "estimator": self.estimator.value,
            "depth": self.depth,
            "recall": self.recall,
            "interval": list(self.interval),
            "confidence": self.confidence,
            "population": self.population,
            "captured": self.captured,
            "observed": self.observed,
            "strata": [
                {
                    "stratum": item.stratum,
                    "queries": item.queries,
                    "observed": item.observed,
                    "captured": item.captured,
                    "population": item.population,
                    "recall": item.recall,
                }
                for item in self.strata
            ],
            "diagnostics": self.diagnostics.to_dict(),
            "repairs": self.repairs.to_dict(),
        }

    def __str__(self) -> str:
        from .render import render

        return render(self)


@dataclass(frozen=True)
class Refusal:
    """No estimate, and the reason why.

    A refusal is the honest answer when the sample cannot support a number. It
    carries the same diagnostics an estimate would, so the caller can see what
    to change.
    """

    verdict: Literal["refused"]
    reason: RefusalReason
    detail: str
    target: str
    estimand: Estimand
    diagnostics: Diagnostics

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "reason": self.reason.value,
            "detail": self.detail,
            "target": self.target,
            "estimand": self.estimand.value,
            "diagnostics": self.diagnostics.to_dict(),
        }

    def __str__(self) -> str:
        from .render import render

        return render(self)


Report = RecallEstimate | Refusal


def estimate_recall(
    captures: Sequence[Capture],
    target: str,
    *,
    depth: int | None = None,
    estimator: Estimator | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    thresholds: Thresholds | None = None,
) -> Report:
    """Estimate how much of the population ``target`` retrieves.

    Args:
        captures: Every source's retrieval for every sampled query.
        target: The source being measured. The others are its recaptures.
        depth: Rank cutoff to apply to every list, or None to use them whole.
        estimator: Population estimator. Defaults to Chapman at two sources and
            Chao above two.
        confidence: Coverage of the reported interval.
        resamples: Bootstrap resamples. Lower it for speed, not for a tighter
            interval; it does not affect the point estimate.
        seed: Bootstrap seed, so a report is reproducible.
        thresholds: The bar the sample must clear. See ``Thresholds``.

    Returns:
        A ``RecallEstimate`` when the sample supports one, otherwise a
        ``Refusal`` naming what is wrong with the sample.

    Raises:
        ValueError: If the captures are malformed. A malformed input is a
            caller error; an uninformative input is a refusal.
    """
    sample = build_sample(captures)
    if target not in sample.sources:
        raise ValueError(f"target {target!r} is not one of the sources {list(sample.sources)}")
    if depth is not None and depth < 1:
        raise ValueError(f"depth must be at least 1, got {depth}")

    chosen = estimator or default_estimator(len(sample.sources))
    table = tabulate(sample.queries, sample.sources, depth)
    diagnostics = measure(
        sample.queries, sample.sources, table, len(sample.excluded_query_ids), depth
    )

    blocked = refuse(diagnostics, thresholds or Thresholds())
    if blocked is not None:
        reason, detail = blocked
        return Refusal(
            verdict="refused",
            reason=reason,
            detail=detail,
            target=target,
            estimand=sample.estimand,
            diagnostics=diagnostics,
        )

    point = estimate(sample.queries, sample.sources, target, chosen, depth)
    interval = bootstrap_interval(
        sample.queries,
        sample.sources,
        target,
        chosen,
        depth=depth,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )
    return RecallEstimate(
        verdict="estimated",
        target=target,
        estimand=sample.estimand,
        estimator=chosen,
        depth=depth,
        recall=point.recall,
        interval=interval,
        confidence=confidence,
        population=point.population,
        captured=point.captured,
        observed=point.observed,
        strata=point.strata,
        diagnostics=diagnostics,
        repairs=price(
            sample.queries,
            sample.sources,
            target,
            chosen,
            population=point.population,
            captured=point.captured,
            depth=depth,
            max_depth=sample.max_depth,
        ),
    )
