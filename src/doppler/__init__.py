"""Doppler: how much of the relevant set your retriever never returns.

Two retrievers running over the same queries are two attempts to capture the
same unknown population of relevant documents. What they both found, what only
one found, and how much they overlap is enough to estimate how much neither of
them found -- without a single relevance label over the corpus.

    from doppler import Capture, estimate_recall

    report = estimate_recall(captures, target="dense")
    print(report)

The estimate comes with an interval, the assumptions it rests on, and a refusal
when the sample cannot support it.
"""

from .captures import Capture, Estimand, QueryCaptures, Sample, build_sample
from .diagnostics import Diagnostics, PairOverlap, RefusalReason, Thresholds
from .estimators import Estimator, StratumEstimate
from .recall import RecallEstimate, Refusal, Report, estimate_recall
from .render import render
from .repairs import DepthPoint, MissedGroup, Repairs, SourceRecovery
from .simulate import Simulation, simulate_captures

__version__ = "0.1.0"

__all__ = [
    "Capture",
    "DepthPoint",
    "Diagnostics",
    "Estimand",
    "Estimator",
    "MissedGroup",
    "PairOverlap",
    "QueryCaptures",
    "RecallEstimate",
    "Refusal",
    "RefusalReason",
    "Repairs",
    "Report",
    "Sample",
    "Simulation",
    "SourceRecovery",
    "StratumEstimate",
    "Thresholds",
    "__version__",
    "build_sample",
    "estimate_recall",
    "render",
    "simulate_captures",
]
