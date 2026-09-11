"""Capture records and the validated sample they form.

A capture is one source's answer to one query: the documents it surfaced, in
rank order, optionally with the subset that was judged relevant. Capture
records are the only input Doppler takes, and every invariant the estimators
rely on is enforced here so that nothing downstream has to re-check it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

DEFAULT_STRATUM = "all"


class Estimand(str, Enum):
    """What the population under estimation actually is.

    The estimand is derived from the captures, never asserted by the caller,
    because the two answers mean very different things and a report that
    confuses them is worse than no report.
    """

    RELEVANT = "relevant"
    """Every capture carries relevance marks, so the population is the set of
    documents relevant to the sampled queries and the result is recall."""

    POOL = "pool"
    """No capture carries relevance marks, so the population is the retrievable
    pool -- the documents these sources sample from -- and the result is
    coverage of that pool, which is an upper bound on relevance recall."""


@dataclass(frozen=True)
class Capture:
    """One source's retrieval for one query.

    Args:
        query_id: Identifier of the query. Captures sharing it are recaptures
            of the same population.
        source: Name of the retriever that produced this list.
        doc_ids: Document identifiers in rank order, best first. Repeats are
            dropped, keeping the first occurrence, because a source that
            returns a document twice has still captured it once.
        relevant_doc_ids: The subset of ``doc_ids`` judged relevant. Supply it
            for every capture in the sample or for none of them.
        stratum: Query group. Capture rates are pooled within a stratum, so a
            stratum should hold queries that behave alike.
    """

    query_id: str
    source: str
    doc_ids: tuple[str, ...]
    relevant_doc_ids: frozenset[str] | None = None
    stratum: str = DEFAULT_STRATUM

    def __init__(
        self,
        query_id: str,
        source: str,
        doc_ids: Iterable[str],
        relevant_doc_ids: Iterable[str] | None = None,
        stratum: str = DEFAULT_STRATUM,
    ) -> None:
        ranked = tuple(dict.fromkeys(doc_ids))
        marked = None if relevant_doc_ids is None else frozenset(relevant_doc_ids)
        _require_text(query_id, "query_id")
        _require_text(source, "source")
        _require_text(stratum, "stratum")
        if marked is not None and not marked <= set(ranked):
            unknown = sorted(marked - set(ranked))[:5]
            raise ValueError(
                f"relevant_doc_ids for query {query_id!r} from source {source!r} "
                f"are not all present in doc_ids: {unknown}"
            )
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "doc_ids", ranked)
        object.__setattr__(self, "relevant_doc_ids", marked)
        object.__setattr__(self, "stratum", stratum)

    def captured(self, depth: int | None = None) -> frozenset[str]:
        """The documents this capture contributes at the given rank depth."""
        ranked = self.doc_ids if depth is None else self.doc_ids[:depth]
        if self.relevant_doc_ids is None:
            return frozenset(ranked)
        return self.relevant_doc_ids & frozenset(ranked)


@dataclass(frozen=True)
class QueryCaptures:
    """Every source's capture for one query, after validation."""

    query_id: str
    stratum: str
    by_source: dict[str, Capture]

    def marked(self, depth: int | None = None) -> dict[str, frozenset[str]]:
        return {name: capture.captured(depth) for name, capture in self.by_source.items()}


@dataclass(frozen=True)
class Sample:
    """A validated set of queries, each captured by the same sources.

    Queries that were not captured by every source cannot contribute recapture
    information, so they are excluded here and counted, rather than silently
    lowering the overlap the estimators see.
    """

    sources: tuple[str, ...]
    estimand: Estimand
    queries: tuple[QueryCaptures, ...]
    excluded_query_ids: tuple[str, ...]

    @property
    def max_depth(self) -> int:
        return max(
            (
                len(capture.doc_ids)
                for query in self.queries
                for capture in query.by_source.values()
            ),
            default=0,
        )

    def strata(self) -> tuple[str, ...]:
        seen = dict.fromkeys(query.stratum for query in self.queries)
        return tuple(seen)


def build_sample(captures: Sequence[Capture]) -> Sample:
    """Group captures into a sample, rejecting inputs the estimators cannot use.

    Raises:
        ValueError: If the captures are empty, mix marked and unmarked
            relevance, repeat a (query, source) pair, or disagree about which
            stratum a query belongs to.
    """
    if not captures:
        raise ValueError("no captures given")

    marked = sum(1 for capture in captures if capture.relevant_doc_ids is not None)
    if marked not in (0, len(captures)):
        raise ValueError(
            "relevance marks are present on some captures and absent on others "
            f"({marked} of {len(captures)} marked); supply them for all or none, "
            "because the two cases estimate different populations"
        )
    estimand = Estimand.RELEVANT if marked else Estimand.POOL

    sources = tuple(dict.fromkeys(capture.source for capture in captures))
    if len(sources) < 2:
        raise ValueError(f"capture-recapture needs at least two sources, got {list(sources)}")

    grouped: dict[str, dict[str, Capture]] = {}
    strata: dict[str, str] = {}
    for capture in captures:
        by_source = grouped.setdefault(capture.query_id, {})
        if capture.source in by_source:
            raise ValueError(
                f"duplicate capture for query {capture.query_id!r} from source "
                f"{capture.source!r}; each source answers each query once"
            )
        by_source[capture.source] = capture
        held = strata.setdefault(capture.query_id, capture.stratum)
        if held != capture.stratum:
            raise ValueError(
                f"query {capture.query_id!r} is labelled both {held!r} and "
                f"{capture.stratum!r}; a query belongs to one stratum"
            )

    complete: list[QueryCaptures] = []
    excluded: list[str] = []
    for query_id, by_source in grouped.items():
        if len(by_source) == len(sources):
            complete.append(QueryCaptures(query_id, strata[query_id], by_source))
        else:
            excluded.append(query_id)

    return Sample(
        sources=sources,
        estimand=estimand,
        queries=tuple(complete),
        excluded_query_ids=tuple(excluded),
    )


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}")
