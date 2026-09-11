"""Plain-text rendering of a report.

The report is the product. It is read far more often than it is parsed, so it
states the estimand, the interval, and the assumptions in the same glance as
the number.
"""

from __future__ import annotations

from .captures import Estimand
from .recall import RecallEstimate, Refusal, Report

_BLOCKS = "▁▂▃▄▅▆▇█"
BAR_WIDTH = 8
STRATA_IN_MISSED_SET = 5
EXAMPLES_IN_LINE = 3
STRATUM_COLUMN = 24

_ESTIMAND_GLOSS = {
    Estimand.RELEVANT: "relevance recall, from captures carrying relevance marks",
    Estimand.POOL: "coverage of the retrievable pool, an upper bound on relevance recall",
}


def render(report: Report) -> str:
    """Render a report for a terminal."""
    if isinstance(report, Refusal):
        return _render_refusal(report)
    return _render_estimate(report)


def _render_estimate(report: RecallEstimate) -> str:
    lines = [
        f"doppler · recall of {report.target!r} over {report.diagnostics.queries} queries",
        "",
        f"  estimand    {_ESTIMAND_GLOSS[report.estimand]}",
        f"  estimator   {report.estimator.value}"
        f" · {len(report.diagnostics.per_source)} sources"
        + (f" · depth {report.depth}" if report.depth else ""),
        "",
        f"  recall      {report.recall:.2f}   "
        f"[{report.interval[0]:.2f}, {report.interval[1]:.2f}]"
        f"   {report.confidence:.0%} interval over resampled queries",
        f"  population  {report.population:,.0f} estimated · "
        f"{report.observed:,} observed · {report.captured:,} found by {report.target}",
        "",
    ]

    if len(report.strata) > 1:
        lines.append("  by stratum")
        width = min(max(len(item.stratum) for item in report.strata), STRATUM_COLUMN)
        for item in sorted(report.strata, key=lambda entry: entry.recall):
            lines.append(
                f"    {_fit(item.stratum, width):<{width}}  {item.queries:>5} queries   "
                f"recall {item.recall:.2f}   "
                f"{item.captured:,} of {item.population:,.0f}"
            )
        lines.append("")

    repairs = report.repairs
    lines.append("  missed set")
    lines.append(
        f"    {report.target} missed an estimated {repairs.estimated_missed:,.0f} documents"
    )
    for recovery in repairs.recoveries:
        if recovery.recovered:
            lines.append(
                f"    {recovery.source} already found {recovery.recovered:,} of them "
                f"({recovery.share_of_missed:.0%} of the missed set)"
            )
    for group in repairs.missed_by_stratum[:STRATA_IN_MISSED_SET]:
        if group.missed:
            examples = ", ".join(group.examples[:EXAMPLES_IN_LINE])
            lines.append(
                f"    {_fit(group.stratum, STRATUM_COLUMN)}: {group.missed:,} missed"
                + (f", such as {examples}" if examples else "")
            )
    lines.append("")

    if repairs.depth_sweep:
        lines.append("  recall against rank cutoff")
        ceiling = max(point.recall for point in repairs.depth_sweep) or 1.0
        for point in repairs.depth_sweep:
            filled = _BLOCKS[
                min(int(point.recall / ceiling * (len(_BLOCKS) - 1)), len(_BLOCKS) - 1)
            ]
            lines.append(f"    k={point.depth:<4} {filled * 8}  {point.recall:.2f}")
        lines.append("")

    lines.extend(_render_notes(report))
    return "\n".join(lines)


def _render_refusal(report: Refusal) -> str:
    lines = [
        f"doppler · no estimate for {report.target!r}",
        "",
        f"  refused     {report.reason.value}",
        f"  because     {_wrap(report.detail, indent=14)}",
        "",
        f"  queries     {report.diagnostics.queries}",
        f"  observed    {report.diagnostics.observed:,} documents · "
        f"{report.diagnostics.recaptured:,} captured more than once",
    ]
    worst = report.diagnostics.max_overlap
    if worst is not None:
        first, second = worst.sources
        lines.append(f"  overlap     {first} vs {second}: Jaccard {worst.jaccard:.2f}")
    lines.append("")
    lines.extend(_render_notes(report))
    return "\n".join(lines)


def _render_notes(report: Report) -> list[str]:
    if not report.diagnostics.notes:
        return []
    lines = ["  notes"]
    for note in report.diagnostics.notes:
        lines.append(f"    - {_wrap(note, indent=6)}")
    return lines


def _fit(text: str, width: int) -> str:
    """Cut a name to the column, keeping the end where names usually differ."""
    return text if len(text) <= width else "..." + text[-(width - 3) :]


def _wrap(text: str, indent: int, width: int = 88) -> str:
    """Wrap to ``width``, continuing lines at ``indent`` spaces."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) + indent > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return ("\n" + " " * indent).join(lines)
