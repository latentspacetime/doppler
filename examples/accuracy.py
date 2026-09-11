"""Measure the estimator against populations whose recall is already known.

Every number in the README's accuracy table comes from this script. Run it to
reproduce them, or change the arguments to the conditions you believe your own
retrieval stack is under and see whether the estimate still lands.

    python examples/accuracy.py
"""

from doppler import Refusal, estimate_recall, simulate_captures

RATES = {"dense": 0.55, "bm25": 0.40}
THIRD_SOURCE = {**RATES, "splade": 0.45}
HETEROGENEITY = (0.0, 0.3, 0.6, 1.0, 1.6)
REPEATS = 5


def row(rates: dict[str, float], heterogeneity: float) -> tuple[float, float, float] | str:
    """Average truth, estimate, and interval coverage over independent traces.

    Returns the refusal reason instead when a trace does not support an
    estimate, which is what a sample too small or too correlated should do.
    """
    truths, estimates, covered = [], [], 0
    for seed in range(REPEATS):
        simulation = simulate_captures(
            rates,
            queries=300,
            relevant_per_query=12,
            heterogeneity=heterogeneity,
            strata=3,
            seed=seed,
        )
        report = estimate_recall(simulation.captures, "dense", resamples=300, seed=seed)
        if isinstance(report, Refusal):
            return report.reason.value
        truth = simulation.recall["dense"]
        truths.append(truth)
        estimates.append(report.recall)
        covered += report.interval[0] <= truth <= report.interval[1]
    return (
        sum(truths) / REPEATS,
        sum(estimates) / REPEATS,
        covered / REPEATS,
    )


def main() -> None:
    for label, rates in (("2 sources", RATES), ("3 sources", THIRD_SOURCE)):
        print(f"\n{label}")
        print(
            f"  {'heterogeneity':>14}  {'true':>6}  {'estimated':>9}  {'error':>7}  {'covered':>7}"
        )
        for heterogeneity in HETEROGENEITY:
            measured = row(rates, heterogeneity)
            if isinstance(measured, str):
                print(f"  {heterogeneity:>14.1f}  refused: {measured}")
                continue
            truth, estimated, coverage = measured
            print(
                f"  {heterogeneity:>14.1f}  {truth:>6.2f}  {estimated:>9.2f}  "
                f"{estimated - truth:>+7.2f}  {coverage:>6.0%}"
            )


if __name__ == "__main__":
    main()
