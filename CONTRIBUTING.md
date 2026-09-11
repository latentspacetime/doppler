# Contributing

Doppler estimates retrieval recall from two or more retrievers using capture-recapture. The library holds the statistics; the caller holds the retrievers.

## Rules

- No runtime dependencies. The library uses only the Python standard library, and a change that adds a dependency needs a reason in the pull request.
- Every claim about accuracy is measured. `examples/accuracy.py` runs the estimator against populations of known size, and any README number about error or coverage comes from a run of that script.
- An uninformative sample gets a `Refusal`, not a number with a caveat. A malformed input raises `ValueError`. Those two cases stay separate.
- The estimand is derived from the captures and never asserted by the caller. Relevance recall and pool coverage are different claims, and the report says which one it made.
- Statistics live in `estimators.py`, sample health in `diagnostics.py`, orchestration in `recall.py`, and text output in `render.py`. Keep the pure functions pure.

## Checks

```
uv venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest
ruff check . && ruff format --check .
```

All three run in CI on Python 3.10 through 3.13 and must pass before merge.
