# Doppler

2026-09-10 20:19 PST

## INTENT

Doppler is a Python library that estimates how much of the document set a search system returned. You run two search systems, called retrievers, on the same queries and count how many documents they both found. From that overlap the library estimates the size of the full set, then reports the fraction returned by the retriever you named, along with a range showing how far off that figure could be. This document covers install, a demo on a simulated log, the contents of the report, how to feed your own logs from Python or from a JSON Lines file, the two quantities the library can measure, which are relevance recall, meaning the share of documents that should have come back, and pool coverage, meaning the share of everything these retrievers together could return, measured accuracy when some documents are easier to find than others, the four cases where the library returns a refusal naming the problem with the sample, how to pick the second retriever, how to group queries into batches of similar queries called strata, the estimator arguments, related methods, and how to run the tests. License is MIT.

## Install

```
pip install git+https://github.com/latentspacetime/doppler
doppler-recall --demo --target dense
```

The demo runs on a simulated log whose true recall is known, so the last line of the report shows the difference between the estimate and the true recall.

## How the estimate is made

Recall is the fraction of the documents that should have come back and did come back. Those missing documents are absent from the result list, so their count has to be estimated.

Run two retrievers over the same sample of queries, where each retriever returns a set of documents for each query. Doppler counts the documents each retriever returned and the documents both retrievers returned, then estimates how many documents exist that were missing from both lists. Recall for one retriever is how much of that estimated set it returned.

This is a capture-recapture estimate, and the default estimator is Chapman at two sources and Chao once there are three or more.

## Report

```
doppler · recall of 'dense' over 300 queries

  estimand    relevance recall, from captures carrying relevance marks
  estimator   chapman · 2 sources

  recall      0.52   [0.50, 0.55]   95% interval over resampled queries
  population  3,168 estimated · 2,177 observed · 1,661 found by dense

  by stratum
    s2    100 queries   recall 0.41   428 of 1,047
    s1    100 queries   recall 0.54   560 of 1,028
    s0    100 queries   recall 0.62   673 of 1,093

  missed set
    dense missed an estimated 1,507 documents
    bm25 already found 516 of them (34% of the missed set)
    s0: 192 missed, such as q0000-d010, q0003-d003, q0003-d004
    s2: 163 missed, such as q0005-d001, q0005-d002, q0005-d009
    s1: 161 missed, such as q0007-d001, q0007-d002, q0007-d007

  recall against rank cutoff
    k=1    ▁▁▁▁▁▁▁▁  0.07
    k=2    ▂▂▂▂▂▂▂▂  0.13
    k=3    ▃▃▃▃▃▃▃▃  0.19
    k=5    ▄▄▄▄▄▄▄▄  0.29
    k=10   ▇▇▇▇▇▇▇▇  0.46
    k=20   ████████  0.52
    k=23   ████████  0.52

  notes
    - The interval covers sampling variation in the query sample. It does not cover
      estimator bias from unequal catchability, which is the larger error wherever it is
      present; the accuracy table in the README gives its measured size and direction.
    - Dependence between two sources cannot be identified from two sources. If they tend
      to find the same documents for reasons other than relevance, the population is
      underestimated and this recall is an upper bound. Add a third, differently built
      source to identify it.

  true recall of 'dense' in this simulated trace: 0.46
```

That output is `doppler-recall --demo --target dense`.

Strata, the missed set, and the rank cutoff curve are the three parts of the report that point to the next change to make. Strata locate the query groups with the lowest recall. The missed set shows how much of the gap a retriever you already run would close, which is a third of it in the example above. The rank cutoff curve shows whether returning more results per query would raise recall further.

## Usage

Every source answers every query, and each answer is one record:

```python
from doppler import Capture, estimate_recall

captures = [
    Capture(query_id="q1", source="dense", doc_ids=["d12", "d40", "d7"], stratum="how-to"),
    Capture(query_id="q1", source="bm25",  doc_ids=["d12", "d88"],       stratum="how-to"),
    # ... the same two sources for every query in the sample
]

report = estimate_recall(captures, target="dense")
print(report)
print(report.recall, report.interval)
```

From the command line, the same captures as JSON Lines:

```jsonl
{"query_id": "q1", "source": "dense", "doc_ids": ["d12", "d40", "d7"], "stratum": "how-to"}
{"query_id": "q1", "source": "bm25",  "doc_ids": ["d12", "d88"],       "stratum": "how-to"}
```

```
doppler-recall captures.jsonl --target dense
doppler-recall captures.jsonl --target dense --json > report.json
```

Exit code is 0 for an estimate, 2 for a refusal, and 1 for bad input.

## Relevance marks

When captures carry `relevant_doc_ids`, the population is the set of relevant documents and the reported number is relevance recall. Those marks come from judging the documents that came back, a few thousand judgements from a cross-encoder, an LLM judge, or click data. When captures carry only returned document lists, the population is every document these retrievers would return and the reported number is coverage of that pool. Pool coverage is an upper bound on relevance recall, since documents outside the pool can be relevant. The report prints which of the two it measured on the estimand line and beside each recall figure.

## Accuracy

Capture-recapture assumes every document is equally likely to be retrieved. Real document collections break that assumption, because a popular document that shares many words with the query gets found by every retriever, while a document worded in unusual terms gets found by few of them. When that happens the overlap looks larger than the underlying population implies, so the population is underestimated and recall comes out too high.

`examples/accuracy.py` measures the size of that effect by running the estimator against simulated populations whose recall is already known, at increasing levels of unequal catchability, and printing this:

| Unequal catchability | True recall | Estimated | Error | 95% interval covered truth |
| --- | --- | --- | --- | --- |
| none | 0.45 | 0.45 | +0.01 | 100% |
| 0.3 | 0.45 | 0.47 | +0.02 | 80% |
| 0.6 | 0.46 | 0.52 | +0.07 | 0% |
| 1.0 | 0.46 | 0.57 | +0.12 | 0% |
| 1.6 | 0.46 | 0.67 | +0.20 | 0% |

With a third retriever in the sample, the estimator switches to Chao's method, which tolerates unequal catchability:

| Unequal catchability | True recall | Estimated | Error | 95% interval covered truth |
| --- | --- | --- | --- | --- |
| none | 0.45 | 0.40 | -0.04 | 0% |
| 0.3 | 0.45 | 0.42 | -0.03 | 40% |
| 0.6 | 0.46 | 0.45 | -0.01 | 80% |
| 1.0 | 0.46 | 0.49 | +0.02 | 80% |
| 1.6 | 0.47 | 0.56 | +0.09 | 0% |

Those tables give the error to expect at each level of unequal catchability. With two retrievers the estimate stays within about 0.02 of truth while catchability is even, and once catchability reaches 0.6 the estimate runs high by 0.07 to 0.20, so treat it as an upper bound on true recall. That two-retriever error always runs high, so a low recall estimate from two retrievers is at least as bad as the true recall. Three retrievers built on different principles keep the error within about 0.04 from even catchability up to 1.0, where the estimate runs about 0.05 low at even catchability and about 0.09 high at catchability 1.6. Match your own catchability level to a row in the table to read the expected error.

## Refusals

When two retrievers return nearly the same documents, the estimator treats their agreement as evidence that few documents were missed, so it reports a narrow interval around a recall figure that is too high. These four conditions return a refusal:

| Refusal | Condition | What to change |
| --- | --- | --- |
| `too_few_queries` | fewer than 30 queries | sample more queries |
| `insufficient_recapture` | fewer than 10 documents found by more than one source | retrieve deeper, or sample more queries |
| `sources_too_similar` | pooled Jaccard overlap at or above 0.8 | compare retrievers built on different principles |
| `no_captures` | every list was present and empty | check the logs being fed in |

```
doppler · no estimate for 'dense'

  refused     sources_too_similar
  because     dense and bm25 agree on 82% of what they return (Jaccard 0.82, at or above
              the limit of 0.80). Near-identical sources make the missed set look empty.
              Compare sources built on different principles, such as a dense retriever
              against a lexical one.

  queries     60
  observed    660 documents · 540 captured more than once
  overlap     dense vs bm25: Jaccard 0.82

  notes
    - The interval covers sampling variation in the query sample. It does not cover
      estimator bias from unequal catchability, which is the larger error wherever it is
      present; the accuracy table in the README gives its measured size and direction.
    - Dependence between two sources cannot be identified from two sources. If they tend
      to find the same documents for reasons other than relevance, the population is
      underestimated and this recall is an upper bound. Add a third, differently built
      source to identify it.
```

The thresholds are arguments, so you can lower them and read the result knowing which check you loosened:

```python
from doppler import Thresholds
estimate_recall(captures, "dense", thresholds=Thresholds(min_queries=15))
```

## Second source

This method needs two retrievers that miss different documents, and a dense retriever against BM25 is the pairing to start with. Two lists from the same retriever at different cutoffs overlap by construction, so the estimator reads that overlap as full coverage and the missed set comes out near zero.

Anything that produces a ranked list of document identifiers works, including a retriever you are evaluating for purchase, an old index kept alongside the new one, or a slow high quality reranker run over a wide candidate set.

## Strata

A stratum is a group of queries you expect to behave alike. Capture rates are computed within each stratum, so a group of easy queries and a group of hard queries each get their own capture rate and their own recall figure, and the per-group figures show which group to improve first. Any grouping you already have works, including intent label, product area, language, query length bucket, or an embedding cluster you computed yourself.

## Arguments

| Argument | Meaning |
| --- | --- |
| `target` | the source being measured; the others are its recaptures |
| `depth` | rank cutoff applied to every list before estimating |
| `estimator` | `chapman` or `chao`; defaults to Chapman at two sources and Chao above two |
| `confidence` | coverage of the reported interval, default 0.95 |
| `resamples` | bootstrap resamples, default 1000 |
| `seed` | makes a report reproducible |
| `thresholds` | query count, recapture count, and overlap limit a sample must meet before an estimate is produced |

The interval comes from resampling whole queries, because two documents retrieved for the same query are not independent observations and resampling documents would report an interval several times too narrow.

## Related work

Capture-recapture has been applied to retrieval before. Bharat and Broder used it in 1998 to estimate the size of a search engine index. TREC pooling estimators handle metrics computed from incomplete judgements. RAGAS asks a language model whether retrieved context covers a reference answer. CONFLARE sets a conformal distance cutoff on a synthetic answerable set.

Doppler takes query logs and the result lists from two or more retrievers, and returns a recall figure with an interval for a production index, a refusal state when the sample cannot support an estimate, and the measured error at each level of unequal catchability.

## Development

```
uv venv && uv pip install -e ".[dev]"
.venv/bin/python -m pytest
ruff check . && ruff format --check .
```

## License

MIT
