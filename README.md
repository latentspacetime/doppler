# Doppler

Estimate how many relevant documents your retriever never returned, using a second retriever and no relevance labels.

```
pip install git+https://github.com/latentspacetime/doppler
doppler-recall --demo --target dense
```

## Intent

This repository holds one Python library that answers a single question about a retrieval system: of all the documents that should have come back for your queries, what fraction did your retriever actually return? Normally you can only answer that by paying people to label a corpus. Doppler answers it by running two retrievers over the same queries and reading their overlap, which is the same method biologists use to count fish in a lake without draining it. The library takes retrieval logs, returns a recall number with a confidence interval, lists the documents one retriever found and the other missed, prices each available repair, and refuses to return a number at all when the sample cannot support one. It has no dependencies outside the Python standard library.

## The problem

A retrieval system returns a bad answer and the team cannot tell which half of the system failed. The generator may have had the right documents and written the wrong text, or the index may never have returned the documents that would have supported a good answer. Precision is easy to look at because you can read what came back. Recall is the hard one, because measuring it means knowing about the documents that did not come back, and there is no list of those. Teams usually guess, and a guess sends people to rewrite prompts for a week when the real miss was in the index.

## How it works

Run two retrievers over the same sample of queries. For any one query, each retriever returns a set of documents, and those two sets overlap by some amount. If both retrievers return almost the same documents, the relevant set is probably small and both of them are finding most of it. If they return mostly different documents, then each one is sampling from a larger pool, and a lot of that pool was returned by neither.

That relationship is exact enough to estimate, and the estimate is old. Ecologists catch 100 fish, mark them, release them, and catch 100 more; if 50 of the second catch carry marks, the lake holds about 200 fish. Doppler treats a retriever as the net and a document as the fish. It counts how many documents each retriever returned, how many both returned, and estimates how many exist that neither one did.

The number that comes back is the estimated size of the population for your sampled queries. Recall for one retriever is how much of that population it returned.

## What you get

```
doppler · recall of 'dense' over 300 queries

  estimand    relevance recall, from captures carrying relevance marks
  estimator   chapman · 2 sources

  recall      0.50   [0.47, 0.54]   95% interval over resampled queries
  population  3,216 estimated · 2,170 observed · 1,621 found by dense

  by stratum
    s2    100 queries   recall 0.42   442 of 1,054
    s1    100 queries   recall 0.49   518 of 1,055
    s0    100 queries   recall 0.60   661 of 1,107

  missed set
    dense missed an estimated 1,595 documents
    bm25 already found 549 of them (34% of the missed set)
    s0: 196 missed, such as q0000-d003, q0000-d005, q0000-d008

  recall against rank cutoff
    k=1    ▁▁▁▁▁▁▁▁  0.09
    k=2    ▂▂▂▂▂▂▂▂  0.18
    k=3    ▄▄▄▄▄▄▄▄  0.27
    k=5    ▆▆▆▆▆▆▆▆  0.41
    k=10   ████████  0.50

  notes
    - Dependence between two sources cannot be identified from two sources. If they tend
      to find the same documents for reasons other than relevance, the population is
      underestimated and this recall is an upper bound. Add a third, differently built
      source to identify it.

  true recall of 'dense' in this simulated trace: 0.45
```

That output is `doppler-recall --demo --target dense`, which runs on a simulated trace whose true recall is known, so the last line shows how close the estimate landed.

Three things in that report are worth acting on. The stratum table says which kinds of query are being served badly, so the work has an address. The missed set says how much of the gap a retriever you already have would close, which in the example is a third of it. The rank cutoff curve says whether returning more results per query would help, and a curve that has flattened says it would not.

## Using it on your own logs

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

## Two different questions, and which one you asked

Retrieved is not the same as relevant, and Doppler will not blur the two.

If your captures carry `relevant_doc_ids`, the population is the set of relevant documents and the answer is recall. You get relevance marks by judging only the documents that came back, which is a few thousand judgements from a cross-encoder, an LLM judge, or click data. You never label the corpus.

If your captures carry no marks, the population is every document these retrievers would return, and the answer is coverage of that pool. Pool coverage is an upper bound on relevance recall, because a retriever that returns everything in the pool still may not have returned everything relevant. The report says which of the two it measured on every line that could be misread.

## Where the estimate is trustworthy

Capture-recapture assumes every document is equally likely to be retrieved. Real corpora break that assumption, because a popular document with strong lexical overlap gets found by every retriever and an obscurely worded one gets found by none. When that happens the overlap looks larger than the underlying population implies, so the population is underestimated and recall comes out too high.

The size of that effect is measured rather than described. `examples/accuracy.py` runs the estimator against simulated populations whose recall is already known, at increasing levels of unequal catchability, and prints this:

| Unequal catchability | True recall | Estimated | Error | 95% interval covered truth |
| --- | --- | --- | --- | --- |
| none | 0.45 | 0.45 | 0.00 | 100% |
| 0.3 | 0.46 | 0.47 | +0.01 | 80% |
| 0.6 | 0.45 | 0.50 | +0.05 | 20% |
| 1.0 | 0.46 | 0.58 | +0.12 | 0% |
| 1.6 | 0.47 | 0.67 | +0.21 | 0% |

With a third retriever in the sample, the estimator switches to Chao's method, which tolerates unequal catchability:

| Unequal catchability | True recall | Estimated | Error | 95% interval covered truth |
| --- | --- | --- | --- | --- |
| none | 0.45 | 0.41 | -0.05 | 0% |
| 0.3 | 0.45 | 0.42 | -0.03 | 20% |
| 0.6 | 0.46 | 0.45 | -0.01 | 80% |
| 1.0 | 0.46 | 0.49 | +0.03 | 20% |
| 1.6 | 0.46 | 0.56 | +0.10 | 0% |

Read those tables as the operating range. Two retrievers give a good number when catchability is fairly even and an upper bound when it is not. Three retrievers built on different principles hold the number much closer across the range, which is the reason the report asks for a third source. In every case the direction of the error is known, so a low estimate is always real news.

## When Doppler refuses

A capture-recapture estimate fails quietly. Two retrievers that return nearly the same documents produce a tight, confident, wrong answer, because the estimator reads their agreement as evidence that little was missed. Four conditions block an estimate instead of producing one:

| Refusal | Condition | What to change |
| --- | --- | --- |
| `too_few_queries` | fewer than 30 queries | sample more queries |
| `insufficient_recapture` | fewer than 10 documents found by more than one source | retrieve deeper, or sample more queries |
| `sources_too_similar` | pooled Jaccard overlap at or above 0.8 | compare retrievers built on different principles |
| `no_captures` | nothing was returned at all | check the logs being fed in |

```
doppler · no estimate for 'dense'

  refused     sources_too_similar
  because     dense and bm25 agree on 82% of what they return (Jaccard 0.82, at or above
              the limit of 0.80). Near-identical sources make the missed set look empty.
              Compare sources built on different principles, such as a dense retriever
              against a lexical one.
```

The thresholds are arguments, so you can lower them deliberately and read the result knowing what you loosened:

```python
from doppler import Thresholds
estimate_recall(captures, "dense", thresholds=Thresholds(min_queries=15))
```

## Choosing the second source

The method wants two retrievers that miss different documents. A dense retriever against BM25 is the standard pairing and the one to start with. A dense retriever against the same dense retriever with a different `k` is the worst pairing, because both find the same documents by construction and the overlap carries no information about what neither found.

Anything that produces a ranked list of document identifiers works, including a retriever you are evaluating for purchase, an old index kept alongside the new one, or a slow high quality reranker run over a wide candidate set.

## Strata

Capture rates are pooled inside a stratum, so a stratum should hold queries that behave alike. Passing a `stratum` on each capture keeps an easy group of queries from carrying a hard one, and the report then shows recall per group, which is where the work usually gets assigned. Any grouping you already have works: intent label, product area, language, query length bucket, or an embedding cluster you computed yourself.

## Reference

| Argument | Meaning |
| --- | --- |
| `target` | the source being measured; the others are its recaptures |
| `depth` | rank cutoff applied to every list before estimating |
| `estimator` | `chapman` or `chao`; defaults to Chapman at two sources and Chao above two |
| `confidence` | coverage of the reported interval, default 0.95 |
| `resamples` | bootstrap resamples, default 1000 |
| `seed` | makes a report reproducible |
| `thresholds` | the bar the sample must clear before an estimate is produced |

The interval comes from resampling whole queries, because two documents retrieved for the same query are not independent observations and resampling documents would report an interval several times too narrow.

## Prior art

Capture-recapture has been applied to retrieval before. Bharat and Broder used it in 1998 to estimate the size of a search engine index. TREC pooling estimators handle metrics computed from incomplete judgements. RAGAS asks a language model whether retrieved context covers a reference answer, which needs reference answers. CONFLARE sets a conformal distance cutoff on a synthetic answerable set.

Doppler takes query logs and two or more retrievers and returns a recall interval for a production index, with the refusal states and the measured operating range that make the number usable.

## Development

```
uv venv && uv pip install -e ".[dev]" pytest
.venv/bin/python -m pytest
ruff check . && ruff format --check .
```

## License

MIT
