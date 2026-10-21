from collections import Counter
from statistics import fmean

from app.evaluation.gate import gate_sweep
from app.evaluation.judge import VERDICT_SCORES
from app.evaluation.stats import bootstrap_ci, paired_difference_ci, percentile


def ci(values) -> dict | None:
    """Mean with a 95% bootstrap interval, or None when there is nothing to average."""
    values = list(values)
    if not values:
        return None
    mean, low, high = bootstrap_ci(values)
    return {"mean": mean, "low": low, "high": high, "n": len(values)}


def _in_split(question: dict, split: str | None) -> bool:
    return split is None or question["split"] == split


# ---------- retrieval ----------

def _delta(questions: list[dict], base_by_id: dict[str, dict], metric: str) -> dict | None:
    pairs = [
        (q["metrics"][metric], base_by_id[q["id"]]["metrics"][metric])
        for q in questions
        if q["id"] in base_by_id
    ]
    if not pairs:
        return None
    mean, low, high = paired_difference_ci([a for a, _ in pairs], [b for _, b in pairs])
    return {"mean": mean, "low": low, "high": high, "significant": low > 0 or high < 0}


def retrieval_summary(results: dict, split: str | None) -> list[dict]:
    k = results["meta"]["k"]
    experiments = results["experiments"]
    metrics = [f"recall@{k}", f"precision@{k}", "mrr", f"ndcg@{k}"]

    def answerable(exp: dict) -> list[dict]:
        return [q for q in exp["questions"] if q["answerable"] and _in_split(q, split)]

    base_by_id = {q["id"]: q for q in answerable(experiments[0])}
    rows = []
    for index, exp in enumerate(experiments):
        qs = answerable(exp)
        latencies = [q["latency_ms"] for q in exp["questions"] if _in_split(q, split)]
        rows.append(
            {
                "name": exp["name"],
                "n": len(qs),
                "metrics": {m: ci(q["metrics"][m] for q in qs) for m in metrics},
                "delta": None
                if index == 0
                else {
                    "recall": _delta(qs, base_by_id, f"recall@{k}"),
                    "mrr": _delta(qs, base_by_id, "mrr"),
                },
                "latency": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)}
                if latencies
                else None,
            }
        )
    return rows


def gate_summary(results: dict, split: str | None, current: float) -> dict | None:
    selected = [
        q for q in results["experiments"][0]["questions"]
        if _in_split(q, split) and q.get("best_dense") is not None
    ]
    answerable = [q["best_dense"] for q in selected if q["answerable"]]
    negatives = [q["best_dense"] for q in selected if not q["answerable"]]
    if not answerable or not negatives:
        return None
    return {
        "answerable": len(answerable),
        "negatives": len(negatives),
        "current": current,
        "rows": [
            {"threshold": r.threshold, "false_refuse": r.false_refuse, "false_accept": r.false_accept}
            for r in gate_sweep(answerable, negatives)
        ],
    }


# ---------- answers ----------

def _cost(meta: dict, tokens: dict, judge_enabled: bool) -> dict | None:
    pricing = meta.get("pricing") or {}
    generation, judge = pricing.get("generation"), pricing.get("judge")
    if not generation:
        return None
    total = (
        tokens["generation_prompt"] / 1e6 * generation["input_per_million"]
        + tokens["generation_completion"] / 1e6 * generation["output_per_million"]
    )
    if judge_enabled:
        if not judge:
            return None  # can't price the judge, so don't report a partial total
        total += (
            tokens["judge_prompt"] / 1e6 * judge["input_per_million"]
            + tokens["judge_completion"] / 1e6 * judge["output_per_million"]
        )
    n = tokens["questions"]
    return {
        "currency": generation["currency"],
        "total": total,
        "per_question": total / n if n else None,
    }


def answer_summary(results: dict, split: str | None) -> dict:
    meta = results["meta"]
    selected = [q for q in results["questions"] if _in_split(q, split)]
    ok = [q for q in selected if not q.get("error")]
    answerable = [q for q in ok if q["answerable"]]
    unanswerable = [q for q in ok if not q["answerable"]]
    judge_enabled = bool((meta.get("judge") or {}).get("enabled"))

    def share(items: list[dict], status: str) -> dict | None:
        return ci(1.0 if q["status"] == status else 0.0 for q in items)

    verdicts = Counter(q["correctness"]["verdict"] for q in answerable if q.get("correctness"))
    judged = sum(verdicts.values())

    end_to_end: list[float] = []
    if judge_enabled:
        for q in answerable:
            if not q.get("has_reference"):
                continue
            if q["status"] != "answered":
                end_to_end.append(0.0)  # a refusal means the user got no answer
            elif q.get("correctness"):
                end_to_end.append(1.0 if q["correctness"]["verdict"] == "correct" else 0.0)

    graded = [q["faithfulness"] for q in ok if q.get("faithfulness") and q["faithfulness"]["claims"] > 0]
    citations = [q["citation"] for q in answerable if q.get("citation")]
    totals = [
        q["latency_ms"]["retrieval"] + q["latency_ms"]["llm"] for q in ok if q.get("latency_ms")
    ]
    tokens = {
        "questions": len(ok),
        "generation_prompt": sum(q["usage"]["prompt_tokens"] for q in ok),
        "generation_completion": sum(q["usage"]["completion_tokens"] for q in ok),
        "judge_prompt": sum(q["usage"]["judge_prompt_tokens"] for q in ok),
        "judge_completion": sum(q["usage"]["judge_completion_tokens"] for q in ok),
    }

    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "errors": len(selected) - len(ok),
        "judge_failures": sum(1 for q in ok if q.get("judge_error")),
        "answerable": {
            "answered": share(answerable, "answered"),
            "refused": share(answerable, "insufficient_evidence"),
            "citation_error": share(answerable, "citation_error"),
        },
        "unanswerable": {
            "refused": share(unanswerable, "insufficient_evidence"),
            "answered": share(unanswerable, "answered"),
            "citation_error": share(unanswerable, "citation_error"),
        },
        "correctness": {
            "judged": judged,
            "verdicts": {name: verdicts[name] for name in VERDICT_SCORES},
            "correct_of_judged": ci(
                1.0 if q["correctness"]["verdict"] == "correct" else 0.0
                for q in answerable if q.get("correctness")
            ),
            "mean_score": ci(
                VERDICT_SCORES[q["correctness"]["verdict"]] for q in answerable if q.get("correctness")
            ),
            "end_to_end_correct": ci(end_to_end),
        },
        "faithfulness": {
            "n": len(graded),
            "per_answer": ci(f["supported"] / f["claims"] for f in graded),
            "claim_level": {
                "supported": sum(f["supported"] for f in graded),
                "total": sum(f["claims"] for f in graded),
            },
        },
        "citation": {
            "precision": ci(c["precision"] for c in citations),
            "recall": ci(c["recall"] for c in citations),
            "hits_relevant": ci(1.0 if c["hits_relevant"] else 0.0 for c in citations),
        },
        "latency": {"p50": percentile(totals, 50), "p95": percentile(totals, 95)} if totals else None,
        "tokens": tokens,
        "cost": _cost(meta, tokens, judge_enabled),
        "mean_total_tokens": (
            fmean(
                q["usage"]["prompt_tokens"] + q["usage"]["completion_tokens"]
                + q["usage"]["judge_prompt_tokens"] + q["usage"]["judge_completion_tokens"]
                for q in ok
            )
            if ok
            else None
        ),
    }