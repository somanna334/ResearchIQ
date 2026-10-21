from collections import Counter, defaultdict
from statistics import fmean

from app.evaluation.gate import DEFAULT_THRESHOLDS, gate_sweep
from app.evaluation.stats import bootstrap_ci, paired_difference_ci, percentile

MIN_RELIABLE_ANSWERABLE = 15


def _titles(k: int) -> dict[str, str]:
    return {
        f"recall@{k}": f"Recall@{k}",
        f"precision@{k}": f"Precision@{k}",
        "mrr": "MRR",
        f"ndcg@{k}": f"nDCG@{k}",
    }


def _answerable(experiment: dict, split: str | None = None) -> list[dict]:
    return [
        q for q in experiment["questions"]
        if q["answerable"] and (split is None or q["split"] == split)
    ]


def _ci_text(values: list[float]) -> str:
    mean, low, high = bootstrap_ci(values)
    return f"{mean:.2f} [{low:.2f}, {high:.2f}]"


def _delta_text(questions: list[dict], base_by_id: dict[str, dict], metric: str) -> str:
    pairs = [
        (q["metrics"][metric], base_by_id[q["id"]]["metrics"][metric])
        for q in questions if q["id"] in base_by_id
    ]
    if not pairs:
        return "-"
    mean, low, high = paired_difference_ci([a for a, _ in pairs], [b for _, b in pairs])
    mark = " *" if low > 0 or high < 0 else ""
    return f"{mean:+.2f} [{low:+.2f}, {high:+.2f}]{mark}"


def _headline(results: dict, split: str, k: int) -> list[str]:
    titles = _titles(k)
    metrics = list(titles)
    experiments = results["experiments"]
    base_by_id = {q["id"]: q for q in _answerable(experiments[0], split)}
    out = [
        f"## Retrieval quality ({split} split)",
        "",
        "| Experiment | n | " + " | ".join(titles[m] for m in metrics)
        + f" | ΔRecall@{k} vs baseline | ΔMRR vs baseline | p50 ms | p95 ms |",
        "|" + "---|" * (len(metrics) + 6),
    ]
    for index, exp in enumerate(experiments):
        qs = _answerable(exp, split)
        latencies = [q["latency_ms"] for q in exp["questions"] if q["split"] == split]
        p50 = f"{percentile(latencies, 50):.0f}" if latencies else "-"
        p95 = f"{percentile(latencies, 95):.0f}" if latencies else "-"
        if qs:
            cells = [_ci_text([q["metrics"][m] for q in qs]) for m in metrics]
            if index == 0:
                deltas = ["baseline", "baseline"]
            else:
                deltas = [
                    _delta_text(qs, base_by_id, f"recall@{k}"),
                    _delta_text(qs, base_by_id, "mrr"),
                ]
        else:
            cells = ["-"] * len(metrics)
            deltas = ["-", "-"]
        out.append(f"| {exp['name']} | {len(qs)} | " + " | ".join(cells + deltas) + f" | {p50} | {p95} |")
    out += [
        "",
        "Means with 95% bootstrap intervals in brackets. A `*` means the paired difference from "
        "the baseline has an interval that excludes 0. Without a `*`, the data can't tell the "
        "configuration from the baseline. Latency is retrieval only, after a warm-up call, over "
        "every question in the split.",
        "",
    ]
    return out


def _by_type(experiment: dict, k: int) -> list[str]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for q in _answerable(experiment):
        groups[q["type"]].append(q)
    out = [
        f"## Question types ({experiment['name']}; dev and test combined, descriptive only)",
        "",
        f"| Type | n | Recall@{k} | MRR |",
        "|---|---|---|---|",
    ]
    for qtype in sorted(groups):
        qs = groups[qtype]
        out.append(
            f"| {qtype} | {len(qs)} | {fmean(q['metrics'][f'recall@{k}'] for q in qs):.2f} "
            f"| {fmean(q['metrics']['mrr'] for q in qs):.2f} |"
        )
    if not groups:
        out.append("| (no answerable questions) | 0 | - | - |")
    return out + [""]


def _gate(experiment: dict, current: float) -> list[str]:
    out = [
        f"## Refusal threshold ({experiment['name']})",
        "",
        "Best dense similarity per question. Choose a threshold on the dev rows, then read the "
        "same row on test. A false refuse blocks an answerable question; a false accept lets an "
        "unanswerable one reach the model.",
        "",
    ]
    for split in ("dev", "test"):
        answerable = [
            q["best_dense"] for q in experiment["questions"]
            if q["answerable"] and q["split"] == split and q["best_dense"] is not None
        ]
        negatives = [
            q["best_dense"] for q in experiment["questions"]
            if not q["answerable"] and q["split"] == split and q["best_dense"] is not None
        ]
        out += [f"### {split} split ({len(answerable)} answerable, {len(negatives)} unanswerable)", ""]
        if not answerable or not negatives:
            out += ["Needs both answerable and unanswerable questions in this split.", ""]
            continue
        now = gate_sweep(answerable, negatives, [current])[0]
        out += [
            f"At the current setting ({current}): refuses {now.false_refuse}/{now.answerable} "
            f"answerable questions and lets through {now.false_accept}/{now.negatives} "
            "unanswerable ones.",
            "",
            "| Threshold | False refuses | False accepts |",
            "|---|---|---|",
        ]
        for row in gate_sweep(answerable, negatives, DEFAULT_THRESHOLDS):
            out.append(
                f"| {row.threshold:.2f} | {row.false_refuse}/{row.answerable} "
                f"| {row.false_accept}/{row.negatives} |"
            )
        out.append("")
    return out


def _failures(experiments: list[dict], k: int) -> list[str]:
    out = ["## Failures (no relevant page in the top results)", ""]
    for exp in experiments:
        answerable = _answerable(exp)
        missed = [q for q in answerable if q["metrics"][f"recall@{k}"] == 0]
        out.append(f"### {exp['name']}: {len(missed)} of {len(answerable)}")
        for q in missed[:10]:
            expected = ", ".join(f"{d} p.{p}" for d, p in q["relevant"])
            got: list[str] = []
            seen: set[tuple[str, int]] = set()
            for r in q["retrieved"]:
                page = (r["document"], r["page"])
                if page not in seen:
                    seen.add(page)
                    got.append(f"{page[0]} p.{page[1]}")
                if len(got) == 3:
                    break
            out.append(f"- [{q['split']}] {q['id']} \"{q['question'][:70]}\" expected {expected}; got {', '.join(got)}")
        out.append("")
    return out


def render_report(results: dict, current_threshold: float, detail: str | None = None) -> str:
    meta = results["meta"]
    k = meta["k"]
    experiments = results["experiments"]
    detail_exp = next((e for e in experiments if e["name"] == detail), experiments[0])

    counts = Counter(
        (q["split"], "answerable" if q["answerable"] else "unanswerable")
        for q in experiments[0]["questions"]
    )
    commit = meta.get("git_commit") or "unknown"
    if meta.get("git_dirty"):
        commit += " (uncommitted changes)"
    lines = [
        f"# Retrieval evaluation {meta['run_id']}",
        "",
        f"- Code: `{commit}`, Python {meta['python']}",
        f"- Dataset: `{meta['dataset']['path']}` (sha256 `{meta['dataset']['sha256'][:12]}`), "
        f"{meta['dataset']['questions']} questions",
        "- Questions: " + ", ".join(
            f"{counts[(s, t)]} {s} {t}" for s in ("dev", "test") for t in ("answerable", "unanswerable")
        ),
        f"- Headline K: {k}. Baseline: `{experiments[0]['name']}` (first experiment)",
        "",
        "| Corpus document | Pages | sha256 |",
        "|---|---|---|",
        *[f"| {d['filename']} | {d['pages']} | `{d['sha256'][:12]}` |" for d in meta["corpus"]],
        "",
    ]
    lines += _headline(results, "test", k)
    lines += _headline(results, "dev", k)
    lines += _by_type(detail_exp, k)
    lines += _gate(detail_exp, current_threshold)
    lines += _failures(experiments, k)

    test_answerable = len(_answerable(experiments[0], "test"))
    caveats = []
    if test_answerable < MIN_RELIABLE_ANSWERABLE:
        caveats.append(
            f"Only {test_answerable} answerable test questions. The intervals above are very wide; "
            "treat every difference as unproven."
        )
    caveats += [
        "Labels are page-level, so a page holding several chunks counts all of them as relevant.",
        "One corpus and one author's questions: results may not transfer to other documents.",
        "Retrieval metrics don't measure answer quality. That comes in the next step.",
    ]
    lines += ["## Caveats", "", *[f"- {c}" for c in caveats], ""]
    return "\n".join(lines)