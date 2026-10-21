import random

from app.evaluation.summary import answer_summary


def _p(c: dict | None) -> str:
    if c is None:
        return "-"
    return f"{c['mean']:.0%} [{c['low']:.0%}, {c['high']:.0%}] (n={c['n']})"


def _section(title: str, s: dict) -> list[str]:
    cor, faith, cit = s["correctness"], s["faithfulness"], s["citation"]
    claims = faith["claim_level"]
    verdicts = cor["verdicts"]
    rows = [
        ("Answerable questions answered", _p(s["answerable"]["answered"])),
        ("Answerable questions refused (false refusal)", _p(s["answerable"]["refused"])),
        ("Answerable, answer withheld for bad citations", _p(s["answerable"]["citation_error"])),
        ("Correct end to end (a refusal counts as wrong)", _p(cor["end_to_end_correct"])),
        ("Correct, among answers that were judged", _p(cor["correct_of_judged"])),
        ("Verdicts (correct / partial / incorrect)",
         f"{verdicts['correct']} / {verdicts['partial']} / {verdicts['incorrect']}"),
        ("Faithfulness, per answer", _p(faith["per_answer"])),
        ("Faithfulness, claims supported", f"{claims['supported']} of {claims['total']}"),
        ("Citation precision", _p(cit["precision"])),
        ("Citation recall", _p(cit["recall"])),
        ("Cites at least one relevant page", _p(cit["hits_relevant"])),
        ("Unanswerable questions refused (correct)", _p(s["unanswerable"]["refused"])),
        ("Unanswerable questions answered (hallucination risk)", _p(s["unanswerable"]["answered"])),
        ("Unanswerable, answer withheld for bad citations", _p(s["unanswerable"]["citation_error"])),
        ("Latency p50 / p95 (retrieval + provider call)",
         f"{s['latency']['p50']:.0f} / {s['latency']['p95']:.0f} ms" if s["latency"] else "-"),
        ("Questions that errored / judge failures", f"{s['errors']} / {s['judge_failures']}"),
    ]
    lines = [
        f"## {title} ({s['n_answerable']} answerable, {s['n_unanswerable']} unanswerable)",
        "",
        "| Metric | Result (95% interval) |",
        "|---|---|",
        *[f"| {name} | {value} |" for name, value in rows],
        "",
    ]
    return lines


def render_answer_report(results: dict) -> str:
    meta = results["meta"]
    judge = meta.get("judge") or {}
    commit = meta.get("git_commit") or "unknown"
    if meta.get("git_dirty"):
        commit += " (uncommitted changes)"
    lines = [
        f"# Answer-quality evaluation {meta['run_id']}",
        "",
        f"- Code: `{commit}`",
        f"- Retrieval: `{meta['mode']}`, k={meta['k']}, chunks `{meta['chunk_config']}`, "
        f"refusal threshold {meta['settings']['min_top_score']}",
        f"- Generation model: `{meta['generation']['model']}`",
        f"- Judge: {'`' + str(judge.get('model')) + '`' if judge.get('enabled') else 'disabled'}",
        f"- Dataset: `{meta['dataset']['path']}` (sha256 `{meta['dataset']['sha256'][:12]}`), "
        f"{meta['dataset']['questions']} questions",
        "",
    ]
    lines += _section("Test split", answer_summary(results, "test"))
    lines += _section("Dev split", answer_summary(results, "dev"))

    overall = answer_summary(results, None)
    tokens, cost = overall["tokens"], overall["cost"]
    lines += [
        "## Tokens and cost (all questions)",
        "",
        f"- Generation tokens: {tokens['generation_prompt']} in / {tokens['generation_completion']} out",
        f"- Judge tokens: {tokens['judge_prompt']} in / {tokens['judge_completion']} out",
    ]
    if cost:
        lines.append(
            f"- Estimated paid-tier cost: {cost['currency']} {cost['total']:.4f} in total"
            + (f", {cost['per_question']:.5f} per question" if cost["per_question"] else "")
        )
    else:
        lines.append("- Cost not estimated: add prices to `eval/pricing.json` from the provider's pricing page.")
    lines += [
        "",
        "## Caveats",
        "",
        "- Correctness and faithfulness come from an LLM judge. Read the audit file and count disagreements before trusting them.",
        "- Citation metrics are deterministic but only as good as the page labels.",
        "- Small samples give wide intervals. A difference whose interval overlaps another's is not established.",
        "- Latency excludes judging and our own rate-limit waits.",
        "- Token counts come from the provider's usage metadata and may exclude hidden reasoning tokens.",
        "",
    ]
    return "\n".join(lines)


def render_audit(results: dict, n: int = 12, seed: int = 0) -> str:
    """A seeded sample of answers beside the judge's verdicts, for a human to check."""
    answered = [q for q in results["questions"] if q.get("status") == "answered"]
    sample = random.Random(seed).sample(answered, min(n, len(answered)))
    lines = [
        f"# Judge audit for {results['meta']['run_id']}",
        "",
        "For each item, decide whether you agree with the judge. Open the cited pages in the app to check "
        "faithfulness. This file contains model-written text and is not meant to be committed.",
        "",
    ]
    for index, q in enumerate(sample, start=1):
        correctness = q.get("correctness")
        faithfulness = q.get("faithfulness")
        lines += [
            f"## {index}. {q['id']} ({q['type']})",
            "",
            f"**Question:** {q['question']}",
            "",
            f"**Reference answer:** {q.get('reference_answer') or '(none)'}",
            "",
            f"**Model answer:** {q['answer']}",
            "",
            "**Cited pages:** "
            + (", ".join(f"{d} p.{p}" for d, p in q["citation"]["cited_pages"]) if q.get("citation") else "n/a"),
            "",
            "**Judge correctness:** "
            + (f"{correctness['verdict']} ({correctness['reason']})" if correctness else "not scored"),
            "",
            "**Judge faithfulness:** "
            + (f"{faithfulness['supported']} of {faithfulness['claims']} claims supported" if faithfulness else "not scored"),
            "",
            "**Do you agree?** ______",
            "",
        ]
    if not sample:
        lines.append("No answered questions to audit.")
    return "\n".join(lines)