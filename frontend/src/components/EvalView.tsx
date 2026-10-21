import { useCallback, useEffect, useState } from "react";
import { getRun, listRuns } from "../evaluation";
import type {
  AnswerSummary,
  AnswersDetail,
  RetrievalDetail,
  RunDetail,
  RunItem,
  SplitName,
} from "../evaluation";
import { ciText, deltaText, money, pct, pctCi } from "../format";
import "../Eval.css";

type Loadable<T> =
  | { state: "idle" }
  | { state: "loading" }
  | { state: "error"; message: string }
  | { state: "done"; data: T };

function messageOf(err: unknown): string {
  return err instanceof Error ? err.message : "Request failed";
}

function RetrievalSection({ detail, split }: { detail: RetrievalDetail; split: SplitName }) {
  const rows = detail.splits[split];
  const gate = detail.gate[split];
  const k = detail.k;
  const n = rows[0]?.n ?? 0;
  return (
    <>
      {n < 15 && (
        <div className="banner warn">
          Only {n} answerable questions in this split, so the intervals are wide. Treat every
          difference as unproven.
        </div>
      )}
      <div className="table-wrap">
        <table className="eval">
          <thead>
            <tr>
              <th>Experiment</th>
              <th>n</th>
              <th>Recall@{k}</th>
              <th>Precision@{k}</th>
              <th>MRR</th>
              <th>nDCG@{k}</th>
              <th>ΔRecall</th>
              <th>ΔMRR</th>
              <th>p50 / p95 ms</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => {
              const recall = row.metrics[`recall@${k}`];
              return (
                <tr key={row.name}>
                  <td>
                    <strong>{row.name}</strong>
                  </td>
                  <td>{row.n}</td>
                  <td>
                    {ciText(recall)}
                    {recall && (
                      <div className="bar">
                        <div style={{ width: pct(recall.mean) }} />
                      </div>
                    )}
                  </td>
                  <td>{ciText(row.metrics[`precision@${k}`])}</td>
                  <td>{ciText(row.metrics["mrr"])}</td>
                  <td>{ciText(row.metrics[`ndcg@${k}`])}</td>
                  <td>{index === 0 ? "baseline" : deltaText(row.delta?.recall ?? null)}</td>
                  <td>{index === 0 ? "baseline" : deltaText(row.delta?.mrr ?? null)}</td>
                  <td>{row.latency ? `${row.latency.p50.toFixed(0)} / ${row.latency.p95.toFixed(0)}` : "–"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="muted small">
        Means with 95% bootstrap intervals. A * means the paired difference from the baseline has
        an interval that excludes 0. Without it, the data can't tell the configuration from the
        baseline.
      </p>

      <h2>Refusal threshold (baseline's best similarity)</h2>
      {gate ? (
        <>
          <p className="muted small">
            {gate.answerable} answerable and {gate.negatives} unanswerable questions. Current
            setting: {gate.current}. Choose on dev, then read the same row on test.
          </p>
          <div className="table-wrap">
            <table className="eval">
              <thead>
                <tr>
                  <th>Threshold</th>
                  <th>False refuses</th>
                  <th>False accepts</th>
                </tr>
              </thead>
              <tbody>
                {gate.rows.map((r) => (
                  <tr key={r.threshold}>
                    <td>{r.threshold.toFixed(2)}</td>
                    <td>
                      {r.false_refuse}/{gate.answerable}
                    </td>
                    <td>
                      {r.false_accept}/{gate.negatives}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <p className="muted small">This split needs both answerable and unanswerable questions.</p>
      )}
    </>
  );
}

function AnswerSection({ detail, split }: { detail: AnswersDetail; split: SplitName }) {
  const s: AnswerSummary = detail.splits[split];
  const judge = detail.meta.judge;
  const claims = s.faithfulness.claim_level;
  const perQuestion = s.tokens.questions > 0 ? s.mean_total_tokens : null;
  const cards: { label: string; value: string; note: string }[] = [
    {
      label: "Correct end to end",
      value: pctCi(s.correctness.end_to_end_correct),
      note: "Refusals count as wrong. Scored by an LLM judge.",
    },
    {
      label: "Faithfulness",
      value: pctCi(s.faithfulness.per_answer),
      note: `${claims.supported} of ${claims.total} claims supported`,
    },
    {
      label: "Citation precision",
      value: pctCi(s.citation.precision),
      note: "Cited passages that sit on a relevant page",
    },
    {
      label: "Cites a relevant page",
      value: pctCi(s.citation.hits_relevant),
      note: `Recall: ${pctCi(s.citation.recall)}`,
    },
    {
      label: "Wrongly refused",
      value: pctCi(s.answerable.refused),
      note: "Answerable questions the system declined",
    },
    {
      label: "Correctly refused",
      value: pctCi(s.unanswerable.refused),
      note: "Unanswerable questions the system declined",
    },
    {
      label: "Wrongly answered",
      value: pctCi(s.unanswerable.answered),
      note: "Unanswerable questions that got an answer (hallucination risk)",
    },
    {
      label: "Latency p50 / p95",
      value: s.latency ? `${s.latency.p50.toFixed(0)} / ${s.latency.p95.toFixed(0)} ms` : "–",
      note: "Retrieval plus the provider call; excludes judging",
    },
    {
      label: "Tokens per question",
      value: perQuestion !== null ? perQuestion.toFixed(0) : "–",
      note: `Generation ${s.tokens.generation_prompt + s.tokens.generation_completion} + judge ${
        s.tokens.judge_prompt + s.tokens.judge_completion
      } in total`,
    },
    {
      label: "Estimated cost",
      value: s.cost ? money(s.cost.total, s.cost.currency) : "not computed",
      note: s.cost
        ? "Paid-tier estimate; the free tier costs nothing"
        : "Add prices to eval/pricing.json",
    },
  ];
  return (
    <>
      <div className="banner warn small">
        {judge?.enabled ? (
          <>
            Correctness and faithfulness are scored by an LLM judge (<code>{judge.model}</code>). It
            can be wrong. Check the audit file from the run before trusting these numbers.
          </>
        ) : (
          <>The judge was disabled, so only citation and refusal metrics exist.</>
        )}
      </div>
      <p className="muted small">
        {s.n_answerable} answerable and {s.n_unanswerable} unanswerable questions
        {s.errors > 0 && ` · ${s.errors} errored`}
        {s.judge_failures > 0 && ` · ${s.judge_failures} judge failures`}. Values show a mean with a
        95% interval.
      </p>
      {s.n_answerable < 15 && (
        <div className="banner warn small">
          Fewer than 15 answerable questions here, so the intervals are very wide.
        </div>
      )}
      <div className="cards">
        {cards.map((card) => (
          <div className="stat" key={card.label}>
            <div className="label">{card.label}</div>
            <div className="value">{card.value}</div>
            <div className="note">{card.note}</div>
          </div>
        ))}
      </div>
    </>
  );
}

export default function EvalView({ active }: { active: boolean }) {
  const [runs, setRuns] = useState<RunItem[]>([]);
  const [listError, setListError] = useState<string | null>(null);
  const [runId, setRunId] = useState("");
  const [split, setSplit] = useState<SplitName>("test");
  const [detail, setDetail] = useState<Loadable<RunDetail>>({ state: "idle" });

  const loadRuns = useCallback(async () => {
    try {
      const items = await listRuns();
      setRuns(items);
      setListError(null);
      setRunId((current) => (items.some((r) => r.id === current) ? current : (items[0]?.id ?? "")));
    } catch (err) {
      setListError(messageOf(err));
    }
  }, []);

  useEffect(() => {
    if (active) void loadRuns();
  }, [active, loadRuns]);

  useEffect(() => {
    if (!runId) {
      setDetail({ state: "idle" });
      return;
    }
    let cancelled = false;
    setDetail({ state: "loading" });
    getRun(runId)
      .then((data) => {
        if (!cancelled) setDetail({ state: "done", data });
      })
      .catch((err: unknown) => {
        if (!cancelled) setDetail({ state: "error", message: messageOf(err) });
      });
    return () => {
      cancelled = true;
    };
  }, [runId]);

  const data = detail.state === "done" ? detail.data : null;

  return (
    <div className="evalview">
      <div className="eval-inner">
        <div className="eval-controls">
          <label className="small">
            Run{" "}
            <select value={runId} onChange={(e) => setRunId(e.target.value)} disabled={runs.length === 0}>
              {runs.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.kind === "answers" ? "Answers" : "Retrieval"} · {r.created_utc.slice(0, 16).replace("T", " ")} ·{" "}
                  {r.label}
                </option>
              ))}
            </select>
          </label>
          <label className="small">
            Split{" "}
            <select value={split} onChange={(e) => setSplit(e.target.value as SplitName)}>
              <option value="test">Test (report this)</option>
              <option value="dev">Dev (tune on this)</option>
              <option value="all">All</option>
            </select>
          </label>
          <button className="link-btn" onClick={() => void loadRuns()}>
            Refresh
          </button>
        </div>

        {listError && <div className="banner bad">{listError}</div>}
        {runs.length === 0 && !listError && (
          <div className="banner warn">
            No evaluation runs yet. From the backend folder, run{" "}
            <code>python -m app.evaluation.run</code> (retrieval) or{" "}
            <code>python -m app.evaluation.answer_run</code> (answers), then press Refresh.
          </div>
        )}
        {detail.state === "loading" && <p className="muted">Loading…</p>}
        {detail.state === "error" && <div className="banner bad">{detail.message}</div>}

        {data && (
          <>
            <p className="muted small">
              {data.meta.dataset.questions} questions · code {data.meta.git_commit ?? "unknown"}
              {data.meta.git_dirty ? " (uncommitted changes)" : ""} · {data.meta.corpus.length}{" "}
              documents
            </p>
            {data.kind === "retrieval" ? (
              <RetrievalSection detail={data} split={split} />
            ) : (
              <AnswerSection detail={data} split={split} />
            )}
          </>
        )}
      </div>
    </div>
  );
}