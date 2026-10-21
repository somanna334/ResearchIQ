import json
from types import SimpleNamespace

import pytest

from app.config import get_settings
from app.database import get_db
from app.embeddings.embedder import get_embedder
from app.evaluation.answers import RateLimiter, ThrottledLLM, run_answer_eval
from app.evaluation.dataset import load_dataset
from app.evaluation.judge import CORRECTNESS_SYSTEM
from app.llm.base import LLMError, LLMResult
from app.main import app

A_PAGES = ["Mitochondria produce energy for the cell", "Photosynthesis happens inside chloroplasts"]
B_PAGES = ["Gradient descent minimizes a loss function", "Backpropagation computes gradients in neural networks"]


class GenLLM:
    name = "fake-gen"
    model_name = "gen-1"

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def generate(self, system, prompt):
        self.calls += 1
        return LLMResult(self.reply, 100, 10)


class JudgeLLM:
    name = "fake-judge"
    model_name = "judge-1"

    def __init__(
        self,
        correctness='{"verdict": "correct", "reason": "matches"}',
        faithfulness='{"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false}]}',
    ):
        self.correctness = correctness
        self.faithfulness = faithfulness
        self.calls = 0

    def generate(self, system, prompt):
        self.calls += 1
        text = self.correctness if system == CORRECTNESS_SYSTEM else self.faithfulness
        return LLMResult(text, 50, 5)


class FlakyLLM:
    name = "flaky"
    model_name = "flaky-1"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate(self, system, prompt):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return LLMResult(outcome, 1, 1)


def _item(qid, text, relevant=None, qtype="factoid", split="test", reference=None):
    item = {"id": qid, "question": text, "answerable": relevant is not None, "type": qtype, "split": split}
    if relevant is not None:
        item["relevant"] = [{"document": d, "pages": p} for d, p in relevant]
    if reference:
        item["reference_answer"] = reference
    return json.dumps(item)


@pytest.fixture
def env(client, make_pdf, upload, tmp_path):
    for name, pages in (("a.pdf", A_PAGES), ("b.pdf", B_PAGES)):
        doc_id = upload(name, make_pdf(pages, name=name).read_bytes()).json()["id"]
        client.post(f"/documents/{doc_id}/embeddings")
    path = tmp_path / "answers.jsonl"
    path.write_text(
        "\n".join(
            [
                _item("q1", "chloroplasts photosynthesis", [("a.pdf", [2])],
                      reference="Photosynthesis happens in chloroplasts."),
                _item("q2", "zzzqqq", None, qtype="off_topic"),
                _item("q3", "gradient descent loss function", [("b.pdf", [1])], split="dev",
                      reference="It minimizes a loss function."),
            ]
        ),
        encoding="utf-8",
    )
    questions, sha = load_dataset(path)
    return SimpleNamespace(
        db=next(app.dependency_overrides[get_db]()),
        settings=app.dependency_overrides[get_settings](),
        embedder=app.dependency_overrides[get_embedder](),
        questions=questions,
        sha=sha,
    )


def run(env, gen, judge, questions=None, **kwargs):
    return run_answer_eval(
        env.db, env.settings, env.embedder, object(), gen, judge, questions or env.questions,
        mode="semantic", k=2, chunk_config="fixed-800-150",
        dataset_path="d.jsonl", dataset_sha=env.sha, run_id="20261019T100000Z-answers",
        log=lambda _: None, **kwargs,
    )


def by_id(results, qid):
    return next(q for q in results["questions"] if q["id"] == qid)


ANSWER = "Photosynthesis happens inside chloroplasts [S1]."


# ---- rate limiting and retries ----

def test_rate_limiter_spaces_calls_apart():
    now = [0.0]
    slept = []

    def fake_sleep(seconds):
        slept.append(round(seconds, 3))
        now[0] += seconds

    limiter = RateLimiter(60, clock=lambda: now[0], sleep=fake_sleep)
    limiter.wait()
    limiter.wait()
    limiter.wait()

    assert slept == [1.0, 1.0]


def _throttled(llm, backoffs=(1.0, 2.0)):
    slept = []
    return ThrottledLLM(llm, RateLimiter(0), backoffs=backoffs, sleep=slept.append), slept


def test_throttled_llm_retries_rate_limit_errors():
    llm = FlakyLLM([LLMError("429", retryable=True), "ok"])
    throttled, slept = _throttled(llm)

    assert throttled.generate("s", "p").text == "ok"
    assert slept == [1.0]
    assert llm.calls == 2 and throttled.retries == 1


def test_throttled_llm_does_not_retry_permanent_errors():
    llm = FlakyLLM([LLMError("bad key", retryable=False), "never"])
    throttled, slept = _throttled(llm)

    with pytest.raises(LLMError):
        throttled.generate("s", "p")
    assert llm.calls == 1 and slept == []


def test_throttled_llm_gives_up_after_the_last_backoff():
    llm = FlakyLLM([LLMError("429", retryable=True)] * 3)
    throttled, slept = _throttled(llm)

    with pytest.raises(LLMError):
        throttled.generate("s", "p")
    assert llm.calls == 3 and slept == [1.0, 2.0]


# ---- answer evaluation ----

def test_answerable_question_gets_citation_correctness_and_faithfulness(env):
    results = run(env, GenLLM(ANSWER), JudgeLLM())
    q1 = by_id(results, "q1")

    assert q1["status"] == "answered"
    assert q1["citation"]["precision"] == 1.0
    assert q1["citation"]["recall"] == 1.0
    assert q1["citation"]["hits_relevant"] is True
    assert q1["correctness"]["verdict"] == "correct"
    assert q1["faithfulness"] == {"claims": 2, "supported": 1}
    assert q1["usage"]["judge_prompt_tokens"] == 100  # two judge calls, 50 each


def test_citing_an_irrelevant_page_lowers_citation_precision(env):
    q1 = by_id(run(env, GenLLM("Both matter [S1][S2]."), JudgeLLM()), "q1")

    assert q1["citation"]["precision"] == 0.5
    assert q1["citation"]["recall"] == 1.0


def test_gated_unanswerable_question_makes_no_model_calls(env):
    gen, judge = GenLLM(ANSWER), JudgeLLM()
    only_q2 = [q for q in env.questions if q.id == "q2"]
    q2 = run(env, gen, judge, questions=only_q2)["questions"][0]

    assert q2["status"] == "insufficient_evidence" and q2["gated"] is True
    assert gen.calls == 0 and judge.calls == 0
    assert q2["citation"] is None and q2["correctness"] is None


def test_completed_questions_are_not_run_again(env):
    first = run(env, GenLLM(ANSWER), JudgeLLM())
    done = {q["id"]: q for q in first["questions"] if q["id"] in ("q1", "q3")}

    gen, judge = GenLLM("unused [S1]."), JudgeLLM()
    second = run(env, gen, judge, done=done)

    assert gen.calls == 0  # q2 is gated; q1 and q3 are reused
    assert [q["id"] for q in second["questions"]] == ["q1", "q2", "q3"]
    assert second["questions"][0] is done["q1"]


def test_unparseable_judge_output_is_flagged_not_fatal(env):
    judge = JudgeLLM(correctness="not json", faithfulness="also not json")
    q1 = by_id(run(env, GenLLM(ANSWER), judge), "q1")

    assert q1["status"] == "answered"
    assert q1["judge_error"] is True
    assert q1["correctness"] is None and q1["faithfulness"] is None


def test_without_a_judge_only_deterministic_metrics_exist(env):
    judge = JudgeLLM()
    results = run(env, GenLLM(ANSWER), judge, use_judge=False)
    q1 = by_id(results, "q1")

    assert judge.calls == 0
    assert q1["correctness"] is None and q1["faithfulness"] is None
    assert q1["judge_error"] is False
    assert q1["citation"]["precision"] == 1.0
    assert results["meta"]["judge"] == {"enabled": False, "model": None}