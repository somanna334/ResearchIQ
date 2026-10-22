import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.auth.deps import require_admin
from app.evaluation.summary import answer_summary, gate_summary, retrieval_summary

router = APIRouter(
    prefix="/eval", tags=["evaluation"], dependencies=[Depends(require_admin)]
)

# Only names the runners generate. This also blocks path tricks like "..".
RUN_ID = re.compile(r"^\d{8}T\d{6}Z(-answers)?$")

SPLITS: tuple[tuple[str, str | None], ...] = (("test", "test"), ("dev", "dev"), ("all", None))
META_KEYS = (
    "run_id", "created_utc", "git_commit", "git_dirty", "k", "mode", "chunk_config",
    "dataset", "corpus", "generation", "judge", "settings", "pricing",
)


class RunItem(BaseModel):
    id: str
    kind: Literal["retrieval", "answers"]
    created_utc: str
    git_commit: str | None
    questions: int
    label: str


def _kind(results: dict) -> str:
    return results.get("kind", "retrieval")


def _read(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("meta"), dict) else None


def _item(run_id: str, results: dict) -> RunItem:
    meta = results["meta"]
    if _kind(results) == "answers":
        label = f"{meta.get('mode', '?')}, k={meta.get('k', '?')}"
        questions = len(results.get("questions", []))
    else:
        names = [e.get("name", "?") for e in results.get("experiments", [])]
        label = f"{len(names)} experiments"
        questions = (meta.get("dataset") or {}).get("questions", 0)
    return RunItem(
        id=run_id,
        kind=_kind(results),  # type: ignore[arg-type]
        created_utc=str(meta.get("created_utc", "")),
        git_commit=meta.get("git_commit"),
        questions=questions,
        label=label,
    )


@router.get("/runs", response_model=list[RunItem])
def list_runs(settings: Annotated[Settings, Depends(get_settings)]) -> list[RunItem]:
    """Evaluation runs on disk, newest first. Unreadable or foreign files are skipped."""
    folder = settings.eval_results_dir
    if not folder.is_dir():
        return []
    items = []
    for path in sorted(folder.glob("*.json"), reverse=True):
        if not RUN_ID.fullmatch(path.stem):
            continue
        results = _read(path)
        if results is not None:
            items.append(_item(path.stem, results))
    return items


@router.get("/runs/{run_id}")
def get_run(run_id: str, settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, Any]:
    if not RUN_ID.fullmatch(run_id):
        raise HTTPException(status_code=404, detail="Run not found")
    path = settings.eval_results_dir / f"{run_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Run not found")
    results = _read(path)
    if results is None:
        raise HTTPException(status_code=422, detail="The results file could not be read")

    meta = {key: results["meta"][key] for key in META_KEYS if key in results["meta"]}
    if _kind(results) == "answers":
        return {
            "id": run_id,
            "kind": "answers",
            "meta": meta,
            "splits": {name: answer_summary(results, split) for name, split in SPLITS},
        }
    threshold = float((results["meta"].get("settings") or {}).get("min_top_score", 0.0))
    return {
        "id": run_id,
        "kind": "retrieval",
        "meta": meta,
        "k": results["meta"]["k"],
        "splits": {name: retrieval_summary(results, split) for name, split in SPLITS},
        "gate": {name: gate_summary(results, split, threshold) for name, split in SPLITS},
    }