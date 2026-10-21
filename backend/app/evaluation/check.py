"""Regression floor.

Write a floor from a run you trust:
    python -m app.evaluation.check --results eval/results/<run>.json --write-floor
Check a later run against it (exit code 1 on regression):
    python -m app.evaluation.check --results eval/results/<run>.json
"""
import argparse
import json
import math
import sys
from pathlib import Path
from statistics import fmean


def _questions(results: dict, experiment: str, split: str) -> list[dict]:
    exp = next((e for e in results["experiments"] if e["name"] == experiment), None)
    if exp is None:
        raise KeyError(experiment)
    return [q for q in exp["questions"] if q["answerable"] and q["split"] == split]


def metric_means(results: dict, experiment: str, split: str) -> tuple[dict[str, float], int]:
    k = results["meta"]["k"]
    questions = _questions(results, experiment, split)
    if not questions:
        return {}, 0
    keys = [f"recall@{k}", f"ndcg@{k}", "mrr"]
    return {m: fmean(q["metrics"][m] for q in questions) for m in keys}, len(questions)


def make_floor(results: dict, experiment: str, split: str, margin: float) -> dict:
    means, n = metric_means(results, experiment, split)
    if not n:
        raise ValueError(f"no answerable '{split}' questions for '{experiment}'")
    return {
        "dataset_sha256": results["meta"]["dataset"]["sha256"],
        "experiment": experiment,
        "split": split,
        "k": results["meta"]["k"],
        "n_questions": n,
        "margin": margin,
        "measured": {m: round(v, 4) for m, v in means.items()},
        # Round down so a run identical to the measured one can never fail.
        "min": {m: max(0.0, math.floor((v - margin) * 10000) / 10000) for m, v in means.items()},
    }


def check_floor(results: dict, floor: dict) -> list[str]:
    failures: list[str] = []
    if results["meta"]["dataset"]["sha256"] != floor["dataset_sha256"]:
        failures.append(
            "the dataset changed since the floor was written; the numbers aren't comparable. "
            "Re-measure and write a new floor."
        )
    if results["meta"]["k"] != floor["k"]:
        failures.append(f"headline k is {results['meta']['k']} but the floor was written for k={floor['k']}")
    try:
        means, n = metric_means(results, floor["experiment"], floor["split"])
    except KeyError:
        failures.append(f"experiment '{floor['experiment']}' not found in the results")
        return failures
    if not n:
        failures.append(f"no answerable '{floor['split']}' questions in the results")
        return failures
    for metric, minimum in floor["min"].items():
        if metric not in means:
            failures.append(f"{metric} was not measured")
        elif means[metric] + 1e-9 < minimum:
            failures.append(f"{metric}: {means[metric]:.3f} is below the floor {minimum:.3f}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--results", required=True)
    parser.add_argument("--floor", default="eval/floor.json")
    parser.add_argument("--experiment", default=None, help="default: the first experiment")
    parser.add_argument("--split", default="test")
    parser.add_argument("--write-floor", action="store_true")
    parser.add_argument("--margin", type=float, default=None,
                        help="default: the larger of 0.05 and one question's worth (1/n)")
    args = parser.parse_args(argv)

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    if args.write_floor:
        experiment = args.experiment or results["experiments"][0]["name"]
        _, n = metric_means(results, experiment, args.split)
        if not n:
            print(f"No answerable '{args.split}' questions for '{experiment}'.")
            return 1
        margin = args.margin if args.margin is not None else max(0.05, round(1 / n, 2))
        floor = make_floor(results, experiment, args.split, margin)
        Path(args.floor).write_text(json.dumps(floor, indent=2), encoding="utf-8")
        print(f"Wrote {args.floor}: {floor['measured']} measured, {floor['min']} minimum (margin {margin}).")
        return 0

    floor = json.loads(Path(args.floor).read_text(encoding="utf-8"))
    failures = check_floor(results, floor)
    if failures:
        print("REGRESSION:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("OK: all metrics are at or above the floor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())