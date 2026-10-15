"""Scoreboard / backtesting runner (ROADMAP section I).

Scores probabilistic forecasts against the labelled resolved-event dataset and
prints a markdown scoreboard. Forecasts can come from:

  * a built-in **walk-forward climatology baseline** (per-category base rate
    computed only from events resolved before each question's ``as_of`` date),
  * and/or an external **predictions file** produced by any model (the engine,
    a human, another LLM), so real forecasts can be graded identically.

Predictions-file format (JSON list):

    [{"id": "uk_2024", "p": 0.72, "predicted_date": "2024-06-01"}, ...]

``p`` is the probability the event occurs within [as_of, deadline]; ``id`` must
match the dataset. ``predicted_date`` (optional) enables lead-time scoring.

Run:
    backend/venv/bin/python backend/evaluation/scoreboard.py
    backend/venv/bin/python backend/evaluation/scoreboard.py --predictions preds.json
    backend/venv/bin/python backend/evaluation/scoreboard.py --json out.json
"""
import argparse
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation import metrics as M
from evaluation.dataset import DATASET, by_id

STRATEGIES = ["always_0.5", "overall_base_rate", "category_base_rate_walkforward"]


def _to_date(iso):
    y, m, d = (int(p) for p in iso.split("-"))
    return date(y, m, d)


def _decimal_year(iso):
    d = _to_date(iso)
    start = date(d.year, 1, 1)
    end = date(d.year + 1, 1, 1)
    return d.year + (d - start).days / (end - start).days


def walkforward_category_base_rate(record, records):
    """Base rate of the record's category among *earlier-resolved* questions.

    Falls back to the overall base rate of earlier questions, then to 0.5.
    """
    as_of = _to_date(record["as_of"])
    earlier = [r for r in records if r["id"] != record["id"]
               and r["resolved_on"] and _to_date(r["resolved_on"]) < as_of]
    same = [r for r in earlier if r["category"] == record["category"]]
    pool = same if same else earlier
    if not pool:
        return 0.5
    return sum(r["outcome"] for r in pool) / len(pool)


def build_baseline(records):
    """Return {strategy: {id: p}} for the built-in baselines."""
    overall = sum(r["outcome"] for r in records) / len(records)
    return {
        "always_0.5": {r["id"]: 0.5 for r in records},
        "overall_base_rate": {r["id"]: overall for r in records},
        "category_base_rate_walkforward": {
            r["id"]: walkforward_category_base_rate(r, records) for r in records
        },
    }


def load_predictions(path):
    with open(path) as fh:
        data = json.load(fh)
    return {str(item["id"]): item for item in data}


def score(name, preds, records, n_bins=5):
    ps = [preds[r["id"]]["p"] if isinstance(preds[r["id"]], dict) else preds[r["id"]]
          for r in records]
    ys = [r["outcome"] for r in records]
    summary = M.summarize(ps, ys, n_bins=n_bins)
    summary["name"] = name
    return summary, ps, ys


def lead_time(preds, records):
    predicted, actual = [], []
    for r in records:
        item = preds.get(r["id"])
        if not r["resolved_on"] or not item:
            continue
        pd = item.get("predicted_date") if isinstance(item, dict) else None
        if not pd:
            continue
        predicted.append(_decimal_year(pd))
        actual.append(_decimal_year(r["resolved_on"]))
    if not predicted:
        return None
    return M.lead_time_errors(predicted, actual)


def main():
    ap = argparse.ArgumentParser(description="Psychohistory evaluation scoreboard")
    ap.add_argument("--predictions", help="JSON file of model predictions to score")
    ap.add_argument("--json", dest="json_out", help="write the scoreboard to a JSON file")
    ap.add_argument("--bins", type=int, default=5)
    args = ap.parse_args()

    records = DATASET
    scores = []
    for name, preds in build_baseline(records).items():
        summary, _, _ = score(name, preds, records, args.bins)
        scores.append(summary)
    if args.predictions:
        ext = load_predictions(args.predictions)
        missing = [r["id"] for r in records if r["id"] not in ext]
        if missing:
            print(f"[warn] predictions missing for {len(missing)} ids: {missing[:5]}...")
        known = [r for r in records if r["id"] in ext]
        summary, _, _ = score("model:" + os.path.basename(args.predictions), ext, known, args.bins)
        scores.append(summary)

    print("=" * 78)
    print("Psychohistory evaluation scoreboard  (n_questions=%d)" % len(records))
    print("=" * 78)
    header = f"{'strategy':<34}{'Brier':>8}{'Skill':>8}{'LogLoss':>9}{'ECE':>7}{'MCE':>7}{'AUC':>7}{'Res':>7}"
    print(header)
    print("-" * len(header))
    for s in scores:
        print(f"{s['name']:<34}{s['brier']:>8.4f}{s['brier_skill']:>8.3f}"
              f"{s['log_loss']:>9.4f}{s['ece']:>7.4f}{s['mce']:>7.4f}"
              f"{s['auc']:>7.3f}{s['resolution']:>7.4f}")

    if args.predictions:
        ext = load_predictions(args.predictions)
        lt = lead_time(ext, records)
        if lt:
            print(f"\nLead-time (predicted vs actual): MAE={lt['mae']:.2f}y  bias={lt['bias']:+.2f}y  n={lt['n']}")

    print("\nReference: Brier skill > 0 beats the base-rate forecast; ECE/MCE -> 0")
    print("means calibrated; AUC -> 1.0 means the forecast ranks outcomes well.")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"n_questions": len(records), "scores": scores}, fh, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
