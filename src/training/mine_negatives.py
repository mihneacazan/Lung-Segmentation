"""
Finds the negative slices a trained model actually gets wrong.

The project already has a `hard_negatives` sampling mode, but it picks negatives
by *distance*: everything within a margin of the tumour's z-range. That is a
proxy. It assumes the slices next to the lesion are the ones the model confuses,
without ever asking the model.

This module asks the model. It runs a finished checkpoint over the training
patients and scores every negative slice by how much tumour it hallucinated
there. The result is a ranking that can be fed back as a sampling pool, so the
second stage trains on the errors the model demonstrably makes rather than the
ones anatomy suggests it might.

Two scores are recorded per slice:

    fp_pixels   pixels above the run's own threshold. The literal reading of
                "the slices where it produces the most false positives", and the
                one used for ranking.
    prob_mass   the summed probability over the slice, thresholdless. Breaks the
                large number of ties at fp_pixels == 0 and keeps a usable
                ordering below the threshold.

A measured caveat that shapes how the pool is used: on the committed baseline
only about 9% of training negatives carry any false positive at all. Ranking is
therefore informative over roughly the top tenth of slices and arbitrary below
it, where every score is zero. A consumer that asks for a pool larger than the
number of non-zero slices is not getting "harder" negatives for the remainder,
it is getting random ones, and `pool_quality` reports exactly that fraction so
the experiment can state it rather than assume otherwise.

Mining runs on the training split by design - that is what the model was fitted
on, so the errors that survive there are the stubborn ones. It also means the
rate understates what the model would do on unseen data.

Usage:
    python -m src.training.mine_negatives --run baseline/seed_42
    python -m src.training.mine_negatives --run baseline/seed_42 --threshold 0.5
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from src.config import OUTPUT_DIR

MINED_FILENAME = "mined_negatives.json"


def _as_slice_first(probs, n_slices):
    """Predictions come back (H, W, S) or (S, H, W); scoring wants slice-first."""
    arr = np.asarray(probs)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D probability volume, got {arr.shape}")
    if arr.shape[-1] == n_slices and arr.shape[0] != n_slices:
        return np.moveaxis(arr, -1, 0)
    return arr


def score_volume(probs, positive_slices, n_slices, threshold):
    """
    Scores each negative slice of one patient.

    Positive slices are skipped entirely: a predicted pixel there may be correct,
    so it is not a false positive and the slice is not a negative to mine.

    Returns:
        dict: {slice_index: {"fp_pixels": int, "prob_mass": float}}
    """
    arr = _as_slice_first(probs, n_slices)
    positives = set(int(s) for s in positive_slices)
    out = {}
    for s in range(min(n_slices, arr.shape[0])):
        if s in positives:
            continue
        plane = arr[s]
        out[s] = {"fp_pixels": int((plane > threshold).sum()),
                  "prob_mass": float(plane.sum())}
    return out


def rank_slices(scored):
    """
    Orders one patient's negatives worst-first.

    Sorts by false-positive pixels, then by probability mass, then by index. The
    second key matters more than it looks: most slices tie at zero pixels, and
    without it the ordering below the threshold would be arbitrary rather than
    merely weak.
    """
    return sorted(scored, key=lambda s: (-scored[s]["fp_pixels"],
                                         -scored[s]["prob_mass"], s))


def pool_quality(mined, pool_sizes=None):
    """
    How far the ranking carries real signal.

    Args:
        mined (dict): {case_id: {slice_index: scores}}.
        pool_sizes (dict): Optional {case_id: n} the consumer intends to draw.
            Where n exceeds the count of slices with a false positive, the
            surplus is effectively random and is reported as such.

    Returns:
        dict: totals, the fraction of negatives carrying any false positive, and
            - when pool_sizes is given - how much of the requested pool is
            backed by an actual error.
    """
    total = sum(len(v) for v in mined.values())
    nonzero = sum(1 for v in mined.values()
                  for s in v if v[s]["fp_pixels"] > 0)
    out = {"n_negative_slices": total,
           "n_with_false_positive": nonzero,
           "fraction_with_false_positive": (nonzero / total) if total else 0.0}
    if pool_sizes:
        want = backed = 0
        for case_id, n in pool_sizes.items():
            scored = mined.get(case_id, {})
            hits = sum(1 for s in scored if scored[s]["fp_pixels"] > 0)
            want += int(n)
            backed += min(int(n), hits)
        out["pool_requested"] = want
        out["pool_backed_by_error"] = backed
        out["pool_backed_fraction"] = (backed / want) if want else 0.0
    return out


def mine_run(run, split="train", threshold=None, preprocessed_name="preprocessed",
             output_dir=None, verbose=True):
    """
    Runs a finished checkpoint over `split` and scores its negative slices.

    Args:
        run (str): "exp_name/seed_dir" under output/experiments/.
        split (str): Which split to mine. 'train' is the point of the exercise.
        threshold (float): Cut for counting a pixel as a false positive.
            Defaults to the run's own validation-chosen threshold, so "false
            positive" means the same thing here as in its reported metrics.

    Returns:
        tuple: (mined dict, metadata dict)
    """
    from src.evaluation.hierarchical_report import load_run, predict_patient

    base = output_dir or OUTPUT_DIR
    run_dir = os.path.join(base, "experiments", *run.split("/"))
    with open(os.path.join(base, preprocessed_name, "index.json")) as f:
        index = json.load(f)

    if threshold is None:
        sweep_path = os.path.join(run_dir, "threshold_sweep.json")
        with open(sweep_path) as f:
            threshold = float(json.load(f)["best_threshold"])

    model, device, config, _ = load_run(run_dir, weights="best_model.pt")

    mined = {}
    case_ids = sorted(index["splits"][split])
    for n, case_id in enumerate(case_ids, 1):
        info = index["cases"][case_id]
        probs = predict_patient(model, device, config, case_id, preprocessed_name)
        mined[case_id] = score_volume(probs, info["positive_slices"],
                                      info["n_slices"], threshold)
        del probs
        if verbose:
            hits = sum(1 for s in mined[case_id]
                       if mined[case_id][s]["fp_pixels"] > 0)
            print(f"  [{n:2}/{len(case_ids)}] {case_id}: "
                  f"{len(mined[case_id])} negatives, {hits} with a false positive",
                  flush=True)

    meta = {"run": run, "split": split, "threshold": float(threshold),
            "quality": pool_quality(mined)}
    return mined, meta


def write_mined(mined, meta, path):
    """Writes the ranking to disk, worst-first per patient."""
    payload = {"meta": meta,
               "ranked": {c: rank_slices(v) for c, v in mined.items()},
               "scores": {c: {str(s): v[s] for s in v} for c, v in mined.items()}}
    with open(path, "w") as f:
        json.dump(payload, f)
    return path


def load_mined(path):
    """
    Reads a mined file back as {case_id: {slice index: false-positive pixels}}.

    The sampler needs magnitudes rather than an ordering. Truncating the ranking
    to a fixed pool and drawing uniformly inside it throws the magnitudes away,
    and measurement showed what that costs: with a pool sized to match the
    distance mode, only 33% of the slices actually drawn carried a false
    positive, against 58% when the draw is weighted. The ordering is still
    written to the file for reporting.
    """
    with open(path) as f:
        payload = json.load(f)
    scores = {c: {int(s): v["fp_pixels"] for s, v in per.items()}
              for c, per in payload["scores"].items()}
    return scores, payload.get("meta", {})


def load_mined_ranking(path):
    """The worst-first ordering, for reports rather than for sampling."""
    with open(path) as f:
        payload = json.load(f)
    return {c: [int(s) for s in v] for c, v in payload["ranked"].items()}


def main():
    parser = argparse.ArgumentParser(
        description="Score the negative slices a trained model gets wrong.")
    parser.add_argument("--run", required=True, help="exp_name/seed_dir")
    parser.add_argument("--split", default="train")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Defaults to the run's own chosen threshold.")
    parser.add_argument("--preprocessed_name", default="preprocessed")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    mined, meta = mine_run(args.run, split=args.split, threshold=args.threshold,
                           preprocessed_name=args.preprocessed_name)
    out = args.out or os.path.join(OUTPUT_DIR, MINED_FILENAME)
    write_mined(mined, meta, out)

    q = meta["quality"]
    print(f"\n  threshold {meta['threshold']:.2f} on the {meta['split']} split")
    print(f"  {q['n_negative_slices']:,} negative slices, "
          f"{q['n_with_false_positive']:,} with a false positive "
          f"({q['fraction_with_false_positive']:.1%})")
    print(f"  Below that fraction the ranking is ties at zero, so a pool larger")
    print(f"  than {q['n_with_false_positive']:,} slices is partly random.")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
