"""
Scores existing checkpoints at three granularities, for cross-study comparison.

This project reports one number per patient: Dice over the whole reconstructed
volume, where a false positive anywhere in the scan counts against the score.
Much of the pulmonary nodule literature reports per-slice or per-lesion Dice
instead, and those are systematically higher for the same prediction. Comparing
across the two without saying which is which is how a model that fires on half
the empty slices comes to look better than one that does not.

The three views, on the same predictions:

    per-slice, tumour slices only   Delineation given that a lesion is present.
                                    False positives on empty slices are invisible.
    per-slice, every slice          Adds the empty slices, so false positives
                                    finally cost something.
    per-lesion                      One score per ground-truth component. False
                                    positives away from every lesion belong to
                                    none of them.
    per-patient                     What this project quotes.

On the Decathlon lung task the per-lesion view is close to degenerate, and the
report says so rather than hiding it: every test patient has one component
holding 99.4% or more of the tumour, with the rest being fragments of a few
voxels. Filtered at any sensible size those fragments vanish and per-lesion Dice
collapses onto per-patient Dice. Unfiltered they score zero and pull the mean
below it. Either way the metric carries no information here that per-patient
Dice does not already carry, which is itself worth being able to demonstrate.

Surface distances are not computed. They cost roughly 160 s per patient against
7 s for everything else and answer a question about boundaries that none of
these three views is asking.

Usage:
    python -m src.evaluation.hierarchical_report
    python -m src.evaluation.hierarchical_report --runs baseline,unet_25d
    python -m src.evaluation.hierarchical_report --min_lesion_voxels 100
"""

import argparse
import csv
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from src.config import OUTPUT_DIR, resolve_nifti_path
from src.evaluation.metrics import (
    compute_2d_slice_metrics,
    compute_dice_3d,
    compute_lesion_metrics,
    compute_precision_3d,
    compute_sensitivity_3d,
    count_false_positive_components,
    filter_predicted_components,
    reconstruct_patient_3d_volume,
)
from src.models.factory import build_model


DEFAULT_RUNS = ("baseline", "unet_25d", "attention_unet", "segresnet")


def load_run(run_dir):
    """Builds the architecture recorded in config.json and loads its weights."""
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model_type"], in_channels=config["n_adjacent"],
                        out_channels=1).to(device)
    state = torch.load(os.path.join(run_dir, "best_model.pt"),
                       map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()

    threshold = 0.5
    report_path = os.path.join(run_dir, "benchmark_report.json")
    if os.path.exists(report_path):
        with open(report_path) as f:
            threshold = float(json.load(f).get("optimal_threshold", 0.5))

    return model, device, config, threshold


def predict_patient(model, device, config, case_id, preprocessed_name="preprocessed"):
    """
    Runs the model over every slice of one patient.

    Neighbour selection for the 2.5D models replicates `LungSliceDataset`, edge
    clipping included: out-of-range neighbours repeat the edge slice rather than
    being zero-filled, because zero is a real intensity here and means air.
    """
    n_adjacent = config["n_adjacent"]
    half = n_adjacent // 2

    volumes = os.path.join(OUTPUT_DIR, preprocessed_name, "volumes")
    img = np.load(os.path.join(volumes, f"{case_id}_img.npy"), mmap_mode="r")
    n_slices = img.shape[2]

    probs = np.zeros((img.shape[0], img.shape[1], n_slices), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n_slices, 16):
            stop = min(start + 16, n_slices)
            batch = [np.stack([np.asarray(img[:, :, int(np.clip(s + o, 0, n_slices - 1))],
                                          dtype=np.float32)
                               for o in range(-half, half + 1)])
                     for s in range(start, stop)]
            out = torch.sigmoid(model(torch.from_numpy(np.stack(batch)).to(device)))
            probs[:, :, start:stop] = np.moveaxis(out.cpu().numpy()[:, 0], 0, -1)

    return probs


def score_prediction(pred, gt, min_lesion_voxels=0):
    """Scores one binary prediction at all three granularities."""
    scores = {
        "dice_3d_patient": compute_dice_3d(pred, gt),
        "sensitivity_3d_patient": compute_sensitivity_3d(pred, gt),
        "precision_3d_patient": compute_precision_3d(pred, gt),
        "fp_components": count_false_positive_components(pred, gt),
    }
    scores.update(compute_2d_slice_metrics(pred, gt))

    found = compute_lesion_metrics(pred, gt, min_lesion_voxels=min_lesion_voxels)
    scores["n_lesions"] = len(found)
    scores["dice_3d_lesion"] = (float(np.mean([l["dice"] for l in found]))
                                if found else float("nan"))
    return scores, found


def score_run(run_dir, case_ids, min_lesion_voxels=0, postproc_min_fraction=None):
    """
    Scores one checkpoint over the test split, raw and post-processed.

    Post-processing is the same component filter the training pipeline reports
    under its `pp_` keys: connected components below a fraction of the largest
    are dropped. It is applied to the reconstructed volume, so every granularity
    sees the same cleaned prediction and the two sets stay comparable.

    Returns:
        tuple: (per-patient rows, per-lesion rows, post-processed per-patient
                rows, post-processed per-lesion rows)
    """
    model, device, config, threshold = load_run(run_dir)
    if postproc_min_fraction is None:
        postproc_min_fraction = float(config.get("postproc_min_fraction", 0.10))

    patients, lesions = [], []
    pp_patients, pp_lesions = [], []

    for case_id in case_ids:
        probs = predict_patient(model, device, config, case_id)

        with open(os.path.join(OUTPUT_DIR, "metadata", f"{case_id}.json")) as f:
            metadata = json.load(f)
        pred = reconstruct_patient_3d_volume(probs, metadata, threshold=threshold,
                                             binarize=True)
        gt = (np.asanyarray(nib.load(
            resolve_nifti_path(f"./labelsTr/{case_id}.nii.gz")).dataobj)
            > 0.5).astype(np.uint8)

        scores, found = score_prediction(pred, gt, min_lesion_voxels)
        patients.append({"case_id": case_id, **scores})
        for lesion in found:
            lesions.append({"case_id": case_id, **lesion})

        cleaned, removed = filter_predicted_components(
            pred, min_fraction=postproc_min_fraction)
        pp_scores, pp_found = score_prediction(cleaned, gt, min_lesion_voxels)
        pp_patients.append({"case_id": case_id, "components_removed": removed,
                            **pp_scores})
        for lesion in pp_found:
            pp_lesions.append({"case_id": case_id, **lesion})

    return patients, lesions, pp_patients, pp_lesions


def summarise(patients, lesions):
    """Averages the per-patient rows and the per-lesion rows into one summary."""
    def mean(rows, key):
        values = [r[key] for r in rows if not np.isnan(r[key])]
        return float(np.mean(values)) if values else float("nan")

    summary = {
        "dice_2d_tumour_slices": mean(patients, "dice_2d_tumour_slices"),
        "dice_2d_all_slices": mean(patients, "dice_2d_all_slices"),
        "dice_3d_patient": mean(patients, "dice_3d_patient"),
        "dice_3d_lesion": (float(np.mean([l["dice"] for l in lesions]))
                           if lesions else float("nan")),
        "sensitivity_3d_patient": mean(patients, "sensitivity_3d_patient"),
        "precision_3d_patient": mean(patients, "precision_3d_patient"),
        "false_alarm_rate_2d": mean(patients, "false_alarm_rate_2d"),
        "failure_rate_2d": mean(patients, "failure_rate_2d"),
        "fp_components": float(np.mean([r["fp_components"] for r in patients])),
        "n_lesions": len(lesions),
        "lesions_missed": sum(l["is_missed"] for l in lesions),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Score existing checkpoints per slice, per lesion and per patient.")
    parser.add_argument("--runs", default=",".join(DEFAULT_RUNS),
                        help="Comma-separated experiment names under "
                             "output/experiments/, or 'all' for every "
                             "checkpoint found. 'all' puts the four base "
                             "architectures first, so the comparison numbers "
                             "land before the long tail.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_lesion_voxels", type=int, default=0,
                        help="Ignore ground-truth components below this size. "
                             "Zero keeps the annotation fragments, which is "
                             "what shows why the per-lesion view does not "
                             "transfer to this dataset.")
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR,
                                                      "hierarchical_report.json"))
    args = parser.parse_args()

    with open(os.path.join(OUTPUT_DIR, "preprocessed", "index.json")) as f:
        case_ids = sorted(json.load(f)["splits"]["test"])

    if args.runs.strip() == "all":
        experiments = os.path.join(OUTPUT_DIR, "experiments")
        found = sorted(name for name in os.listdir(experiments)
                       if os.path.exists(os.path.join(
                           experiments, name, f"seed_{args.seed}", "best_model.pt")))
        run_names = ([n for n in DEFAULT_RUNS if n in found]
                     + [n for n in found if n not in DEFAULT_RUNS])
    else:
        run_names = [r.strip() for r in args.runs.split(",") if r.strip()]

    print(f"=== HIERARCHICAL EVALUATION: {len(case_ids)} test patients ===\n")
    print(f"  Runs: {len(run_names)}")
    print(f"  Lesion size filter: {args.min_lesion_voxels} voxels\n")

    results = {}
    for name in run_names:
        run_dir = os.path.join(OUTPUT_DIR, "experiments", name, f"seed_{args.seed}")
        if not os.path.exists(os.path.join(run_dir, "best_model.pt")):
            print(f"  [SKIP] {name}: no checkpoint at {run_dir}")
            continue

        started = time.time()
        patients, lesions, pp_patients, pp_lesions = score_run(
            run_dir, case_ids, min_lesion_voxels=args.min_lesion_voxels)
        summary = summarise(patients, lesions)
        pp_summary = summarise(pp_patients, pp_lesions)
        results[name] = {"summary": summary, "postprocessed_summary": pp_summary,
                         "per_patient": patients, "per_lesion": lesions,
                         "postprocessed_per_patient": pp_patients,
                         "postprocessed_per_lesion": pp_lesions}

        for tag, block in (("raw", summary), ("pp ", pp_summary)):
            print(f"  {name:18} {tag}  2D tumour {block['dice_2d_tumour_slices']:.4f} | "
                  f"2D all {block['dice_2d_all_slices']:.4f} | "
                  f"3D lesion {block['dice_3d_lesion']:.4f} | "
                  f"3D patient {block['dice_3d_patient']:.4f}", flush=True)
        print(f"  {'':18} ({time.time() - started:.0f}s)", flush=True)

        # Written after every run rather than at the end. Scoring the full set of
        # checkpoints takes hours, and a failure on the last one should not throw
        # away the ones already measured.
        write_results(results, args.out)

    print(f"\n  {args.out}")
    print(f"  {args.out.replace('.json', '.csv')}")


def write_results(results, out_path):
    """Writes the full JSON and a flat CSV, one row per run and post-processing state."""
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    if not results:
        return

    first = next(iter(results.values()))["summary"]
    with open(out_path.replace(".json", ".csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "postprocessed"] + list(first))
        writer.writeheader()
        for name, payload in results.items():
            writer.writerow({"run": name, "postprocessed": 0, **payload["summary"]})
            writer.writerow({"run": name, "postprocessed": 1,
                             **payload["postprocessed_summary"]})


if __name__ == "__main__":
    main()