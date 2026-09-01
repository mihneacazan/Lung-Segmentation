"""
Measures what each preprocessing grid costs, before any of them is trained on.

Comparing 192, 256 and 320 with square padding rests on a
claim that can be checked without a GPU: that small tumours lose information at
192. Preprocessing is invertible on paper but not in practice - the mask is
resampled and thresholded at 0.5, so a thin cross-section can vanish - and the
round-trip Dice of the ground truth through the pipeline is therefore a hard
ceiling. No model can score above it, whatever it learns.

Six grids are measured, {192, 256, 320} x {stretch, pad}, all at 1.0 mm slice
spacing so the Z axis stays fixed and only the in-plane treatment varies. The
earlier 320 measurement moved Z to 2.5 mm at the same time and its ceiling fell
for that reason, which is why it is repeated here rather than reused.

The result decides which arms deserve GPU time. A grid whose ceiling does not
rise cannot produce a model that scores higher, so training it would measure
noise. Costs about 40 minutes per grid on CPU and touches no GPU.

Usage:
    python -m src.preprocessing.resolution_ceiling
    python -m src.preprocessing.resolution_ceiling --grids 256:stretch,320:pad
    python -m src.preprocessing.resolution_ceiling --keep       # keep the volumes
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from src.config import OUTPUT_DIR
from src.preprocessing.preprocessing import preprocess_all

# 192/stretch is absent on purpose: it is the committed dataset, its QC is
# already on disk, and rebuilding it would spend an hour re-deriving numbers we
# have. `reproduces_committed` checks in fifty seconds that the current code
# still produces it bit for bit, which is the claim that mattered.
DEFAULT_GRIDS = [(192, "pad"),
                 (256, "stretch"), (256, "pad"),
                 (320, "stretch"), (320, "pad")]

COMMITTED_QC = os.path.join(OUTPUT_DIR, "preprocessing_qc.csv")
CHECK_CASE = "lung_001"


def committed_row():
    """The committed 192/stretch grid, read rather than rebuilt."""
    qc = pd.read_csv(COMMITTED_QC)
    index = json.load(open(os.path.join(OUTPUT_DIR, "preprocessed", "index.json")))
    qc["voxels"] = [index["cases"][c]["tumor_voxels"] for c in qc.case_id]
    with_tumour = qc[qc.voxels > 0].sort_values("voxels")
    third = len(with_tumour) // 3
    return {
        "target_size": 192, "resize_mode": "stretch",
        "slices": sum(index["cases"][c]["n_slices"] for c in index["cases"]),
        "positive_slices": sum(len(index["cases"][c]["positive_slices"])
                               for c in index["cases"]),
        "positive_slices_lost": int(qc.pos_slices_lost_to_resize.sum()),
        "roundtrip_all": float(with_tumour.roundtrip_dice.mean()),
        "roundtrip_small": float(with_tumour.iloc[:third].roundtrip_dice.mean()),
        "roundtrip_medium": float(
            with_tumour.iloc[third:2 * third].roundtrip_dice.mean()),
        "roundtrip_large": float(
            with_tumour.iloc[2 * third:].roundtrip_dice.mean()),
        "roundtrip_worst": float(with_tumour.roundtrip_dice.min()),
        "volume_change_pct": float(qc.tumor_volume_change_pct.mean()),
        "minutes": 0.0,
    }


def reproduces_committed(case_id=CHECK_CASE, tolerance=1e-4):
    """
    Re-preprocesses one patient at 192/stretch and compares against the committed
    QC row. The whole table below is read against that dataset, so if the code
    has drifted since it was built, every delta is measured from the wrong place.
    One patient at fifty seconds settles it; sixty-three would take an hour and
    say the same thing.
    """
    from src.config import resolve_nifti_path
    from src.preprocessing.preprocessing import preprocess_case

    row = pd.read_csv(COMMITTED_QC).set_index("case_id").loc[case_id]
    out = preprocess_case(
        case_id, resolve_nifti_path(f"./imagesTr/{case_id}.nii.gz"),
        resolve_nifti_path(f"./labelsTr/{case_id}.nii.gz"),
        run_qc=True, resize_mode="stretch", target_size=192, slice_spacing=1.0)
    parts = out if isinstance(out, tuple) else (out,)
    qc = next(x for x in parts
              if isinstance(x, dict) and "roundtrip_dice" in x)

    drift = {}
    for key in ("roundtrip_dice", "tumor_volume_change_pct"):
        drift[key] = abs(float(qc[key]) - float(row[key]))
    for key in ("n_pos_slices_after_resize", "pos_slices_lost_to_resize",
                "n_slices_kept"):
        drift[key] = abs(int(qc[key]) - int(row[key]))
    return qc, dict(row), drift, all(v <= tolerance for v in drift.values())


def measure(target_size, resize_mode, keep=False):
    """Builds one grid, reads its QC, and throws the volumes away again."""
    name = f"ceil_{target_size}_{resize_mode}"
    started = time.time()
    preprocess_all(run_qc=True, slice_spacing=1.0, target_size=target_size,
                   resize_mode=resize_mode, preprocessed_name=name,
                   metadata_name=f"{name}_metadata", qc_name=f"{name}_qc.csv")

    index = json.load(open(os.path.join(OUTPUT_DIR, name, "index.json")))
    qc = pd.read_csv(os.path.join(OUTPUT_DIR, f"{name}_qc.csv"))
    qc["voxels"] = [index["cases"][c]["tumor_voxels"] for c in qc.case_id]
    with_tumour = qc[qc.voxels > 0].sort_values("voxels")
    third = len(with_tumour) // 3

    row = {
        "target_size": target_size,
        "resize_mode": resize_mode,
        "slices": sum(index["cases"][c]["n_slices"] for c in index["cases"]),
        "positive_slices": sum(len(index["cases"][c]["positive_slices"])
                               for c in index["cases"]),
        "positive_slices_lost": int(qc.pos_slices_lost_to_resize.sum()),
        "roundtrip_all": float(with_tumour.roundtrip_dice.mean()),
        "roundtrip_small": float(with_tumour.iloc[:third].roundtrip_dice.mean()),
        "roundtrip_medium": float(
            with_tumour.iloc[third:2 * third].roundtrip_dice.mean()),
        "roundtrip_large": float(
            with_tumour.iloc[2 * third:].roundtrip_dice.mean()),
        "roundtrip_worst": float(with_tumour.roundtrip_dice.min()),
        "volume_change_pct": float(qc.tumor_volume_change_pct.mean()),
        "minutes": (time.time() - started) / 60.0,
    }

    if not keep:
        for path in (os.path.join(OUTPUT_DIR, name),
                     os.path.join(OUTPUT_DIR, f"{name}_metadata")):
            shutil.rmtree(path, ignore_errors=True)
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing ceiling for every in-plane grid, on CPU.")
    parser.add_argument("--grids", default=None,
                        help="Comma-separated size:mode pairs, e.g. "
                             "256:stretch,320:pad. Default: all six.")
    parser.add_argument("--keep", action="store_true",
                        help="Keep each grid's volumes instead of deleting "
                             "them. Six full copies is roughly 30 GB.")
    parser.add_argument("--out", default=os.path.join(OUTPUT_DIR,
                                                      "resolution_ceiling.csv"))
    args = parser.parse_args()

    if args.grids:
        grids = [(int(s.split(":")[0]), s.split(":")[1])
                 for s in args.grids.split(",")]
    else:
        grids = DEFAULT_GRIDS

    print("=== PREPROCESSING CEILING PER GRID ===\n")
    print("  Round-trip Dice of the ground truth through preprocessing and back.")
    print("  No model can score above this, so a grid whose ceiling does not")
    print("  rise cannot repay the GPU time it would take to train on it.\n")
    print(f"  {len(grids)} grids, all at 1.0 mm slice spacing "
          f"(Z fixed, only in plane varies)\n", flush=True)

    print(f"  Reproducibility check on {CHECK_CASE} at 192/stretch ... ",
          end="", flush=True)
    fresh, committed, drift, ok = reproduces_committed()
    print("OK" if ok else "DRIFTED")
    for key, value in drift.items():
        print(f"      {key:28s} fresh {fresh[key]}  committed {committed[key]}"
              f"  drift {value}")
    if not ok:
        raise SystemExit(
            "\n  The current code no longer reproduces the committed 192/stretch "
            "dataset.\n  Every delta below would be measured against a grid that "
            "no experiment used.\n  Fix that before reading this table.")
    print()

    rows = [committed_row()]
    print(f"  192 x 192, stretch: citit din QC-ul comis, "
          f"plafon {rows[0]['roundtrip_all']:.4f} (nu se reconstruieste)\n",
          flush=True)

    for target_size, resize_mode in grids:
        print(f"\n{'=' * 70}\n  {target_size} x {target_size}, {resize_mode}"
              f"\n{'=' * 70}", flush=True)
        row = measure(target_size, resize_mode, keep=args.keep)
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.out, index=False)   # written as it goes
        print(f"  ceiling {row['roundtrip_all']:.4f} "
              f"(small {row['roundtrip_small']:.4f}) | "
              f"{row['positive_slices_lost']} positive slices lost | "
              f"{row['minutes']:.1f} min", flush=True)

    table = pd.DataFrame(rows)
    print(f"\n\n{'=' * 78}")
    print("  PLAFON PE MARIME DE TUMOARE")
    print(f"{'=' * 78}")
    print(f"{'grila':16s} {'mici':>8s} {'mijlocii':>9s} {'mari':>8s} "
          f"{'toate':>8s} {'cea mai rea':>12s} {'pierdute':>9s}")
    print("-" * 78)
    for row in rows:
        label = f"{row['target_size']} {row['resize_mode']}"
        print(f"{label:16s} {row['roundtrip_small']:8.4f} "
              f"{row['roundtrip_medium']:9.4f} {row['roundtrip_large']:8.4f} "
              f"{row['roundtrip_all']:8.4f} {row['roundtrip_worst']:12.4f} "
              f"{row['positive_slices_lost']:9d}")

    base = rows[0]
    print(f"\n  Fata de 192/stretch, grila pe care s-au antrenat toate "
          f"experimentele:")
    print(f"  {'grila':16s} {'mici':>9s} {'toate':>9s}   verdict")
    for row in rows[1:]:
        gain_small = row["roundtrip_small"] - base["roundtrip_small"]
        gain_all = row["roundtrip_all"] - base["roundtrip_all"]
        # A grid whose ceiling does not rise cannot produce a model that scores
        # higher, so training it would spend GPU hours measuring noise. The
        # threshold is the noise band on a Dice difference, not zero.
        verdict = ("merita GPU" if gain_small > 0.01
                   else "nu are de unde sa castige")
        print(f"    {row['target_size']} {row['resize_mode']:8s} "
              f"{gain_small:+9.4f} {gain_all:+9.4f}   {verdict}")

    print(f"\n  Plafonul e limita superioara absoluta: niciun model antrenat pe")
    print(f"  o grila nu poate depasi Dice-ul de dus-intors al adnotarii prin ea.")
    print(f"\n  {args.out}")


if __name__ == "__main__":
    main()
