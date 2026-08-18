"""
End-to-end smoke test for the training pipeline.

Runs a complete experiment — training loop, early stopping, checkpointing,
validation threshold sweep, 3D reconstruction into original NIfTI geometry, and
the final per-patient report — on a tiny synthetic dataset, on CPU, in a few
seconds.

The point is to find out that an experiment is broken before spending GPU hours
on it. The expensive failures in a pipeline like this one are not crashes during
training — they are stages that only execute after training finishes, such as the
threshold sweep and the 3D reconstruction, and configurations that fail at
argument parsing and therefore never start. Both are reachable here, on CPU, in
seconds.

Run:
    python -m pytest tests/test_training_smoke.py -q
"""

import json

import numpy as np
import pytest
import nibabel as nib
import torch
from torch.amp import GradScaler

import src.config
import src.training.train as train_mod


SHAPE = (32, 32, 10)          # small enough to train in seconds
SPLITS = {"train": ["c_tr1", "c_tr2"], "val": ["c_va1"], "test": ["c_te1"]}
POSITIVE = [3, 4, 5]


@pytest.fixture
def tiny_project(tmp_path, monkeypatch):
    """
    Builds a complete miniature project on disk: preprocessed volumes, an index,
    reconstruction metadata, and matching ground-truth NIfTI files.

    Returns:
        pathlib.Path: the fake OUTPUT_DIR.
    """
    output_dir = tmp_path / "output"
    data_dir = tmp_path / "archive"
    volumes = output_dir / "preprocessed" / "volumes"
    metadata_dir = output_dir / "metadata"
    labels_dir = data_dir / "labelsTr"

    for d in (volumes, metadata_dir, labels_dir):
        d.mkdir(parents=True, exist_ok=True)

    h, w, d_slices = SHAPE
    cases = {}

    # LAS, exactly like the real dataset: axis 0 is flipped on the way to RAS.
    affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    ornt = [[0, -1], [1, 1], [2, 1]]

    for split, case_ids in SPLITS.items():
        for case_id in case_ids:
            rng = np.random.default_rng(abs(hash(case_id)) % 2**32)
            img = rng.uniform(0.1, 0.3, SHAPE).astype(np.float16)
            lbl = np.zeros(SHAPE, dtype=np.uint8)
            for s in POSITIVE:
                lbl[10:20, 12:22, s] = 1
                img[10:20, 12:22, s] = 0.95      # learnable: tumour is bright

            np.save(volumes / f"{case_id}_img.npy", img)
            np.save(volumes / f"{case_id}_lbl.npy", lbl)

            # The ground truth on disk is in original (LAS) orientation, so it is
            # the mirror of the preprocessed stack. Reconstruction has to undo
            # that flip for these to line up.
            gt_original = np.flip(lbl, axis=0).copy()
            nib.save(nib.Nifti1Image(gt_original, affine),
                     str(labels_dir / f"{case_id}.nii.gz"))

            (metadata_dir / f"{case_id}.json").write_text(json.dumps({
                "case_id": case_id,
                "original_affine": affine.tolist(),
                "original_shape": list(SHAPE),
                "original_spacing": [1.0, 1.0, 1.0],
                "ornt": ornt,
                "canonical_shape": list(SHAPE),
                "canonical_spacing": [1.0, 1.0, 1.0],
                "target_spacing": [1.0, 1.0, 1.0],
                "resampled_shape": list(SHAPE),
                "crop_bbox": {"x_min": 0, "x_max": h, "y_min": 0, "y_max": w,
                              "z_min": 0, "z_max": d_slices},
                "cropped_shape": list(SHAPE),
                "target_slice_size": [h, w],
                "hu_min": -1000.0, "hu_max": 400.0,
            }))

            cases[case_id] = {
                "split": split,
                "n_slices": d_slices,
                "positive_slices": list(POSITIVE),
                "body_slices": list(range(d_slices)),
                "tumor_voxels": int(lbl.sum()),
            }

    (output_dir / "preprocessed" / "index.json").write_text(
        json.dumps({"splits": SPLITS, "cases": cases}))

    monkeypatch.setattr(src.config, "DATA_DIR", str(data_dir))
    monkeypatch.setattr(train_mod, "OUTPUT_DIR", str(output_dir))

    return output_dir


def run(tiny_project, **overrides):
    """Runs a short experiment with sensible test defaults."""
    kwargs = dict(
        epochs=2, batch_size=4, lr=1e-3, model_type="unet", loss_type="dice_ce",
        augment="none", sampling="balanced", n_adjacent=1, seed=42,
        patience=10, min_epochs=1, exp_name="smoke", num_workers=0,
    )
    kwargs.update(overrides)
    return train_mod.run_training_experiment(**kwargs)


# ============================================================================
#  THE PIPELINE COMPLETES
# ============================================================================

def test_experiment_runs_end_to_end(tiny_project):
    """
    The whole run must complete: training, threshold sweep, reconstruction, and
    reporting. The stages after training are the ones that go unexercised, because
    reaching them on real data costs an hour of GPU first.
    """
    report = run(tiny_project)

    assert report["epochs_run"] == 2
    assert 0.0 <= report["optimal_threshold"] <= 1.0
    assert report["parameters_trainable"] > 0

    summary = report["test_metrics_summary"]
    for key in ("mean_dice_3d", "mean_iou_3d", "mean_hd95_mm", "mean_asd_mm",
                "mean_sensitivity", "mean_precision", "mean_specificity",
                "mean_fp_components", "failure_rate_percent"):
        assert key in summary, f"missing required metric: {key}"

    assert "macro" in report["macro_average"] or "dice_3d" in report["macro_average"]
    assert set(report["stratified_report"]) >= {"small", "medium", "large"}

    exp_dir = tiny_project / "experiments" / "smoke" / "seed_42"
    for artifact in ("config.json", "best_model.pt", "checkpoint.pt",
                     "training_history.csv", "threshold_sweep.json",
                     "test_results_per_patient.csv", "benchmark_report.json"):
        assert (exp_dir / artifact).exists(), f"missing artifact: {artifact}"


def test_threshold_sweep_produces_a_real_curve(tiny_project):
    """
    The sweep must evaluate several thresholds and produce different scores for
    them. An identical score at every threshold is the signature of a sweep that
    is comparing predictions against themselves rather than against ground
    truth — which still runs, still prints, and still picks a "best" value.
    """
    report = run(tiny_project)
    sweep = report["threshold_sweep"]
    assert len(sweep) >= 10
    assert len(set(round(v, 6) for v in sweep.values())) > 1


def test_history_records_both_dice_variants(tiny_project):
    """Both the soft signal and the reported hard Dice are logged per epoch."""
    import csv
    run(tiny_project)
    path = tiny_project / "experiments" / "smoke" / "seed_42" / "training_history.csv"
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 2
    for row in rows:
        assert "val_dice_soft" in row and "val_dice_hard" in row
        assert 0.0 <= float(row["val_dice_soft"]) <= 1.0


# ============================================================================
#  EVERY CONFIGURATION THE CLI OFFERS MUST ACTUALLY RUN
# ============================================================================

@pytest.mark.parametrize("model_type", ["unet", "attention_unet", "segresnet"])
def test_every_architecture_trains(tiny_project, model_type):
    """
    All three architectures must run. `--model_type` did not exist, so the
    attention U-Net and SegResNet experiments died at argument parsing and those
    two model files were never executed at all.
    """
    report = run(tiny_project, model_type=model_type, exp_name=f"smoke_{model_type}")
    assert report["parameters_trainable"] > 0
    assert report["config"]["model_type"] == model_type


@pytest.mark.parametrize("loss_type", ["dice_ce", "dice_focal", "tversky",
                                       "focal_tversky"])
def test_every_loss_trains(tiny_project, loss_type):
    """
    All four losses must run. `focal_tversky` passed a `gamma` argument that
    MONAI's TverskyLoss does not accept, so that experiment raised TypeError
    before the first epoch.
    """
    report = run(tiny_project, loss_type=loss_type, exp_name=f"smoke_{loss_type}")
    assert np.isfinite(report["best_val_dice_soft"])


@pytest.mark.parametrize("sampling", ["balanced", "all", "hard_negatives"])
def test_every_sampling_mode_trains(tiny_project, sampling):
    report = run(tiny_project, sampling=sampling, exp_name=f"smoke_{sampling}")
    assert report["config"]["sampling"] == sampling


@pytest.mark.parametrize("n_adjacent", [1, 3])
def test_2d_and_25d_both_train(tiny_project, n_adjacent):
    """A 2.5D model must accept its stacked input all the way through."""
    report = run(tiny_project, n_adjacent=n_adjacent, exp_name=f"smoke_{n_adjacent}")
    assert report["config"]["n_adjacent"] == n_adjacent


# ============================================================================
#  NUMERICAL STABILITY
# ============================================================================

class _NanLoss(torch.nn.Module):
    """A loss that always returns NaN, standing in for a diverged run."""

    def forward(self, pred, target):
        return pred.sum() * float("nan")


class _HugeLoss(torch.nn.Module):
    """A loss whose gradient is large enough that clipping has to bite."""

    def forward(self, pred, target):
        return pred.sum() * 1e4


def _one_batch():
    return [{"image": torch.ones(2, 1, 4, 4), "label": torch.zeros(2, 1, 4, 4)}]


def test_gradient_clipping_shrinks_an_exploding_step():
    """
    Two runs from an identical initialisation, one clipped and one not. With
    lr=1.0 and the norm clipped to 1.0, no weight may move by more than 1.0.

    Nothing bounded the step size before, and two runs in the first benchmark
    diverged to a non-finite loss and never recovered.
    """
    def fresh():
        torch.manual_seed(0)
        model = torch.nn.Conv2d(1, 1, 1)
        return model, torch.optim.SGD(model.parameters(), lr=1.0)

    device = torch.device("cpu")

    model_a, opt_a = fresh()
    before = model_a.weight.detach().clone()
    train_mod.train_one_epoch(model_a, _one_batch(), _HugeLoss(), opt_a, device,
                              None, max_grad_norm=0)
    unclipped = (model_a.weight.detach() - before).abs().max().item()

    model_b, opt_b = fresh()
    train_mod.train_one_epoch(model_b, _one_batch(), _HugeLoss(), opt_b, device,
                              None, max_grad_norm=1.0)
    clipped = (model_b.weight.detach() - before).abs().max().item()

    assert clipped < unclipped, "clipping did not reduce the step"
    assert clipped <= 1.0 + 1e-6, f"clipped step {clipped} exceeds the norm bound"


def test_nonfinite_batches_are_counted_and_kept_out_of_the_mean():
    """
    A NaN batch must not turn the reported epoch loss into NaN, and must be
    counted so the caller can tell a diverged epoch from a healthy one.
    """
    model = torch.nn.Conv2d(1, 1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = _one_batch() * 3

    loss, n_nonfinite, n_batches = train_mod.train_one_epoch(
        model, batches, _NanLoss(), optimizer, torch.device("cpu"), None)

    assert (n_nonfinite, n_batches) == (3, 3)
    assert np.isfinite(loss), "a NaN batch leaked into the reported loss"


def test_diverged_run_stops_instead_of_burning_the_patience_budget(
        tiny_project, monkeypatch):
    """
    Once every batch is non-finite the optimizer is not stepping, so further
    epochs cannot change anything. The first benchmark spent 9 and 10 such
    epochs before early stopping noticed. Training must stop at the first one.
    """
    monkeypatch.setattr(train_mod, "build_loss_function",
                        lambda *_args, **_kwargs: _NanLoss())
    report = run(tiny_project, epochs=10, min_epochs=1, exp_name="smoke_diverged")
    assert report["epochs_run"] == 1, (
        f"ran {report['epochs_run']} epochs after full divergence")


# ============================================================================
#  THE LOSS IS EVALUATED IN FULL PRECISION
# ============================================================================

class _HalfModel(torch.nn.Module):
    """Emits fp16 activations, as any network does inside autocast."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv(x).half()


class _DtypeProbe(torch.nn.Module):
    """Records the dtype of the logits it is handed."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, pred, target):
        self.seen.append(pred.dtype)
        return pred.pow(2).mean()


def test_fp16_flushes_a_confident_background_to_exactly_zero():
    """
    Why the loss has to leave autocast. fp16's smallest subnormal is about
    6e-8, so a background logit past roughly -17 makes every sigmoid output
    round to exactly zero and the summed probability with it. The Dice
    denominator then degenerates, and the loss falls off a cliff for reasons
    that have nothing to do with segmentation quality.
    """
    logits = torch.full((1, 1, 32, 32), -20.0)
    assert torch.sigmoid(logits.half()).sum().item() == 0.0
    assert torch.sigmoid(logits).sum().item() > 0.0


def test_criterion_receives_fp32_logits_under_amp():
    """
    The model may run in half precision — that is where the speed is — but the
    criterion must not.
    """
    model = _HalfModel()
    probe = _DtypeProbe()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_mod.train_one_epoch(model, _one_batch(), probe, optimizer,
                              torch.device("cpu"), GradScaler("cuda"),
                              max_grad_norm=1.0)

    assert probe.seen == [torch.float32], (
        f"criterion was handed {probe.seen}, so the loss is still under autocast")


# ============================================================================
#  THE LOSS MUST NOT PAY FOR SILENCE
# ============================================================================

def test_an_empty_target_offers_no_reward_for_going_quiet():
    """
    On a slice whose target is empty the intersection is identically zero, so
    the Dice term reduces to `smooth_nr / (|P| + smooth_dr)` — an expression the
    model raises simply by shrinking its own output. With the MONAI default that
    dial is worth almost a full unit of loss — enough for a network to abandon
    segmentation entirely and collect it. The configured loss must be flat
    instead.
    """
    from monai.losses import DiceLoss
    from src.training.losses import SMOOTH_DR, SMOOTH_NR

    target = torch.zeros(1, 1, 32, 32)
    logits = (-2.0, -6.0, -12.0, -20.0, -30.0)

    tuned = DiceLoss(sigmoid=True, smooth_nr=SMOOTH_NR, smooth_dr=SMOOTH_DR)
    ours = [tuned(torch.full_like(target, L), target).item() for L in logits]
    assert max(ours) - min(ours) < 1e-6, (
        f"loss still varies with prediction magnitude on an empty target: {ours}")

    # The same sweep under the default, to pin down what was actually wrong.
    stock = DiceLoss(sigmoid=True)
    theirs = [stock(torch.full_like(target, L), target).item() for L in logits]
    assert max(theirs) - min(theirs) > 0.5, (
        "the MONAI default no longer rewards silence; this guard is obsolete")


def test_real_lesions_are_scored_exactly_as_before():
    """
    Removing the numerator constant must be invisible wherever there is a real
    lesion: 2 * |P and G| counts hundreds of pixels, against which 1e-5 is noise.
    Without this, the fix would be trading one bias for another.
    """
    from monai.losses import DiceLoss
    from src.training.losses import SMOOTH_DR, SMOOTH_NR

    target = torch.zeros(1, 1, 32, 32)
    target[0, 0, 8:16, 8:16] = 1
    pred = torch.full_like(target, -6.0) + target * 12.0

    tuned = DiceLoss(sigmoid=True, smooth_nr=SMOOTH_NR, smooth_dr=SMOOTH_DR)
    stock = DiceLoss(sigmoid=True)
    assert abs(tuned(pred, target).item() - stock(pred, target).item()) < 1e-4


@pytest.mark.parametrize("loss_type", ["dice_ce", "dice_focal", "tversky",
                                       "focal_tversky"])
def test_silence_never_scores_better_than_segmentation(loss_type):
    """
    The property the whole fix exists to guarantee, checked on every loss the
    CLI offers: a network that has switched off entirely must never be preferred
    to one that is imperfectly finding the lesions.

    The batch mirrors `--sampling all` — mostly empty slices, a couple carrying
    a tumour — because that mix is what made the degenerate solution profitable.
    """
    from src.training.losses import build_loss_function

    target = torch.zeros(16, 1, 32, 32)
    target[0, 0, 12:18, 12:18] = 1
    target[1, 0, 10:20, 10:20] = 1

    learning = torch.full_like(target, -4.0) + target * 10.0
    collapsed = torch.full_like(target, -25.0)

    criterion = build_loss_function(loss_type)
    learning_loss = criterion(learning, target).item()
    collapsed_loss = criterion(collapsed, target).item()

    assert collapsed_loss > learning_loss, (
        f"{loss_type} pays {collapsed_loss:.4f} for silence against "
        f"{learning_loss:.4f} for segmenting")


def test_tversky_reduces_to_dice_at_equal_weights():
    """
    The Tversky index with alpha = beta = 0.5 is Dice, by construction. This
    bounds the tuning range from below: the published 0.3/0.7 over-segments by
    roughly eightfold on this target, and anything at or past 0.5/0.5 has stopped
    being a Tversky experiment at all.
    """
    from monai.losses import DiceLoss

    from src.training.losses import SMOOTH_DR, SMOOTH_NR, build_loss_function

    target = torch.zeros(1, 1, 32, 32)
    target[0, 0, 8:16, 8:16] = 1
    pred = torch.full_like(target, -3.0) + target * 5.0

    tversky = build_loss_function("tversky", tversky_alpha=0.5, tversky_beta=0.5)
    dice = DiceLoss(sigmoid=True, smooth_nr=SMOOTH_NR, smooth_dr=SMOOTH_DR)
    assert abs(tversky(pred, target).item() - dice(pred, target).item()) < 1e-6


def test_tversky_weights_actually_change_the_penalty():
    """
    A guard against the weights being accepted and then ignored: raising beta must
    make a miss more expensive relative to a false alarm.
    """
    from src.training.losses import build_loss_function

    target = torch.zeros(1, 1, 32, 32)
    target[0, 0, 8:16, 8:16] = 1
    missing = torch.full_like(target, -6.0)                       # sees nothing
    spurious = torch.full_like(target, -6.0) + target * 12.0
    spurious[0, 0, 20:28, 20:28] = 6.0                            # marks the lesion, plus a blob

    for loss_type in ("tversky", "focal_tversky"):
        timid = build_loss_function(loss_type, tversky_alpha=0.4, tversky_beta=0.6)
        eager = build_loss_function(loss_type, tversky_alpha=0.3, tversky_beta=0.7)

        assert eager(missing, target).item() > timid(missing, target).item(), (
            f"{loss_type}: raising beta did not make a miss more expensive")
        assert eager(spurious, target).item() < timid(spurious, target).item(), (
            f"{loss_type}: raising beta did not make a false alarm cheaper")


def test_tversky_weights_reach_the_experiment_and_are_recorded(tiny_project):
    """
    The weights have to survive the whole path — CLI to config to criterion — or
    a run labelled beta=0.6 would silently train at the default and the report
    would be wrong about what was measured.
    """
    report = run(tiny_project, loss_type="tversky", tversky_alpha=0.4,
                 tversky_beta=0.6, exp_name="smoke_tversky_weights")
    assert report["config"]["tversky_alpha"] == 0.4
    assert report["config"]["tversky_beta"] == 0.6


# ============================================================================
#  COLLAPSE TO AN ALL-BACKGROUND PREDICTION
# ============================================================================

def test_collapse_detector_waits_for_the_model_to_have_learned_something():
    """
    Validation Dice is legitimately zero while a network is still warming up on
    a target occupying well under 1% of the volume. Stopping there would kill
    every healthy run, so the detector must stay disarmed until Dice has been
    above the floor at least once.
    """
    detector = train_mod.CollapseDetector(min_peak=0.10, patience=3)
    for _ in range(20):
        assert not detector.step(0.0), "fired during warm-up"


def test_collapse_detector_fires_on_the_observed_failure_pattern():
    """
    The observed collapse signature: seventeen epochs climbing to around 0.39,
    then an abrupt and permanent 0.0000. Three zero epochs are enough to conclude,
    since the saturated sigmoid leaves no gradient on the empty slices and
    therefore no route back.
    """
    detector = train_mod.CollapseDetector(min_peak=0.10, patience=3)
    climb = [0.00, 0.07, 0.30, 0.31, 0.26, 0.38, 0.30, 0.38, 0.32, 0.35,
             0.29, 0.39, 0.33, 0.34, 0.39, 0.37, 0.28]
    for score in climb:
        assert not detector.step(score), "fired while the model was learning"

    fired = [detector.step(0.0) for _ in range(3)]
    assert fired == [False, False, True], (
        f"expected a stop on the third zero epoch, got {fired}")


def test_collapse_detector_forgives_a_single_zero_epoch():
    """
    One zero reading is noise, not a collapse; the counter has to reset when the
    model comes back.
    """
    detector = train_mod.CollapseDetector(min_peak=0.10, patience=3)
    detector.step(0.40)
    assert not detector.step(0.0)
    assert not detector.step(0.0)
    assert not detector.step(0.35), "a recovered epoch was treated as collapsed"
    assert not detector.step(0.0), "the counter did not reset on recovery"


def test_collapsed_run_stops_early_instead_of_training_dead_weights(
        tiny_project, monkeypatch):
    """
    End to end: once the detector fires the run must stop, well short of both the
    epoch budget and the patience budget. Without it a collapsed run keeps
    training for the full patience window on weights that can no longer change.
    """
    scores = iter([0.4] + [0.0] * 30)

    def fake_eval(*_args, **_kwargs):
        return {"case": {"dice_soft": next(scores), "dice_hard": 0.0}}

    monkeypatch.setattr(train_mod, "evaluate_fast", fake_eval)
    report = run(tiny_project, epochs=25, patience=20, min_epochs=1,
                 exp_name="smoke_collapsed")

    assert report["epochs_run"] == 4, (
        f"ran {report['epochs_run']} epochs; expected a stop on the third zero")


def test_lr_schedule_period_is_independent_of_the_epoch_budget(tiny_project):
    """
    Tying `T_max` to `--epochs` means raising the budget stretches the cosine
    decay instead of extending training, so a longer run can score *worse* than a
    shorter one: it early-stops before the learning rate has annealed at all.
    `--lr_t_max` has to pin the schedule independently of the budget.
    """
    import csv

    def lr_after_first_epoch(t_max, name):
        run(tiny_project, epochs=2, lr_t_max=t_max, exp_name=name)
        path = tiny_project / "experiments" / name / "seed_42" / "training_history.csv"
        return float(list(csv.DictReader(open(path)))[0]["lr"])

    short = lr_after_first_epoch(2, "smoke_tmax_short")
    long = lr_after_first_epoch(50, "smoke_tmax_long")

    assert long > short, (
        f"lr_t_max is not reaching the scheduler: {long} vs {short}")
    # Same budget, different schedule: the long period has barely decayed.
    assert long > 9e-4 and short < 6e-4


def test_lr_never_climbs_back_after_the_annealing_period(tiny_project):
    """
    CosineAnnealingLR is periodic. With an epoch budget larger than the
    annealing period, stepping past T_max walks the learning rate back up
    towards its peak — the opposite of what a longer budget should do.
    """
    import csv

    run(tiny_project, epochs=6, lr_t_max=2, min_epochs=1, patience=99,
        exp_name="smoke_tmax_hold")
    path = tiny_project / "experiments" / "smoke_tmax_hold" / "seed_42" / "training_history.csv"
    lrs = [float(r["lr"]) for r in csv.DictReader(open(path))]

    assert len(lrs) == 6
    assert all(b <= a + 1e-12 for a, b in zip(lrs, lrs[1:])), (
        f"learning rate rose again after T_max: {lrs}")
    assert lrs[-1] == pytest.approx(1e-3 * 0.01, rel=1e-6), (
        "the tail should hold at eta_min, not collapse to zero")


def test_config_records_the_stability_settings(tiny_project):
    """Both knobs must land in config.json, or a run cannot be reproduced."""
    report = run(tiny_project, lr_t_max=7, max_grad_norm=0.5, exp_name="smoke_cfg")
    assert report["config"]["lr_t_max"] == 7
    assert report["config"]["max_grad_norm"] == 0.5


# ============================================================================
#  POST-PROCESSING IS REPORTED ALONGSIDE THE RAW RESULT
# ============================================================================

def test_report_carries_both_raw_and_post_processed_metrics(tiny_project):
    """
    One run must answer whether dropping satellite components helps, without
    needing a second run to compare against.
    """
    import csv

    report = run(tiny_project, exp_name="smoke_pp")

    assert "postprocessed" in report
    for key in ("dice_3d", "hd95_mm", "precision", "fp_components"):
        assert key in report["postprocessed"]["macro"], f"missing pp_{key}"
    assert "dice_3d" in report["postprocessed"]["micro"]

    path = tiny_project / "experiments" / "smoke_pp" / "seed_42" / "test_results_per_patient.csv"
    header = next(csv.reader(open(path)))
    assert "dice_3d" in header and "pp_dice_3d" in header


def test_post_processing_can_be_switched_off(tiny_project):
    """
    With filtering disabled the per-patient CSV must keep exactly the original
    schema, so downstream readers of older runs do not break.
    """
    import csv

    report = run(tiny_project, postproc_min_fraction=0.0, exp_name="smoke_nopp")

    assert "postprocessed" not in report
    path = tiny_project / "experiments" / "smoke_nopp" / "seed_42" / "test_results_per_patient.csv"
    header = next(csv.reader(open(path)))
    assert not any(col.startswith("pp_") for col in header)


# ============================================================================
#  GEOMETRY IS HONOURED END TO END
# ============================================================================

def test_reconstruction_is_not_mirrored(tiny_project):
    """
    The decisive test. The synthetic ground truth on disk is the mirror of the
    preprocessed stack, exactly as in the real LAS dataset.

    A perfect prediction is fed straight through the reconstruction path. If the
    canonical reorientation is not inverted, the two do not overlap at all and
    Dice is 0.0.
    """
    from src.evaluation.metrics import reconstruct_patient_3d_volume

    case_id = "c_te1"
    metadata = json.loads(
        (tiny_project / "metadata" / f"{case_id}.json").read_text())
    lbl = np.load(tiny_project / "preprocessed" / "volumes" / f"{case_id}_lbl.npy")

    perfect = {s: lbl[:, :, s].astype(np.float32) for s in range(lbl.shape[2])}
    reconstructed = reconstruct_patient_3d_volume(perfect, metadata, threshold=0.5)

    gt = (np.asanyarray(nib.load(
        str(tiny_project.parent / "archive" / "labelsTr" / f"{case_id}.nii.gz")
    ).dataobj) > 0.5).astype(np.uint8)

    inter = np.logical_and(reconstructed > 0, gt > 0).sum()
    dice = 2.0 * inter / (reconstructed.sum() + gt.sum())
    assert dice == pytest.approx(1.0), (
        f"round-trip Dice {dice:.4f}; a value near 0 means the LAS->RAS flip "
        f"is not being inverted")


def test_validation_is_never_balanced(tiny_project):
    """
    Regardless of the training sampling mode, validation must cover every slice
    of every validation volume.
    """
    from src.training.dataset import build_dataloaders

    _, datasets, _ = build_dataloaders(
        str(tiny_project / "preprocessed"), batch_size=4,
        sampling="balanced", augment="none", num_workers=0)

    expected = SHAPE[2] * len(SPLITS["val"])
    assert len(datasets["val"]) == expected
    assert len(datasets["test"]) == SHAPE[2] * len(SPLITS["test"])
    # Training, by contrast, is balanced and therefore smaller.
    assert len(datasets["train"]) < SHAPE[2] * len(SPLITS["train"])


def test_resume_restores_training_state(tiny_project):
    """A checkpoint must carry enough state to continue rather than restart."""
    run(tiny_project, epochs=2, exp_name="smoke_resume")
    ckpt_path = tiny_project / "experiments" / "smoke_resume" / "seed_42" / "checkpoint.pt"

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 2
    for key in ("model_state_dict", "optimizer_state_dict",
                "scheduler_state_dict", "config", "history"):
        assert key in ckpt, f"checkpoint is missing {key}"

    report = run(tiny_project, epochs=4, exp_name="smoke_resume",
                 resume_path=str(ckpt_path))
    assert report["epochs_run"] == 4, "resume should continue, not restart"
