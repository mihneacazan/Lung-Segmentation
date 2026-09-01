# Experiment index

One directory per run, `{folder}/seed_{n}/`, each holding `config.json`,
`benchmark_report.json`, `threshold_sweep.json`, `training_history.csv`,
`test_results_per_patient.csv` and `best_model.pt`.

Directory names describe what the run varies. The `exp_name` recorded inside
`config.json` is the original short name the run was launched with, kept as-is so
the artifacts still match the logs; the last column maps between the two.

`baseline_unet_dicece_allslices/` additionally holds `multi_seed_summary.json`,
`threshold_rescore.json` (the three seeds re-swept in original geometry) and
`ensemble_report.json` (the three averaged into one prediction).

Results for every run are collected in
[`../all_experiment_results.csv`](../all_experiment_results.csv); the reasoning
behind each is in [`../../EXPERIMENTS.md`](../../EXPERIMENTS.md).

## Reference baseline

| directory | seeds | configuration | `exp_name` in config |
|---|---|---|---|
| `anneal_over_91pct_of_budget` | 42 | 49,390 steps | `anneal_91pct` |
| `baseline_corrected_seed42` | 42 | 49,390 steps | `base_corrected_seed42` |
| `baseline_corrected_seed43` | 43 | 49,390 steps | `base_corrected_seed43` |
| `baseline_corrected_seed44` | 44 | 49,390 steps | `base_corrected_seed44` |
| `baseline_on_fold2_split` | 42 | 100 epochs | `committed_fold2split` |
| `baseline_unet_dicece_allslices` | 42, 43, 44 | 100 epochs | `baseline` |
| `budget_double_98780steps` | 42 | 98,780 steps | `budget_calibration` |

## Evaluation protocol

| directory | seeds | configuration | `exp_name` in config |
|---|---|---|---|
| `positivetrain_selected_on_all_slices` | 42 | sampling=positives, 49,390 steps | `prot_positives_sel_all` |
| `positivetrain_selected_on_positives_only` | 42 | sampling=positives, 49,390 steps | `prot_positives_sel_positives` |

## Cross-validation

| directory | seeds | configuration | `exp_name` in config |
|---|---|---|---|
| `crossval_epochbudget_fold0` | 42 | 100 epochs | `cv_fold0` |
| `crossval_epochbudget_fold1` | 42 | 100 epochs | `cv_fold1` |
| `crossval_epochbudget_fold2` | 42 | 100 epochs | `cv_fold2` |
| `crossval_epochbudget_fold3` | 42 | 100 epochs | `cv_fold3` |
| `crossval_epochbudget_fold4` | 42 | 100 epochs | `cv_fold4` |
| `crossval_stepbudget_fold0` | 42 | 49,390 steps | `cv6_fold0` |
| `crossval_stepbudget_fold1` | 42 | 49,390 steps | `cv6_fold1` |
| `crossval_stepbudget_fold2` | 42 | 49,390 steps | `cv6_fold2` |
| `crossval_stepbudget_fold3` | 42 | 49,390 steps | `cv6_fold3` |
| `crossval_stepbudget_fold4` | 42 | 49,390 steps | `cv6_fold4` |

## Inter-slice context

| directory | seeds | configuration | `exp_name` in config |
|---|---|---|---|
| `context_1channel_2d` | 42 | 49,390 steps | `context_n1` |
| `context_3channels_pm1mm` | 42 | 3 channels, 49,390 steps | `context_n3` |
| `context_5channels_pm2mm` | 42 | 5 channels, 49,390 steps | `context_n5` |
| `context_7channels_pm3mm` | 42 | 7 channels, 49,390 steps | `context_n7` |

## Resolution and resize mode

| directory | seeds | configuration | `exp_name` in config |
|---|---|---|---|
| `resolution_192_pad_1channel` | 42 | 49,390 steps | `pad192` |
| `resolution_256_pad_1channel` | 42 | 49,390 steps | `pad256` |
| `resolution_256_stretch_1channel` | 42 | 49,390 steps | `res_256` |
| `resolution_320_pad_1channel` | 42 | 49,390 steps | `pad320` |
| `resolution_320_pad_5channels` | 42 | 5 channels, 49,390 steps | `pad320_n5` |
| `resolution_320_pad_7channels` | 42 | 7 channels, 49,390 steps | `pad320_n7` |
| `resolution_320_stretch_1channel` | 42 | 49,390 steps | `res_320` |
| `resolution_320_stretch_5channels_BEST` | 42 | 5 channels, 49,390 steps | `stretch320_n5` |
| `resolution_320_zspacing2.5mm_1channel` | 42 | 90 epochs | `A_res320_2d` |
| `resolution_320_zspacing2.5mm_7channels` | 42 | 7 channels, 90 epochs | `B_res320_25d7` |

## Two-stage training

| directory | seeds | configuration | `exp_name` in config |
|---|---|---|---|
| `twostage_stage1_allslices` | 42 | 49,390 steps | `stage1_all` |
| `twostage_stage2_distance_negatives` | 42 | sampling=hard_negatives, lr=0.0001, fine-tuned from stage 1, 12,000 steps | `stage2_distance` |
| `twostage_stage2_hardneg_ratio0.5_lr1e-3` | 42 | sampling=hard_negatives, neg_ratio=0.5, fine-tuned from stage 1, 12,000 steps | `s2_hn0.5_hi` |
| `twostage_stage2_hardneg_ratio0.5_lr1e-4` | 42 | sampling=hard_negatives, lr=0.0001, neg_ratio=0.5, fine-tuned from stage 1, 12,000 steps | `s2_hn0.5_lo` |
| `twostage_stage2_hardneg_ratio1.0_lr1e-3` | 42 | sampling=hard_negatives, fine-tuned from stage 1, 12,000 steps | `s2_hn1.0_hi` |
| `twostage_stage2_hardneg_ratio1.0_lr1e-4` | 42 | sampling=hard_negatives, lr=0.0001, fine-tuned from stage 1, 12,000 steps | `s2_hn1.0_lo` |
| `twostage_stage2_mined_negatives` | 42 | sampling=mined_negatives, lr=0.0001, fine-tuned from stage 1, 12,000 steps | `stage2_mined` |
| `twostage_stage2_positives_lr1e-3` | 42 | sampling=positives, fine-tuned from stage 1, 12,000 steps | `s2_pos_hi` |
| `twostage_stage2_positives_lr1e-4` | 42 | sampling=positives, lr=0.0001, fine-tuned from stage 1, 12,000 steps | `s2_pos_lo` |
| `twostage_stage2_random_negatives` | 42 | sampling=balanced, lr=0.0001, fine-tuned from stage 1, 12,000 steps | `stage2_random` |

## Lung mask

| directory | seeds | configuration | `exp_name` in config |
|---|---|---|---|
| `lungmask_cascade_masked_input` | 42 | 49,390 steps | `mask_cascade` |
| `lungmask_control_unmasked` | 42 | 49,390 steps | `mask_control` |

## Mask interpolation

| directory | seeds | configuration | `exp_name` in config |
|---|---|---|---|
| `nearestmask_attention_unet` | 42 | attention_unet, 100 epochs | `nn_attention_unet` |
| `nearestmask_segresnet` | 42 | segresnet, lr=0.0003, 100 epochs | `nn_segresnet` |
| `nearestmask_unet` | 42, 43 | 100 epochs | `nn_baseline` |
| `nearestmask_unet_3channels` | 42 | 3 channels, 100 epochs | `nn_unet_25d` |

## Architectures

| directory | seeds | configuration | `exp_name` in config |
|---|---|---|---|
| `arch_attention_unet` | 42 | attention_unet, 100 epochs | `attention_unet` |
| `arch_attention_unet_3channels` | 42 | attention_unet, 3 channels, 100 epochs | `attention_unet_25d` |
| `arch_attention_unet_50ep_patience10` | 42 | attention_unet, 50 epochs | `attention_unet_50ep_pat10` |
| `arch_segresnet_3channels` | 42 | segresnet, 3 channels, lr=0.0003, 100 epochs | `segresnet_25d` |
| `arch_segresnet_lr3e-4` | 42 | segresnet, lr=0.0003, 100 epochs | `segresnet` |
| `arch_unet_3channels` | 42 | 3 channels, 100 epochs | `unet_25d` |
| `arch_unet_double_width_lr3e-4` | 42 | lr=0.0003, 100 epochs | `unet_wide_lr0.0003` |
| `arch_unet_lr3e-4` | 42 | lr=0.0003, 100 epochs | `unet_lowlr` |

## Loss functions

| directory | seeds | configuration | `exp_name` in config |
|---|---|---|---|
| `loss_dice_focal` | 42 | dice_focal, 100 epochs | `dice_focal` |
| `loss_focal_tversky_beta0.6` | 42 | focal_tversky, 100 epochs | `focal_tversky_b06` |
| `loss_focal_tversky_beta0.7` | 42 | focal_tversky, 100 epochs | `focal_tversky` |
| `loss_tversky_beta0.6` | 42 | tversky, 100 epochs | `tversky_b06` |
| `loss_tversky_beta0.7` | 42 | tversky, 100 epochs | `tversky` |

## Sampling and augmentation

| directory | seeds | configuration | `exp_name` in config |
|---|---|---|---|
| `augment_none` | 42 | augment=none, 100 epochs | `no_augment` |
| `sampling_balanced_1to1` | 42 | sampling=balanced, 100 epochs | `dicece_balanced` |
| `sampling_distance_negatives_attention_unet` | 42 | attention_unet, sampling=hard_negatives, 100 epochs | `hard_negatives_attention_unet` |
| `sampling_distance_negatives_segresnet` | 42 | segresnet, sampling=hard_negatives, lr=0.0003, 100 epochs | `hard_negatives_segresnet` |
| `sampling_distance_negatives_unet` | 42 | sampling=hard_negatives, 100 epochs | `hard_negatives_unet` |
| `sampling_distance_negatives_unet_3channels` | 42 | sampling=hard_negatives, 3 channels, 100 epochs | `hard_negatives_unet_25d` |

## Tumour-centred crop

| directory | seeds | configuration | `exp_name` in config |
|---|---|---|---|
| `tumourcrop96_attention_unet` | 42 | attention_unet, crop=tumor, 100 epochs | `crop96_attention_unet` |
| `tumourcrop96_segresnet` | 42 | segresnet, crop=tumor, lr=0.0003, 100 epochs | `crop96_segresnet` |
| `tumourcrop96_unet` | 42 | crop=tumor, 100 epochs | `crop96_unet` |
| `tumourcrop96_unet_3channels` | 42 | 3 channels, crop=tumor, 100 epochs | `crop96_unet_25d` |

## Repeats inside later notebooks

| directory | seeds | configuration | `exp_name` in config |
|---|---|---|---|
| `repeat_attention_unet_50ep_3seeds` | 42, 43, 44 | attention_unet, 50 epochs | `expA_attn_all` |
| `repeat_attention_unet_50ep_patience10` | 42 | attention_unet, 50 epochs | `expH1_attn_50ep_pat10` |
| `repeat_attention_unet_arch_study` | 42 | attention_unet, 100 epochs | `expE2_attention_unet` |
| `repeat_augment_none` | 42 | augment=none, 100 epochs | `expG2_no_augment` |
| `repeat_dice_focal_loss_study` | 42 | dice_focal, 100 epochs | `expF1_dice_focal` |
| `repeat_focal_tversky_beta0.6` | 42 | focal_tversky, 100 epochs | `expH3_focal_tversky_b06` |
| `repeat_focal_tversky_loss_study` | 42 | focal_tversky, 100 epochs | `expF3_focal_tversky` |
| `repeat_sampling_balanced` | 42 | sampling=balanced, 100 epochs | `expG1_dicece_balanced` |
| `repeat_segresnet_gradient_clipped` | 42 | segresnet, 50 epochs | `expC_segresnet_all_clipped` |
| `repeat_segresnet_lr3e-4_arch_study` | 42 | segresnet, lr=0.0003, 100 epochs | `expE4_segresnet_lowlr` |
| `repeat_tversky_beta0.6` | 42 | tversky, 100 epochs | `expH2_tversky_b06` |
| `repeat_tversky_loss_study` | 42 | tversky, 100 epochs | `expF2_tversky` |
| `repeat_unet_100ep_tmax50` | 42 | 100 epochs | `expC_unet_all_100ep_tmax50` |
| `repeat_unet_3channels_arch_study` | 42 | 3 channels, 100 epochs | `expE3_unet25d` |
| `repeat_unet_baseline_arch_study` | 42 | 100 epochs | `expE1_unet_baseline` |
| `repeat_unet_baseline_seed43` | 43 | 100 epochs | `expG3_unet_baseline_s43` |
| `repeat_unet_baseline_seed44` | 44 | 100 epochs | `expG4_unet_baseline_s44` |
| `repeat_unet_lr3e-4_loss_study` | 42 | lr=0.0003, 100 epochs | `expF4_unet_lowlr` |

