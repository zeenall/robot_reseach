# Lightweight KAN-Based Search for Heterogeneous Multi-Robot Task Allocation

**Code, frozen weights, datasets, and reproducibility artifacts for Paper 2 (research manuscript; publication status not specified).**

This package studies a lightweight **WarmRelationalKAN / Stage A** decision policy for heterogeneous multi-robot task allocation, using shared **Guided–Scarcity search** with selectable inference budgets K1, K2, Adaptive K2→K4 and K4. It also reproduces comparisons to the published authors' **AttentionNet**, including its native greedy and ten-sample inference pathways.

## Research findings (reported cases only)

| Set / test | Stage A K4 | AttentionNet reference | Interpretation |
| --- | ---: | ---: | --- |
| Mixed 501 / matched K4 | 25.5974 | 25.6704 | paired CI includes zero (no established mean benefit) |
| SADCHER 300 / matched K4 | 21.2680 | 22.0121 | Stage A better on this generated distribution |
| Mixed 501 / native AttentionNet greedy | 25.5974 | 27.1207 | unequal inference budget; Stage A K4 uses four rollouts |
| SADCHER 300 / native AttentionNet greedy | 21.2680 | 23.3209 | unequal inference budget |
| Mixed 501 / native AttentionNet Sampling-10 | 25.5974 | **25.2414** | AttentionNet lower makespan; Stage A K4 quicker |
| SADCHER 300 / native AttentionNet Sampling-10 | **21.2680** | 21.5574 | Stage A K4 lower makespan and quicker |

All 801 cases finished successfully for the six-way direct evaluation; official sampling-10 completed without recorded timeouts in the archived measurements. Source: `results/results37/`.

**Important:** This project does **not** claim KAN intrinsically outperforms a same-size MLP, or that the reported post-hoc tests constitute a new independent confirmatory split. Running ten native AttentionNet sampled episodes consumes more compute than four guided Stage A rollouts.

## Repository map

- `benchmark/`: canonical runnable final benchmark, trained Stage A weights, exported official weights, all 801 locked references and external test-case records.
- `benchmark/official/`: upstream HeteroMRTA components **plus** study-specific network, training and evaluation scripts (see `CODE_PROVENANCE.md`).
- `checkpoints/`: independently trained ablation checkpoints, intermediate/alternate weights, and original primary Stage A checkpoint.
- `results/`: audit-ready numerical results, including Results 34, 36 and direct Results 37.
- `audits/`: saved independent analyses and original protocol notes.
- `scripts/verify_release.py`: checks file hashes, frozen weights, basic result row counts, and Python syntax.
- `REPRODUCIBILITY.md`: command-line evaluation recipes, methodology and cautions.

## Quick start (no retraining)

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
python benchmark/run_direct_runtime_sampling10.py --mixed-per-mode 1 --sadcher-count 1 --threads 1 --native-timeout 25 --bootstrap-draws 300 --out outputs/smoke
```

If you only want the paper's primary Stage A model, load `benchmark/weights/stage_a_best_model.pt` via `load_stage_a()` in `benchmark/official/train_selective_distillation.py`.

For an independent full run, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The `Kaggle_Run_6Ways.ipynb` notebook offers a simpler entry point.

## Attribution and availability

The selected KAN V7.1 pretrained starting weights are retained at `checkpoints/pretraining/`, while the final Stage A weights are `benchmark/weights/stage_a_best_model.pt`.

The original AttentionNet architecture and simulation code come from [marmotlab/HeteroMRTA](https://github.com/marmotlab/HeteroMRTA) and are credited under their upstream Apache-2.0 license. Other research scripts, preprocessing and evaluation logic are study-developed and must not be confused with the upstream implementation. A list of provenance and licensing boundaries is in [CODE_PROVENANCE.md](CODE_PROVENANCE.md).

> This repository's source may be posted publicly before the paper is accepted. Confirm any target journal's code/dataset availability policy and coauthor agreement before making the public GitHub repository visible.

## GitHub publishing

See [README_UPLOAD_AR.md](README_UPLOAD_AR.md) for upload instructions and `PUBLISH_TO_GITHUB.ps1` for CLI-based one-step publishing.
