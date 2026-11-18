# Exact benchmark reproduction

All commands below run from the repository root. The 801 locked test cases include **501 mixed synthetic HeteroMRTA instances** (167 onehot, 167 multihot, 167 hard), and **300 independently generated SADCHER instances** evaluated under the same HeteroMRTA simulator. Every reported result was produced on frozen Stage A and original authors' weights, with *no additional training*.

## Setup

```bash
python -m pip install -r requirements.txt
python scripts/verify_release.py
```

## Four-way KAN search / evaluation

```bash
python benchmark/run_four_way.py --help
python benchmark/run_four_way.py --mixed-per-mode 1 --sadcher-count 1 --out outputs/four_smoke
python benchmark/run_four_way.py --mixed-per-mode 167 --sadcher-count 300 --out outputs/four_full
```

The four methods are K1 (`guided_greedy`), K2 (guided/shortage-greedy), validation-frozen Adaptive K2→K4, and K4 (guided/scarcity greedy + guided/scarcity samples).

## Native full-system comparison (Results 35)

```bash
python benchmark/run_official_fullsystem.py --help
python benchmark/run_official_fullsystem.py --mixed-per-mode 1 --sadcher-count 1 --out outputs/native_smoke
python benchmark/run_official_fullsystem.py --mixed-per-mode 167 --sadcher-count 300 --native-policy greedy --out outputs/native_full
```

## Direct six-way timing (Results 37)

```bash
python benchmark/run_direct_runtime_sampling10.py --help
python benchmark/run_direct_runtime_sampling10.py --mixed-per-mode 1 --sadcher-count 1 --bootstrap-draws 300 --out outputs/timing_smoke
python benchmark/run_direct_runtime_sampling10.py --mixed-per-mode 167 --sadcher-count 300 --bootstrap-draws 10000 --out outputs/timing_full
```

Note that `--native-timeout` defaults in the runner may differ from the original Kaggle run. To reproduce the Results 37 protocol use `--native-timeout 25` and `--threads 1` on CPU. Real-time seconds vary with machine and contention; they are not deterministic across hardware.

## Scientific interpretation

1. The primary Stage A checkpoint is the 25,138-parameter **WarmRelationalKAN**. The comparison AttentionNet has 2,581,632 parameters.
2. K4 with matched search budgets isolates performance under a shared *custom* search; it does **not** compare the unmodified complete systems or independently prove that KAN is better than a parameter-matched MLP.
3. Authors' native one-rollout greedy and ten categorical-sampling episodes are **unequal inference budgets** to K4; present them as separate system-level comparisons.
4. K4 is best by Makespan among the study's four Stage A decoding/search settings. Adaptive trades some quality for shorter runtime, with the rule selected on 72 validation instances.
5. Mixed 501 vs SADCHER 300 are distinct distributions; do not pool their makespan means or claim cross-simulator validation.
6. Results 34–37 are post-hoc analysis of reused cases, not unseen confirmatory tests. Avoid equivalence claims from non-significant p-values.
