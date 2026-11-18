The `benchmark/` folder is the canonical package for exact Results 34–37 reproduction.

- `weights/stage_a_best_model.pt`: frozen user-trained 25,138 parameter Stage A model.
- `weights/AttentionNet_published.pt`: verified exported state dict from [the authors' public checkpoint](https://github.com/marmotlab/HeteroMRTA/tree/main/model/save) (Apache-2.0 upstream code and attribution retained).
- `reference/stage_a_K4_801.csv`: 801-case K4 locked reference.
- `reference/four_way_results36_compact.csv`: locked K1, K2, Adaptive, K4 per-case decisions.
- `reference/adaptive_frozen_rule_results36.json`: 72-case validation-frozen threshold.
- `cases/sadcher300_official_original_cases.json.xz`: independent cross-dataset 300 test cases evaluated through the original HeteroMRTA simulator.
- `run_official_fullsystem.py`: Stage A K4 versus authors' native greedy or sampled policy.
- `run_four_way.py`: 4 Stage A decoding/search budgets.
- `run_direct_runtime_sampling10.py`: six-way independent wall-time measurements.

**Do not treat the authors' native greedy one-rollout benchmark as an equal-compute neural-network architecture comparison.**
