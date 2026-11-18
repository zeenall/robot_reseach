# Code provenance and third-party notices

This repository contains original study-specific scripts and **files from an upstream project**.

* Authors' project: [marmotlab/HeteroMRTA](https://github.com/marmotlab/HeteroMRTA), accompanying W. Dai et al., *Heterogeneous Multi-robot Task Allocation and Scheduling via Reinforcement Learning*, IEEE RA-L (2025), DOI: 10.1109/LRA.2025.3534682.
* Upstream files embedded unchanged in `benchmark/official/`: `attention.py`, `worker.py`, `parameters.py`, and `env/task_env.py` (plus empty `env/__init__.py`). The benchmark validates selected upstream source SHA-256 checksums before evaluation.
* Upstream Apache License 2.0 text: `benchmark/official/LICENSE_AUTHORS_APACHE2.txt`. Keep this notice when redistributing upstream files. The Apache license should **not** be represented as a blanket license grant for the other original research files.
* The `AttentionNet_published.pt` file is an export of the authors' official checkpoint `model/save/checkpoint.pth`; the export contains `weights`, `model_name`, and source metadata and is verified by its file SHA-256 (see `MANIFEST.sha256`). Original checkpoint Git blob SHA-1: `ced8f9cab156916dcfdc43ea0c76de3554dc5859`. The source is the authors' project.
* SADCHER generated instances are **cross-dataset** inputs only: all schedulers still use the HeteroMRTA environment. Cross-simulator equivalence is not established. SADCHER source: [jakbichler/Sadcher](https://github.com/jakbichler/Sadcher).

## Licensing of new code

The original code written for this study is made available here for research inspection and reproducibility. **No explicit license for these original files has yet been chosen by the research author(s)**. This statement does not grant additional rights beyond applicable law. Choose an appropriate license with all contributors' agreement before describing the entire repository as open-source.

* The previous-generation KAN V7.1 selection weights are included in `checkpoints/pretraining/` for historical warm-start reproducibility, but are **not** the final Stage A weights used in Results 34–37.
