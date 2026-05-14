# ViroGlobe

**ViroGlobe: Scalable Agent-Based Modeling for High-Resolution, Global-Scale Mosquito-Borne Disease Benchmark and Simulator**

Jiarui Zhu, Xiangcheng Bao, Ziqian Guan, Danyang Chen, Fukang Ge, Yanrui Lu, Yuting Wang, YAO ZIANG, Lin Gu, Jinhao Bi, Yingying Zhu

NeurIPS 2026 Evaluations and Datasets Track Submission  
Initial submission: 01 May 2026  
Last modified: 10 May 2026  
Code license: MIT License

Keywords: Mosquito-borne viruses; Cross-species transmission; Agent-based modeling; Multi-virus dynamics; GPU-accelerated simulation

ViroGlobe is a GPU-accelerated, high-resolution global simulator for multi-virus cross-species mosquito-borne transmission. It integrates climate, ecology, geography, host population, vector-related information, and epidemiological records into a unified spatiotemporal data layer, then simulates mosquito-borne transmission as a spatial agent-based model over tensors of cells, mosquito species, viruses, and time.

The simulator preserves key transmission processes including biting dynamics, infection pressure, vector competence, extrinsic incubation, recovery, severe disease progression, and mortality. Conventional CPU-style ABM updates are reformulated as GPU-vectorized matrix and tensor operations, enabling high-resolution global simulation, long-horizon forecasting, validation, and intervention analysis on a single NVIDIA A100 or Huawei Ascend 910B GPU.

This folder is a GitHub-ready source release. It intentionally excludes local data, generated cache files, experiment outputs, and Python bytecode.

## Repository Layout

```text
src/                  Python package source
scripts/              Utility scripts for validation targets, benchmarks, and parameter training
run.py                Compatibility CLI entry point
requirements.txt      pip dependencies
environment.yml       conda environment
pyproject.toml        package metadata and console script
```

## Data

Large input datasets are not included. By default the simulator expects:

```text
data/static_layers_0_05.nc4
data/climate/MERRA2_*.statD_2d_slv_Nx.YYYYMMDD_0_05.nc4
```

Validation and initial-state CSV files can be supplied through CLI flags such as `--validation-targets` and `--initial-state`.

## Install

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -e .
```

For a conda environment:

```bash
conda env create -f environment.yml
conda activate mosq
pip install -e .
```

## Run

Show CLI options:

```bash
python -m mosq --help
```

Run a small CPU-backed job:

```bash
python -m mosq --cpu --days 7 --start-date 2020-01-01 --allow-cpu-fallback
```

The default output directory is `output/mosq_vectorized_gpu/`.

## Notes

- GPU execution uses CuPy when available.
- Distributed multi-GPU execution additionally requires PyTorch for `torch.distributed`.
- `data/`, `output/`, `scratch/`, climate caches, `.npz`, and `.nc4` files are ignored by default to keep the GitHub repository source-only.
