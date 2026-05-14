# MOSQ

Vectorized mosquito-borne disease simulation code for static geospatial layers and daily climate windows.

This folder is a GitHub-ready source release. It intentionally excludes local data, generated cache files, experiment outputs, and Python bytecode.

## Repository Layout

```text
mosq/                 Python package
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
