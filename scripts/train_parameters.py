from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
VIRUS_NAMES = [
    "dengue",
    "zika",
    "yellow_fever",
    "west_nile",
    "japanese_encephalitis",
]


def _resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _float_list(values: list[str] | None, default: float = 1.0) -> list[float]:
    if values is None:
        return [float(default)] * len(VIRUS_NAMES)
    parsed = [float(value) for value in values]
    if len(parsed) == 1:
        return parsed * len(VIRUS_NAMES)
    if len(parsed) != len(VIRUS_NAMES):
        raise ValueError(f"Expected 1 or {len(VIRUS_NAMES)} values, got {len(parsed)}")
    return parsed


def _format_values(values: list[float]) -> list[str]:
    return [f"{value:.6g}" for value in values]


def _simulation_command(args: argparse.Namespace, output_dir: Path, host_scale: list[float], vector_scale: list[float]) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "run.py"),
        "--days",
        str(args.days),
        "--start-date",
        args.start_date,
        "--output-dir",
        str(output_dir),
        "--temperature-units",
        args.temperature_units,
        "--precipitation-units",
        args.precipitation_units,
        "--humidity-units",
        args.humidity_units,
        "--disable-default-seeds",
        "--initial-state",
        str(_resolve_path(args.initial_state)),
        "--initial-vector-seed-rate",
        str(args.initial_vector_seed_rate),
        "--reservoir-force-scale",
        str(args.reservoir_force_scale),
        "--immunity-waning-rate",
        str(args.immunity_waning_rate),
        "--host-replenishment-rate",
        str(args.host_replenishment_rate),
        "--spatial-diffusion-rate",
        str(args.spatial_diffusion_rate),
        "--host-transmission-coeff",
        str(args.host_transmission_coeff),
        "--vector-transmission-coeff",
        str(args.vector_transmission_coeff),
        "--host-transmission-scale",
        *_format_values(host_scale),
        "--vector-transmission-scale",
        *_format_values(vector_scale),
        "--validation-targets",
        str(_resolve_path(args.validation_targets)),
        "--validation-frequency",
        args.validation_frequency,
    ]
    if args.allow_cpu_fallback:
        command.append("--allow-cpu-fallback")
    if args.cpu:
        command.append("--cpu")
    return command


def _load_validation_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_virus = {name: {"observed": 0.0, "predicted": 0.0} for name in VIRUS_NAMES}
    squared_log_errors: list[float] = []
    smape_terms: list[float] = []
    for row in rows:
        virus = row["virus"]
        observed = float(row["observed_cases"])
        predicted = float(row["predicted_cases"])
        by_virus[virus]["observed"] += observed
        by_virus[virus]["predicted"] += predicted
        squared_log_errors.append((math.log1p(predicted) - math.log1p(observed)) ** 2)
        denom = abs(observed) + abs(predicted)
        if denom > 0.0:
            smape_terms.append(2.0 * abs(predicted - observed) / denom)
    return {
        "log_rmse": math.sqrt(sum(squared_log_errors) / max(len(squared_log_errors), 1)),
        "smape": 100.0 * sum(smape_terms) / max(len(smape_terms), 1),
        "by_virus": by_virus,
    }


def _update_scales(
    current: list[float],
    by_virus: dict[str, dict[str, float]],
    step_exponent: float,
    min_update: float,
    max_update: float,
    min_scale: float,
    max_scale: float,
) -> list[float]:
    updated: list[float] = []
    for index, virus in enumerate(VIRUS_NAMES):
        observed = by_virus[virus]["observed"]
        predicted = by_virus[virus]["predicted"]
        ratio = observed / max(predicted, 1e-6)
        multiplier = ratio ** step_exponent
        multiplier = min(max(multiplier, min_update), max_update)
        updated.append(min(max(current[index] * multiplier, min_scale), max_scale))
    return updated


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Iteratively train per-virus transmission scales by rerunning the simulator.")
    parser.add_argument("--output-root", default="output/parameter_training")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--days", type=int, required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--initial-state", required=True)
    parser.add_argument("--validation-targets", required=True)
    parser.add_argument("--validation-frequency", choices=("year", "month", "day"), default="year")
    parser.add_argument("--temperature-units", default="kelvin")
    parser.add_argument("--precipitation-units", default="meter")
    parser.add_argument("--humidity-units", default="percent")
    parser.add_argument("--initial-vector-seed-rate", type=float, default=0.002)
    parser.add_argument("--reservoir-force-scale", type=float, default=0.0)
    parser.add_argument("--immunity-waning-rate", type=float, default=0.0005)
    parser.add_argument("--host-replenishment-rate", type=float, default=0.00005)
    parser.add_argument("--spatial-diffusion-rate", type=float, default=0.25)
    parser.add_argument("--host-transmission-coeff", type=float, default=0.85)
    parser.add_argument("--vector-transmission-coeff", type=float, default=0.95)
    parser.add_argument("--host-transmission-scale", nargs="+", default=None)
    parser.add_argument("--vector-transmission-scale", nargs="+", default=None)
    parser.add_argument("--step-exponent", type=float, default=0.25)
    parser.add_argument("--min-update", type=float, default=0.35)
    parser.add_argument("--max-update", type=float, default=2.50)
    parser.add_argument("--min-scale", type=float, default=0.02)
    parser.add_argument("--max-scale", type=float, default=12.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")

    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    host_scale = _float_list(args.host_transmission_scale)
    vector_scale = _float_list(args.vector_transmission_scale)

    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for iteration in range(args.iterations):
        output_dir = output_root / f"iter_{iteration:02d}"
        command = _simulation_command(args, output_dir, host_scale, vector_scale)
        print(f"[train] iteration {iteration}: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=REPO_ROOT, check=True)

        validation_path = output_dir / "validation.csv"
        metrics = _metrics(_load_validation_rows(validation_path))
        row: dict[str, Any] = {
            "iteration": iteration,
            "log_rmse": metrics["log_rmse"],
            "smape": metrics["smape"],
        }
        for index, virus in enumerate(VIRUS_NAMES):
            row[f"host_scale_{virus}"] = host_scale[index]
            row[f"vector_scale_{virus}"] = vector_scale[index]
            row[f"observed_{virus}"] = metrics["by_virus"][virus]["observed"]
            row[f"predicted_{virus}"] = metrics["by_virus"][virus]["predicted"]
        history.append(row)

        if best is None or metrics["log_rmse"] < best["metrics"]["log_rmse"]:
            best = {
                "iteration": iteration,
                "metrics": metrics,
                "host_transmission_scale": list(host_scale),
                "vector_transmission_scale": list(vector_scale),
                "output_dir": str(output_dir),
                "command": command,
            }

        host_scale = _update_scales(
            host_scale,
            metrics["by_virus"],
            args.step_exponent,
            args.min_update,
            args.max_update,
            args.min_scale,
            args.max_scale,
        )
        vector_scale = _update_scales(
            vector_scale,
            metrics["by_virus"],
            args.step_exponent,
            args.min_update,
            args.max_update,
            args.min_scale,
            args.max_scale,
        )

    _write_csv(output_root / "training_history.csv", history)
    if best is not None:
        (output_root / "best_parameters.json").write_text(
            json.dumps(best, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (output_root / "best_command.txt").write_text(" ".join(best["command"]) + "\n", encoding="utf-8")
        print(f"[train] best iteration: {best['iteration']}")
        print(f"[train] best log_rmse: {best['metrics']['log_rmse']:.6g}")
        print(f"[train] wrote {output_root / 'best_parameters.json'}")


if __name__ == "__main__":
    main()
