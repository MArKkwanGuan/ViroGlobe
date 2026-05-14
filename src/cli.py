from __future__ import annotations

import argparse
import calendar
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import torch
    import torch.distributed as dist
    TORCH_DISTRIBUTED_AVAILABLE = True
except ImportError:
    torch = None
    dist = None
    TORCH_DISTRIBUTED_AVAILABLE = False

from .backend import MultiGPUBackend, get_backend, get_multi_gpu_backend, get_world_size, is_distributed_run
from .calibration import (
    apply_calibration_to_windows,
    build_calibrated_summary,
    build_calibration_result,
    load_calibration_targets,
)
from .constants import N_VIRUS, VIRUS_NAMES
from .evaluation import evaluate_output, load_validation_observations, write_evaluation_outputs
from .io import (
    audit_climate_dataset,
    audit_static_dataset,
    climate_day_cache_path,
    climate_window_cache_path,
    climate_window_shard_available,
    climate_window_to_backend_async,
    climate_window_to_backend,
    dump_json,
    list_climate_files,
    load_climate_window_cpu,
    load_static_grid,
    partition_static_grid,
    prebuild_climate_shard,
    prebuild_climate_windows,
    validate_inputs,
)
from .multi_gpu import SpatialDecomposition
from .simulation import SimulationConfig, VectorizedABMSimulator, WindowWriter, write_metadata


PROJECT_ROOT = Path(__file__).resolve().parent
ADDITIVE_WINDOW_KEYS = (
    "new_exposed",
    "new_infectious",
    "new_severe",
    "new_deaths",
    "reported",
    "adult_vectors",
    "infectious_vectors",
    "precip_sum",
)
MEAN_WINDOW_KEYS = ("water_level_mean", "temp_mean")
SUMMARY_KEYS = (
    "reported_cases_total",
    "dead_total",
    "infectious_humans_total",
    "adult_vectors_total",
    "infectious_vectors_total",
)


class ClimateWindowPrefetcher:
    def __init__(
        self,
        starts: list[int],
        load_fn: Any,
        prefetch_windows: int,
        use_threads: bool = True,
        can_prefetch_fn: Any | None = None,
    ):
        self.starts = starts
        self.load_fn = load_fn
        self.prefetch_windows = max(int(prefetch_windows), 0)
        self.use_threads = use_threads
        self.can_prefetch_fn = can_prefetch_fn

    def _should_prefetch(self, start_day: int) -> bool:
        if not self.use_threads or self.prefetch_windows == 0:
            return False
        if self.can_prefetch_fn is None:
            return True
        return bool(self.can_prefetch_fn(start_day))

    def __iter__(self):
        if self.prefetch_windows == 0 or not self.use_threads:
            for start_day in self.starts:
                yield start_day, self.load_fn(start_day)
            return

        with ThreadPoolExecutor(max_workers=min(self.prefetch_windows, 4)) as executor:
            pending: list[tuple[int, Future | None]] = []
            starts_iter = iter(self.starts)

            def _schedule(start_day: int) -> tuple[int, Future | None]:
                if self._should_prefetch(start_day):
                    return start_day, executor.submit(self.load_fn, start_day)
                return start_day, None

            for _ in range(self.prefetch_windows):
                try:
                    start_day = next(starts_iter)
                except StopIteration:
                    break
                pending.append(_schedule(start_day))

            while pending:
                start_day, future = pending.pop(0)
                yield start_day, future.result() if future is not None else self.load_fn(start_day)
                try:
                    next_start = next(starts_iter)
                except StopIteration:
                    continue
                pending.append(_schedule(next_start))


class DistributedReducer:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.initialized_here = False
        self.rank = 0
        self.world_size = 1

    def initialize(self) -> None:
        if not self.enabled:
            return
        if not TORCH_DISTRIBUTED_AVAILABLE:
            raise RuntimeError("Multi-GPU execution requires torch.distributed for inter-process reduction.")
        if dist is None or not dist.is_available():
            raise RuntimeError("torch.distributed is not available in the current environment.")
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo", init_method="env://")
            self.initialized_here = True
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

    def finalize(self) -> None:
        if self.enabled and self.initialized_here and dist is not None and dist.is_initialized():
            dist.destroy_process_group()

    def barrier(self) -> None:
        if self.enabled and dist is not None and dist.is_initialized():
            dist.barrier()

    def gather_object(self, obj: Any) -> list[Any] | None:
        if not self.enabled:
            return [obj]
        gathered = [None for _ in range(self.world_size)] if self.rank == 0 else None
        dist.gather_object(obj, gathered, dst=0)
        return gathered

    def reduce_sum_array(self, arr: np.ndarray) -> np.ndarray | None:
        arr_np = np.asarray(arr, dtype=np.float32)
        if not self.enabled:
            return arr_np
        tensor = torch.from_numpy(arr_np.copy())
        dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            return tensor.cpu().numpy().copy()
        return None

    def reduce_weighted_mean_array(self, arr: np.ndarray, weight: float) -> np.ndarray | None:
        arr_np = np.asarray(arr, dtype=np.float32)
        if not self.enabled:
            return arr_np
        weighted = torch.from_numpy((arr_np * np.float32(weight)).copy())
        total_weight = torch.tensor([float(weight)], dtype=torch.float32)
        dist.reduce(weighted, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(total_weight, dst=0, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            denom = max(float(total_weight.item()), 1.0)
            return (weighted.cpu().numpy() / np.float32(denom)).astype(np.float32)
        return None

    def reduce_window_payload(
        self,
        dates: list[str],
        local_results: dict[str, np.ndarray],
        active_cells: int,
    ) -> dict[str, np.ndarray] | None:
        reduced: dict[str, np.ndarray] = {}
        for key in ADDITIVE_WINDOW_KEYS:
            reduced_value = self.reduce_sum_array(np.asarray(local_results[key], dtype=np.float32))
            if reduced_value is not None:
                reduced[key] = reduced_value
        for key in MEAN_WINDOW_KEYS:
            reduced_value = self.reduce_weighted_mean_array(np.asarray(local_results[key], dtype=np.float32), float(active_cells))
            if reduced_value is not None:
                reduced[key] = reduced_value
        if not self.enabled or self.rank == 0:
            reduced["dates"] = np.asarray(dates)
            return reduced
        return None

    def reduce_summary_payload(self, local_summary: dict[str, Any]) -> dict[str, Any] | None:
        reduced = {
            "virus_names": local_summary["virus_names"],
            "species_names": local_summary["species_names"],
        }
        for key in SUMMARY_KEYS:
            reduced_value = self.reduce_sum_array(np.asarray(local_summary[key], dtype=np.float32))
            if reduced_value is not None:
                reduced[key] = reduced_value.tolist()
        if not self.enabled or self.rank == 0:
            return reduced
        return None


def _date_label_to_datetime(value: Any) -> datetime:
    text = str(value).strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m", "%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"Could not parse date or period value: {value!r}")


def _period_key_from_date(date_value: datetime, frequency: str) -> str:
    if frequency == "year":
        return date_value.strftime("%Y")
    if frequency == "month":
        return date_value.strftime("%Y-%m")
    if frequency == "day":
        return date_value.strftime("%Y-%m-%d")
    raise ValueError(f"Unsupported calibration frequency: {frequency}")


def _period_day_count(date_value: datetime, frequency: str) -> int:
    if frequency == "year":
        return 366 if calendar.isleap(date_value.year) else 365
    if frequency == "month":
        return calendar.monthrange(date_value.year, date_value.month)[1]
    if frequency == "day":
        return 1
    raise ValueError(f"Unsupported calibration frequency: {frequency}")


class OnlineWindowCalibrator:
    def __init__(
        self,
        target_path: str,
        observations: list[Any],
        frequency: str,
        min_scale: float,
        max_scale: float,
    ):
        self.target_path = target_path
        self.frequency = frequency
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.target_by_period: dict[tuple[int, str], float] = {}
        for obs in observations:
            key = (int(obs.virus_index), str(obs.period))
            self.target_by_period[key] = self.target_by_period.get(key, 0.0) + float(obs.observed_cases)

        self.current_scale = np.ones(N_VIRUS, dtype=np.float64)
        self.predicted_to_date = np.zeros(N_VIRUS, dtype=np.float64)
        self.target_to_date = np.zeros(N_VIRUS, dtype=np.float64)
        self.calibrated_to_date = np.zeros(N_VIRUS, dtype=np.float64)
        self.target_days_seen = np.zeros(N_VIRUS, dtype=np.int64)
        self.period_predicted: dict[str, np.ndarray] = {}
        self.period_target: dict[str, np.ndarray] = {}
        self.period_calibrated: dict[str, np.ndarray] = {}
        self.period_days_seen: dict[str, np.ndarray] = {}
        self.period_scale: dict[str, np.ndarray] = {}
        self.finalized_period_scales_applied = False
        self.finalized_window_files: list[str] = []
        self.history: list[dict[str, Any]] = []

    def manifest_header(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "method": "online_multiplicative_reported_cases",
            "target_path": self.target_path,
            "frequency": self.frequency,
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
            "target_allocation": (
                "Period totals are prorated evenly across calendar days; scale estimates reset "
                "for each calibration period."
            ),
            "reported_calibrated_key": "reported_calibrated",
            "history_file": "online_calibration.json",
        }

    def _target_for_day(self, raw_date: Any) -> tuple[np.ndarray, np.ndarray]:
        date_value = _date_label_to_datetime(raw_date)
        period = _period_key_from_date(date_value, self.frequency)
        period_days = max(_period_day_count(date_value, self.frequency), 1)
        target = np.zeros(N_VIRUS, dtype=np.float64)
        active = np.zeros(N_VIRUS, dtype=bool)
        for virus_index in range(N_VIRUS):
            period_target = self.target_by_period.get((virus_index, period))
            if period_target is None:
                continue
            target[virus_index] = float(period_target) / float(period_days)
            active[virus_index] = True
        return target, active

    def update_window(
        self,
        dates: list[Any] | np.ndarray,
        reported: np.ndarray,
        window_index: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        reported_np = np.asarray(reported, dtype=np.float64)
        if reported_np.ndim != 2 or reported_np.shape[1] != N_VIRUS:
            raise ValueError(f"reported must have shape (days, {N_VIRUS}) for online calibration")

        dates_list = np.asarray(dates).tolist()
        window_target = np.zeros(N_VIRUS, dtype=np.float64)
        window_target_days = np.zeros(N_VIRUS, dtype=np.int64)
        period_by_day: list[str] = []
        for raw_date in dates_list:
            period = _period_key_from_date(_date_label_to_datetime(raw_date), self.frequency)
            period_by_day.append(period)
            daily_target, active = self._target_for_day(raw_date)
            window_target += daily_target
            window_target_days += active.astype(np.int64)

        self.predicted_to_date += np.sum(reported_np, axis=0)
        self.target_to_date += window_target
        self.target_days_seen += window_target_days

        calibrated = np.zeros_like(reported_np, dtype=np.float32)
        period_scales: dict[str, list[float]] = {}
        period_statuses: dict[str, list[str]] = {}
        for period in sorted(set(period_by_day)):
            indices = [idx for idx, item in enumerate(period_by_day) if item == period]
            period_reported = np.sum(reported_np[indices], axis=0)
            period_target_window = np.zeros(N_VIRUS, dtype=np.float64)
            period_days_window = np.zeros(N_VIRUS, dtype=np.int64)
            for idx in indices:
                daily_target, active = self._target_for_day(dates_list[idx])
                period_target_window += daily_target
                period_days_window += active.astype(np.int64)

            predicted = self.period_predicted.setdefault(period, np.zeros(N_VIRUS, dtype=np.float64))
            target = self.period_target.setdefault(period, np.zeros(N_VIRUS, dtype=np.float64))
            days_seen = self.period_days_seen.setdefault(period, np.zeros(N_VIRUS, dtype=np.int64))
            predicted += period_reported
            target += period_target_window
            days_seen += period_days_window

            scale = self.period_scale.get(period, np.ones(N_VIRUS, dtype=np.float64)).copy()
            statuses: list[str] = []
            for virus_index in range(N_VIRUS):
                if days_seen[virus_index] == 0:
                    statuses.append("no_target")
                    continue
                predicted_value = float(predicted[virus_index])
                target_value = float(target[virus_index])
                if predicted_value <= 1e-6:
                    raw_scale = 1.0 if target_value <= 1e-6 else self.max_scale
                    status = "target_and_prediction_zero" if target_value <= 1e-6 else "prediction_zero_clamped"
                else:
                    raw_scale = target_value / predicted_value
                    status = "ok"
                scale_value = float(np.clip(raw_scale, self.min_scale, self.max_scale))
                if status == "ok" and scale_value != raw_scale:
                    status = "clamped"
                scale[virus_index] = scale_value
                statuses.append(status)

            self.period_scale[period] = scale
            period_scales[period] = [float(value) for value in scale]
            period_statuses[period] = statuses
            calibrated[indices] = (reported_np[indices] * scale[None, :]).astype(np.float32)
            period_calibrated = self.period_calibrated.setdefault(period, np.zeros(N_VIRUS, dtype=np.float64))
            period_calibrated += np.sum(calibrated[indices], axis=0)

        if period_by_day:
            self.current_scale = self.period_scale[period_by_day[-1]].copy()
        self.calibrated_to_date += np.sum(calibrated, axis=0)

        record = {
            "window_index": int(window_index),
            "start_date": str(dates_list[0]) if len(dates_list) else None,
            "end_date": str(dates_list[-1]) if len(dates_list) else None,
            "periods": sorted(set(period_by_day)),
            "scale": [float(value) for value in self.current_scale],
            "status": period_statuses.get(period_by_day[-1], []) if period_by_day else [],
            "period_scales": period_scales,
            "period_statuses": period_statuses,
            "window_target": [float(value) for value in window_target],
            "target_to_date": [float(value) for value in self.target_to_date],
            "predicted_to_date": [float(value) for value in self.predicted_to_date],
            "calibrated_to_date": [float(value) for value in self.calibrated_to_date],
        }
        self.history.append(record)
        return calibrated, record

    def summary(self) -> dict[str, Any]:
        payload = self.manifest_header()
        payload.update(
            {
                "updates": len(self.history),
                "finalized_period_scales_applied": bool(self.finalized_period_scales_applied),
                "finalized_window_files": list(self.finalized_window_files),
                "final_scale": [float(value) for value in self.current_scale],
                "period_final_scale": {
                    period: [float(value) for value in scale]
                    for period, scale in sorted(self.period_scale.items())
                },
                "target_to_date": {
                    name: float(self.target_to_date[index])
                    for index, name in enumerate(VIRUS_NAMES)
                },
                "predicted_to_date": {
                    name: float(self.predicted_to_date[index])
                    for index, name in enumerate(VIRUS_NAMES)
                },
                "calibrated_to_date": {
                    name: float(self.calibrated_to_date[index])
                    for index, name in enumerate(VIRUS_NAMES)
                },
            }
        )
        return payload

    def file_payload(self) -> dict[str, Any]:
        payload = self.summary()
        payload["history"] = self.history
        return payload


class SingleGPUWindowPipeline:
    def __init__(self, backend: Any):
        self.backend = backend
        self.enabled = bool(getattr(backend, "gpu_enabled", False) and hasattr(backend.xp, "cuda"))
        self.transfer_stream = backend.xp.cuda.Stream(non_blocking=True) if self.enabled else None

    def _stage(self, climate_window_cpu: Any):
        if not self.enabled:
            return climate_window_to_backend_async(climate_window_cpu, self.backend, use_pinned_memory=False)
        return climate_window_to_backend_async(
            climate_window_cpu,
            self.backend,
            stream=self.transfer_stream,
            use_pinned_memory=True,
        )

    def iter_ready(self, cpu_iter: Any):
        iterator = iter(cpu_iter)
        try:
            current_start_day, current_cpu_window = next(iterator)
        except StopIteration:
            return

        current_transfer = self._stage(current_cpu_window)

        for next_start_day, next_cpu_window in iterator:
            ready_window = current_transfer.wait()
            next_transfer = self._stage(next_cpu_window)
            yield current_start_day, current_cpu_window, ready_window
            current_start_day = next_start_day
            current_cpu_window = next_cpu_window
            current_transfer = next_transfer

        yield current_start_day, current_cpu_window, current_transfer.wait()


def _resolve_input_path(path_like: str, must_exist: bool = True) -> str:
    candidate = Path(path_like)
    search_order = [candidate]
    if not candidate.is_absolute():
        search_order.append(PROJECT_ROOT / candidate)
    for item in search_order:
        if not must_exist or item.exists():
            return str(item.resolve())
    raise FileNotFoundError(f"Could not resolve path: {path_like}")


def _resolve_output_path(path_like: str) -> str:
    candidate = Path(path_like)
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())


def _reporting_rate_scale_arg(args: argparse.Namespace) -> list[float] | None:
    if args.reporting_rate_scale is None:
        return None
    return [float(value) for value in args.reporting_rate_scale]


def _float_list_arg(values: list[float] | None) -> list[float] | None:
    if values is None:
        return None
    return [float(value) for value in values]


def _seasonal_seed_cases_arg(args: argparse.Namespace) -> list[float] | None:
    return _float_list_arg(args.seasonal_seed_cases)


def _optional_input_path(path_like: str | None) -> str | None:
    if not path_like:
        return None
    return _resolve_input_path(path_like)


def _parse_start_date(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError("--start-date must be YYYY, YYYYMMDD, or YYYY-MM-DD")


def _select_climate_files_from_start(climate_files: list[Any], start_date: str | None) -> list[Any]:
    parsed = _parse_start_date(start_date)
    if parsed is None:
        return climate_files
    selected = [record for record in climate_files if record.date >= parsed]
    if not selected:
        first = climate_files[0].date.strftime("%Y-%m-%d") if climate_files else "none"
        last = climate_files[-1].date.strftime("%Y-%m-%d") if climate_files else "none"
        raise ValueError(
            f"No climate files found on or after {parsed.strftime('%Y-%m-%d')}; "
            f"available range is {first} to {last}."
        )
    return selected


def _validation_prediction_key(args: argparse.Namespace) -> str:
    if args.validation_prediction_key:
        return str(args.validation_prediction_key)
    return "reported_calibrated" if args.apply_calibration or args.online_calibration else "reported"


def _build_online_window_calibrator(args: argparse.Namespace) -> OnlineWindowCalibrator | None:
    if not args.online_calibration:
        return None
    target_path = _resolve_input_path(args.calibration_targets)
    observations = load_validation_observations(target_path, frequency=args.calibration_frequency)
    return OnlineWindowCalibrator(
        target_path=target_path,
        observations=observations,
        frequency=args.calibration_frequency,
        min_scale=args.calibration_min_scale,
        max_scale=args.calibration_max_scale,
    )


def _apply_online_calibration_to_payload(
    payload: dict[str, np.ndarray],
    online_calibrator: OnlineWindowCalibrator | None,
    window_index: int,
) -> dict[str, Any] | None:
    if online_calibrator is None:
        return None
    calibrated, record = online_calibrator.update_window(
        dates=payload["dates"],
        reported=payload["reported"],
        window_index=window_index,
    )
    payload["reported_calibrated"] = calibrated
    payload["online_calibration_scale"] = np.asarray(record["scale"], dtype=np.float32)
    return record


def _finalize_online_calibration_windows(
    output_dir: Path,
    manifest: dict[str, Any],
    online_calibrator: OnlineWindowCalibrator | None,
) -> None:
    if online_calibrator is None:
        return
    output_path = Path(output_dir)
    totals = np.zeros(N_VIRUS, dtype=np.float64)
    updated_files: list[str] = []
    for window in manifest.get("window_files", []):
        file_name = window["file"] if isinstance(window, dict) else str(window)
        chunk_path = output_path / "chunks" / file_name
        with np.load(chunk_path, allow_pickle=False) as payload:
            data = {key: payload[key] for key in payload.files}
        if "reported" not in data:
            continue
        dates = np.asarray(data["dates"]).tolist()
        reported = np.asarray(data["reported"], dtype=np.float32)
        calibrated = np.zeros_like(reported, dtype=np.float32)
        scale_by_day = np.ones_like(reported, dtype=np.float32)
        for day_index, raw_date in enumerate(dates):
            period = _period_key_from_date(_date_label_to_datetime(raw_date), online_calibrator.frequency)
            scale = np.asarray(
                online_calibrator.period_scale.get(period, np.ones(N_VIRUS, dtype=np.float64)),
                dtype=np.float32,
            )
            calibrated[day_index] = reported[day_index] * scale
            scale_by_day[day_index] = scale
        data["reported_calibrated"] = calibrated
        data["online_period_final_calibration_scale"] = scale_by_day
        np.savez(chunk_path, **data)
        totals += np.sum(calibrated, axis=0)
        updated_files.append(file_name)
        if isinstance(window, dict):
            window["finalized_online_calibration"] = True

    online_calibrator.calibrated_to_date = totals
    online_calibrator.finalized_period_scales_applied = True
    online_calibrator.finalized_window_files = updated_files
    manifest["online_calibration"] = online_calibrator.manifest_header()
    manifest["online_calibration"]["finalized_period_scales_applied"] = True
    manifest["online_calibration"]["finalized_window_file_count"] = len(updated_files)


def _attach_online_calibration_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    online_calibrator: OnlineWindowCalibrator | None,
) -> None:
    if online_calibrator is None:
        return
    payload = online_calibrator.file_payload()
    dump_json(output_dir / "online_calibration.json", payload)
    summary["reported_cases_total_online_calibrated"] = [
        float(value) for value in online_calibrator.calibrated_to_date
    ]
    summary["online_calibration"] = online_calibrator.summary()
    manifest["online_calibration"] = online_calibrator.manifest_header()


def _write_validation_outputs(
    output_dir: Path,
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not args.validation_targets:
        return None

    target_path = _resolve_input_path(args.validation_targets)
    prediction_key = _validation_prediction_key(args)
    observations = load_validation_observations(target_path, frequency=args.validation_frequency)
    evaluation = evaluate_output(
        output_dir=output_dir,
        manifest=manifest,
        observations=observations,
        prediction_key=prediction_key,
        frequency=args.validation_frequency,
    )
    files = write_evaluation_outputs(
        output_dir=output_dir,
        evaluation=evaluation,
        output_prefix=args.validation_output_prefix,
    )
    validation_manifest = {
        "target_path": target_path,
        "frequency": args.validation_frequency,
        "prediction_key": prediction_key,
        "csv_file": files["csv"],
        "summary_file": files["summary"],
    }
    manifest["validation"] = validation_manifest
    return validation_manifest


def _write_run_outputs(
    output_dir: Path,
    summary: dict[str, Any],
    manifest: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    dump_json(output_dir / "summary.json", summary)

    if args.calibration_targets and not args.online_calibration:
        target_path = _resolve_input_path(args.calibration_targets)
        targets = load_calibration_targets(target_path)
        calibration = build_calibration_result(
            summary=summary,
            targets=targets,
            min_scale=args.calibration_min_scale,
            max_scale=args.calibration_max_scale,
        )
        calibration["target_path"] = target_path

        calibrated_summary = build_calibrated_summary(summary, calibration)
        if args.apply_calibration:
            calibration["applied_window_files"] = apply_calibration_to_windows(
                output_dir=output_dir,
                manifest=manifest,
                reporting_rate_scale=calibration["reporting_rate_scale"],
            )
            dump_json(output_dir / "summary_calibrated.json", calibrated_summary)

        manifest["calibration"] = {
            "target_path": target_path,
            "calibration_file": "calibration.json",
            "summary_calibrated_file": "summary_calibrated.json" if args.apply_calibration else None,
            "applied_to_window_chunks": bool(args.apply_calibration),
            "reported_calibrated_key": "reported_calibrated" if args.apply_calibration else None,
        }
        dump_json(output_dir / "calibration.json", calibration)

    _write_validation_outputs(output_dir, manifest, args)
    dump_json(output_dir / "manifest.json", manifest)


def _validate_cli_args(args: argparse.Namespace) -> None:
    if args.apply_calibration and not args.calibration_targets:
        raise ValueError("--apply-calibration requires --calibration-targets")
    if args.online_calibration and not args.calibration_targets:
        raise ValueError("--online-calibration requires --calibration-targets")
    if args.online_calibration and args.apply_calibration:
        raise ValueError("--online-calibration and --apply-calibration both write reported_calibrated; use only one")
    if args.online_calibration and args.validate_only:
        raise ValueError("--online-calibration runs during simulation and cannot be combined with --validate-only")
    if args.online_calibration and args.resume_checkpoint:
        raise ValueError("--online-calibration is not resume-safe yet; rerun from the beginning or disable it")
    if args.finalize_online_calibration and not args.online_calibration:
        raise ValueError("--finalize-online-calibration requires --online-calibration")
    if args.calibration_min_scale < 0.0 or not math.isfinite(args.calibration_min_scale):
        raise ValueError("--calibration-min-scale must be finite and >= 0")
    if args.calibration_max_scale < args.calibration_min_scale or not math.isfinite(args.calibration_max_scale):
        raise ValueError("--calibration-max-scale must be finite and >= --calibration-min-scale")
    if args.validate_only and not args.validation_targets:
        raise ValueError("--validate-only requires --validation-targets")
    if args.validation_targets and not args.validation_output_prefix:
        raise ValueError("--validation-output-prefix must not be empty")


def _load_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Could not find manifest for validation: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def run_validation_only(args: argparse.Namespace) -> None:
    output_dir = Path(_resolve_output_path(args.output_dir))
    manifest = _load_manifest(output_dir)
    _write_validation_outputs(output_dir, manifest, args)
    dump_json(output_dir / "manifest.json", manifest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPU-oriented vectorized multi-disease mosquito-borne simulator")
    parser.add_argument("--static-path", default="data/static_layers_0_05.nc4")
    parser.add_argument("--climate-dir", default="data/climate")
    parser.add_argument("--output-dir", default="output/mosq_vectorized_gpu")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--lat-min", type=float, default=None)
    parser.add_argument("--lat-max", type=float, default=None)
    parser.add_argument("--lon-min", type=float, default=None)
    parser.add_argument("--lon-max", type=float, default=None)
    parser.add_argument("--start-date", default=None, help="Climate start date: YYYY, YYYYMMDD, or YYYY-MM-DD")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--downsample-factor", type=int, default=1, help="Factor by which to downsample the grid (e.g. 10 for 10x larger cells)")
    parser.add_argument("--cpu", action="store_true", help="Force NumPy backend")
    parser.add_argument("--allow-cpu-fallback", action="store_true", help="Allow falling back to NumPy when GPU backend is unavailable")
    parser.add_argument("--climate-window-days", type=int, default=7)
    parser.add_argument("--climate-memory-days", type=int, default=7, help="Rolling climate history used by aquatic and vector dynamics")
    parser.add_argument("--temperature-units", default=None, help="Explicit temperature units override; defaults to dataset metadata")
    parser.add_argument("--precipitation-units", default=None, help="Explicit precipitation units override; defaults to dataset metadata")
    parser.add_argument("--humidity-units", default=None, help="Explicit humidity units override; defaults to dataset metadata")
    parser.add_argument("--checkpoint-interval-windows", type=int, default=0, help="Write a restart checkpoint every N windows; 0 disables checkpoints")
    parser.add_argument("--resume-checkpoint", default=None, help="Resume from a checkpoint written by a prior run")
    parser.add_argument("--host-scale", type=float, default=1.0)
    parser.add_argument("--mosquito-scale", type=float, default=0.03)
    parser.add_argument("--min-population", type=float, default=1.0)
    parser.add_argument("--host-transmission-coeff", type=float, default=0.11)
    parser.add_argument("--vector-transmission-coeff", type=float, default=0.14)
    parser.add_argument(
        "--host-transmission-scale",
        type=float,
        nargs="+",
        default=None,
        help="Per-virus host infection multiplier; provide one value or five values in virus order",
    )
    parser.add_argument(
        "--vector-transmission-scale",
        type=float,
        nargs="+",
        default=None,
        help="Per-virus vector infection multiplier; provide one value or five values in virus order",
    )
    parser.add_argument(
        "--disable-default-seeds",
        action="store_true",
        help="Do not apply the built-in outbreak seed records from seed_data.py",
    )
    parser.add_argument(
        "--initial-state",
        default=None,
        help="CSV used once at day 0 to initialize exposed/infectious hosts and infectious vectors",
    )
    parser.add_argument(
        "--initial-state-window-days",
        type=int,
        default=365,
        help="Fallback duration for converting reported initial-state cases to active E/I compartments",
    )
    parser.add_argument(
        "--initial-vector-seed-rate",
        type=float,
        default=1e-4,
        help="Fallback fraction of suitable adult vectors initialized as infectious for initial-state rows",
    )
    parser.add_argument(
        "--initial-state-include-reported",
        action="store_true",
        help="Add initial-state reported cases to summary totals; disabled by default for backtests",
    )
    parser.add_argument(
        "--importation-targets",
        default=None,
        help="CSV/JSON observed cases used as ongoing external infection pressure",
    )
    parser.add_argument(
        "--importation-frequency",
        choices=("year", "month", "day"),
        default="year",
        help="Period granularity in --importation-targets",
    )
    parser.add_argument(
        "--importation-fraction",
        type=float,
        default=0.0,
        help="Fraction of observed cases injected as daily external exposed hosts; 0 disables",
    )
    parser.add_argument(
        "--seasonal-seed-cases",
        type=float,
        nargs="+",
        default=None,
        help="External exposed hosts injected once per year; provide one value or one per virus",
    )
    parser.add_argument("--seasonal-seed-month", type=int, default=1)
    parser.add_argument("--seasonal-seed-day", type=int, default=1)
    parser.add_argument(
        "--reservoir-force-scale",
        type=float,
        default=0.0,
        help="Non-human reservoir mosquito infection force for west_nile and japanese_encephalitis",
    )
    parser.add_argument(
        "--virus-reservoir-force-scale",
        type=float,
        nargs="+",
        default=None,
        help="Per-virus non-human reservoir force; provide one value or five values in virus order",
    )
    parser.add_argument(
        "--disease-seasonality-preset",
        choices=("none", "ecology"),
        default="none",
        help="Enable disease-specific seasonal transmission multipliers",
    )
    parser.add_argument(
        "--virus-seasonal-peak-month",
        type=float,
        nargs="+",
        default=None,
        help="Per-virus seasonal peak month; provide one value or five values in virus order",
    )
    parser.add_argument(
        "--virus-seasonal-amplitude",
        type=float,
        nargs="+",
        default=None,
        help="Per-virus seasonal multiplier amplitude; provide one value or five values in virus order",
    )
    parser.add_argument(
        "--virus-seasonal-floor",
        type=float,
        nargs="+",
        default=None,
        help="Per-virus minimum seasonal multiplier; provide one value or five values in virus order",
    )
    parser.add_argument("--virus-aedes-scale", type=float, nargs="+", default=None, help="Per-virus Aedes vector multiplier")
    parser.add_argument("--virus-culex-scale", type=float, nargs="+", default=None, help="Per-virus Culex vector multiplier")
    parser.add_argument("--virus-other-vector-scale", type=float, nargs="+", default=None, help="Per-virus non-Aedes/Culex vector multiplier")
    parser.add_argument("--virus-urban-preference", type=float, nargs="+", default=None, help="Per-virus urban habitat preference weight")
    parser.add_argument("--virus-wetland-preference", type=float, nargs="+", default=None, help="Per-virus wetland/rice-water habitat preference weight")
    parser.add_argument("--virus-forest-preference", type=float, nargs="+", default=None, help="Per-virus forest/sylvatic habitat preference weight")
    parser.add_argument("--virus-rural-preference", type=float, nargs="+", default=None, help="Per-virus rural habitat preference weight")
    parser.add_argument("--virus-habitat-floor", type=float, nargs="+", default=None, help="Per-virus minimum habitat multiplier")
    parser.add_argument("--virus-precipitation-sensitivity", type=float, nargs="+", default=None, help="Per-virus rain/standing-water response")
    parser.add_argument("--virus-humidity-sensitivity", type=float, nargs="+", default=None, help="Per-virus humidity response")
    parser.add_argument("--immunity-waning-rate", type=float, nargs="+", default=None, help="Daily recovered-to-susceptible rate; provide one value or one per virus")
    parser.add_argument("--host-replenishment-rate", type=float, default=0.0, help="Daily death-to-susceptible replacement rate")
    parser.add_argument("--spatial-diffusion-rate", type=float, default=0.0, help="Neighbor mixing rate for infectious pressure")
    parser.add_argument(
        "--reporting-rate-scale",
        type=float,
        nargs="+",
        default=None,
        help="Multiplicative reporting-rate scale; provide one value or one value per virus",
    )
    parser.add_argument(
        "--calibration-targets",
        default=None,
        help="CSV/JSON target reported-case totals used to compute per-virus calibration scales",
    )
    parser.add_argument(
        "--calibration-frequency",
        choices=("year", "month", "day"),
        default="year",
        help="Period granularity in --calibration-targets when --online-calibration is enabled",
    )
    parser.add_argument("--calibration-min-scale", type=float, default=0.0)
    parser.add_argument("--calibration-max-scale", type=float, default=100.0)
    parser.add_argument(
        "--apply-calibration",
        action="store_true",
        help="Write summary_calibrated.json and add reported_calibrated arrays to window chunks",
    )
    parser.add_argument(
        "--online-calibration",
        action="store_true",
        help="Update per-virus reported-case calibration factors after every simulation window",
    )
    parser.add_argument(
        "--finalize-online-calibration",
        action="store_true",
        help="For backtests/training, rewrite window chunks with the final scale for each calibration period",
    )
    parser.add_argument(
        "--validation-targets",
        default=None,
        help="CSV/JSON historical observed cases used to evaluate predictions",
    )
    parser.add_argument(
        "--validation-frequency",
        choices=("year", "month", "day"),
        default="year",
        help="Time period used to align predictions and historical observations",
    )
    parser.add_argument(
        "--validation-prediction-key",
        default=None,
        help="Window array to validate; defaults to reported_calibrated when calibration is applied, otherwise reported",
    )
    parser.add_argument("--validation-output-prefix", default="validation")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Evaluate existing files in --output-dir and skip simulation",
    )
    parser.add_argument("--multi-gpu", action="store_true", help="Enable multi-GPU distributed execution")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=None, help="List of GPU IDs to use (e.g., 0 1 2 3)")
    parser.add_argument("--grid-rows", type=int, default=1, help="Number of rows in the GPU grid partition")
    parser.add_argument("--grid-cols", type=int, default=None, help="Number of columns in the GPU grid partition (default: auto)")
    parser.add_argument("--climate-cache-dir", default=None, help="Directory for normalized climate shard cache files")
    parser.add_argument("--disable-climate-cache", action="store_true", help="Disable normalized climate shard caching")
    parser.add_argument("--prefetch-windows", type=int, default=2, help="How many climate windows to prefetch ahead in auto mode")
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=0,
        help="Worker processes for aggressive climate prebuild; 0 selects an automatic value",
    )
    parser.add_argument(
        "--preprocess-mode",
        choices=("off", "auto", "aggressive"),
        default="auto",
        help="How aggressively to prebuild climate window cache bundles before simulation starts",
    )
    return parser


def build_distributed_config(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        static_path=_resolve_input_path(args.static_path),
        climate_dir=_resolve_input_path(args.climate_dir),
        output_dir=_resolve_output_path(args.output_dir),
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        start_date=args.start_date,
        days=args.days,
        climate_window_days=args.climate_window_days,
        climate_memory_days=args.climate_memory_days,
        seed=args.seed,
        use_gpu=True,
        host_scale=args.host_scale,
        mosquito_scale=args.mosquito_scale,
        min_population=args.min_population,
        temperature_units=args.temperature_units,
        precipitation_units=args.precipitation_units,
        humidity_units=args.humidity_units,
        checkpoint_interval_windows=args.checkpoint_interval_windows,
        reporting_rate_scale=_reporting_rate_scale_arg(args),
        disable_default_seeds=args.disable_default_seeds,
        initial_state_path=_optional_input_path(args.initial_state),
        initial_state_window_days=args.initial_state_window_days,
        initial_vector_seed_rate=args.initial_vector_seed_rate,
        initial_state_include_reported=args.initial_state_include_reported,
        importation_targets=_optional_input_path(args.importation_targets),
        importation_frequency=args.importation_frequency,
        importation_fraction=args.importation_fraction,
        seasonal_seed_cases=_seasonal_seed_cases_arg(args),
        seasonal_seed_month=args.seasonal_seed_month,
        seasonal_seed_day=args.seasonal_seed_day,
        reservoir_force_scale=args.reservoir_force_scale,
        virus_reservoir_force_scale=_float_list_arg(args.virus_reservoir_force_scale),
        disease_seasonality_preset=args.disease_seasonality_preset,
        virus_seasonal_peak_month=_float_list_arg(args.virus_seasonal_peak_month),
        virus_seasonal_amplitude=_float_list_arg(args.virus_seasonal_amplitude),
        virus_seasonal_floor=_float_list_arg(args.virus_seasonal_floor),
        virus_aedes_scale=_float_list_arg(args.virus_aedes_scale),
        virus_culex_scale=_float_list_arg(args.virus_culex_scale),
        virus_other_vector_scale=_float_list_arg(args.virus_other_vector_scale),
        virus_urban_preference=_float_list_arg(args.virus_urban_preference),
        virus_wetland_preference=_float_list_arg(args.virus_wetland_preference),
        virus_forest_preference=_float_list_arg(args.virus_forest_preference),
        virus_rural_preference=_float_list_arg(args.virus_rural_preference),
        virus_habitat_floor=_float_list_arg(args.virus_habitat_floor),
        virus_precipitation_sensitivity=_float_list_arg(args.virus_precipitation_sensitivity),
        virus_humidity_sensitivity=_float_list_arg(args.virus_humidity_sensitivity),
        immunity_waning_rate=_float_list_arg(args.immunity_waning_rate),
        host_replenishment_rate=args.host_replenishment_rate,
        spatial_diffusion_rate=args.spatial_diffusion_rate,
        host_transmission_coeff=args.host_transmission_coeff,
        vector_transmission_coeff=args.vector_transmission_coeff,
        host_transmission_scale=_float_list_arg(args.host_transmission_scale),
        vector_transmission_scale=_float_list_arg(args.vector_transmission_scale),
    )


def _resolve_climate_cache_dir(args: argparse.Namespace, output_dir: str | Path) -> str | None:
    if args.disable_climate_cache:
        return None
    if args.climate_cache_dir:
        return _resolve_output_path(args.climate_cache_dir)
    return str((Path(output_dir) / "climate_cache").resolve())


def _window_cache_ready(
    climate_files: list[Any],
    static_grid: Any,
    start_day: int,
    window_days: int,
    cache_dir: str | None,
    temperature_units: str | None,
    precipitation_units: str | None,
    humidity_units: str | None,
) -> bool:
    if cache_dir is None:
        return False
    try:
        if climate_window_shard_available(
            climate_files=climate_files,
            static_grid=static_grid,
            cache_dir=cache_dir,
            start_day=start_day,
            window_days=window_days,
            temperature_units=temperature_units,
            precipitation_units=precipitation_units,
            humidity_units=humidity_units,
        ):
            return True
        window_cache = climate_window_cache_path(
            climate_files=climate_files,
            static_grid=static_grid,
            cache_dir=cache_dir,
            start_day=start_day,
            window_days=window_days,
            temperature_units=temperature_units,
            precipitation_units=precipitation_units,
            humidity_units=humidity_units,
        )
    except IndexError:
        return False
    if window_cache.exists():
        return True
    selected = climate_files[start_day : start_day + window_days]
    if not selected:
        return False
    return all(
        climate_day_cache_path(
            record=record,
            static_grid=static_grid,
            cache_dir=cache_dir,
            temperature_units=temperature_units,
            precipitation_units=precipitation_units,
            humidity_units=humidity_units,
        ).exists()
        for record in selected
    )


def _prebuild_window_target_count(
    preprocess_mode: str,
    total_windows: int,
    prefetch_windows: int,
    multi_gpu: bool,
) -> int:
    if total_windows <= 0 or preprocess_mode == "off":
        return 0
    if preprocess_mode == "aggressive":
        return total_windows
    if multi_gpu:
        return min(total_windows, 1)
    return min(total_windows, max(int(prefetch_windows), 0) + 2)


def _resolve_preprocess_workers(
    requested_workers: int,
    preprocess_mode: str,
    total_windows: int,
    multi_gpu: bool,
) -> int:
    if preprocess_mode != "aggressive" or total_windows <= 1:
        return 1
    if requested_workers > 0:
        return max(1, min(int(requested_workers), total_windows))
    if multi_gpu:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, total_windows, 8))


def _prebuild_window_cache(
    climate_files: list[Any],
    static_grid: Any,
    window_starts: list[int],
    cfg: SimulationConfig,
    cache_dir: str | None,
    preprocess_mode: str,
    prefetch_windows: int,
    multi_gpu: bool,
    rank: int = 0,
    preprocess_workers: int = 0,
) -> int:
    target_count = _prebuild_window_target_count(preprocess_mode, len(window_starts), prefetch_windows, multi_gpu)
    if cache_dir is None or target_count == 0:
        return 0

    selected_starts = window_starts[:target_count]
    worker_count = _resolve_preprocess_workers(preprocess_workers, preprocess_mode, len(selected_starts), multi_gpu)
    progress = None
    if TQDM_AVAILABLE and (not multi_gpu or rank == 0):
        progress = tqdm(total=len(selected_starts), desc="Prebuild climate cache", unit="window")

    def _update_progress(_start_day: int, _window_days: int, _path: Any) -> None:
        if progress is not None:
            progress.update(1)

    try:
        if preprocess_mode == "aggressive":
            return prebuild_climate_shard(
                climate_files=climate_files,
                static_grid=static_grid,
                window_starts=selected_starts,
                window_days=cfg.climate_window_days,
                temperature_units=cfg.temperature_units,
                precipitation_units=cfg.precipitation_units,
                humidity_units=cfg.humidity_units,
                cache_dir=cache_dir,
                progress_callback=_update_progress,
                num_workers=worker_count,
            )
        return prebuild_climate_windows(
            climate_files=climate_files,
            static_grid=static_grid,
            window_starts=selected_starts,
            window_days=cfg.climate_window_days,
            temperature_units=cfg.temperature_units,
            precipitation_units=cfg.precipitation_units,
            humidity_units=cfg.humidity_units,
            cache_dir=cache_dir,
            progress_callback=_update_progress,
        )
    finally:
        if progress is not None:
            progress.close()


def run_single_gpu(args: argparse.Namespace) -> None:
    cfg = SimulationConfig(
        static_path=_resolve_input_path(args.static_path),
        climate_dir=_resolve_input_path(args.climate_dir),
        output_dir=_resolve_output_path(args.output_dir),
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        start_date=args.start_date,
        days=args.days,
        climate_window_days=args.climate_window_days,
        climate_memory_days=args.climate_memory_days,
        seed=args.seed,
        use_gpu=not args.cpu,
        host_scale=args.host_scale,
        mosquito_scale=args.mosquito_scale,
        min_population=args.min_population,
        temperature_units=args.temperature_units,
        precipitation_units=args.precipitation_units,
        humidity_units=args.humidity_units,
        checkpoint_interval_windows=args.checkpoint_interval_windows,
        reporting_rate_scale=_reporting_rate_scale_arg(args),
        disable_default_seeds=args.disable_default_seeds,
        initial_state_path=_optional_input_path(args.initial_state),
        initial_state_window_days=args.initial_state_window_days,
        initial_vector_seed_rate=args.initial_vector_seed_rate,
        initial_state_include_reported=args.initial_state_include_reported,
        importation_targets=_optional_input_path(args.importation_targets),
        importation_frequency=args.importation_frequency,
        importation_fraction=args.importation_fraction,
        seasonal_seed_cases=_seasonal_seed_cases_arg(args),
        seasonal_seed_month=args.seasonal_seed_month,
        seasonal_seed_day=args.seasonal_seed_day,
        reservoir_force_scale=args.reservoir_force_scale,
        virus_reservoir_force_scale=_float_list_arg(args.virus_reservoir_force_scale),
        disease_seasonality_preset=args.disease_seasonality_preset,
        virus_seasonal_peak_month=_float_list_arg(args.virus_seasonal_peak_month),
        virus_seasonal_amplitude=_float_list_arg(args.virus_seasonal_amplitude),
        virus_seasonal_floor=_float_list_arg(args.virus_seasonal_floor),
        virus_aedes_scale=_float_list_arg(args.virus_aedes_scale),
        virus_culex_scale=_float_list_arg(args.virus_culex_scale),
        virus_other_vector_scale=_float_list_arg(args.virus_other_vector_scale),
        virus_urban_preference=_float_list_arg(args.virus_urban_preference),
        virus_wetland_preference=_float_list_arg(args.virus_wetland_preference),
        virus_forest_preference=_float_list_arg(args.virus_forest_preference),
        virus_rural_preference=_float_list_arg(args.virus_rural_preference),
        virus_habitat_floor=_float_list_arg(args.virus_habitat_floor),
        virus_precipitation_sensitivity=_float_list_arg(args.virus_precipitation_sensitivity),
        virus_humidity_sensitivity=_float_list_arg(args.virus_humidity_sensitivity),
        immunity_waning_rate=args.immunity_waning_rate,
        host_replenishment_rate=args.host_replenishment_rate,
        spatial_diffusion_rate=args.spatial_diffusion_rate,
        host_transmission_coeff=args.host_transmission_coeff,
        vector_transmission_coeff=args.vector_transmission_coeff,
        host_transmission_scale=_float_list_arg(args.host_transmission_scale),
        vector_transmission_scale=_float_list_arg(args.vector_transmission_scale),
    )
    cfg.validate()

    backend = get_backend(force_gpu=cfg.use_gpu, require_gpu=cfg.use_gpu and not args.allow_cpu_fallback)
    static_grid = load_static_grid(
        static_path=cfg.static_path,
        lat_min=cfg.lat_min,
        lat_max=cfg.lat_max,
        lon_min=cfg.lon_min,
        lon_max=cfg.lon_max,
        min_population=cfg.min_population,
        downsample_factor=args.downsample_factor,
    )
    climate_files = _select_climate_files_from_start(list_climate_files(cfg.climate_dir), cfg.start_date)
    validation = validate_inputs(static_grid, climate_files, requested_days=cfg.days)
    if cfg.days > len(climate_files):
        raise ValueError(f"Requested {cfg.days} days but only found {len(climate_files)} climate files.")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    climate_cache_dir = _resolve_climate_cache_dir(args, output_dir)
    gpu_window_pipeline = SingleGPUWindowPipeline(backend)

    simulator = VectorizedABMSimulator(cfg, static_grid, backend)
    online_calibrator = _build_online_window_calibrator(args)
    start_day_offset = 0
    window_index_offset = 0
    if args.resume_checkpoint:
        checkpoint = WindowWriter.read_checkpoint(args.resume_checkpoint)
        start_day_offset, window_index_offset = simulator.load_checkpoint(checkpoint)
    window_starts = list(range(start_day_offset, cfg.days, cfg.climate_window_days))

    input_audit = {
        "static": audit_static_dataset(cfg.static_path),
        "climate_sample": audit_climate_dataset(climate_files[0].path),
        "selected_start_date": climate_files[0].date.strftime("%Y-%m-%d"),
        "selected_end_date": climate_files[cfg.days - 1].date.strftime("%Y-%m-%d"),
    }
    write_metadata(output_dir, cfg, static_grid, backend, validation, input_audit, simulator)

    writer = WindowWriter(output_dir)
    manifest = {
        "backend": backend.name,
        "days": cfg.days,
        "window_days": cfg.climate_window_days,
        "climate_memory_days": cfg.climate_memory_days,
        "temperature_units": cfg.temperature_units,
        "precipitation_units": cfg.precipitation_units,
        "humidity_units": cfg.humidity_units,
        "reporting_rate_scale": cfg.reporting_rate_scale,
        "disable_default_seeds": cfg.disable_default_seeds,
        "initial_state_path": cfg.initial_state_path,
        "initial_state_window_days": cfg.initial_state_window_days,
        "initial_vector_seed_rate": cfg.initial_vector_seed_rate,
        "initial_state_include_reported": cfg.initial_state_include_reported,
        "importation_targets": cfg.importation_targets,
        "importation_frequency": cfg.importation_frequency,
        "importation_fraction": cfg.importation_fraction,
        "seasonal_seed_cases": cfg.seasonal_seed_cases,
        "seasonal_seed_month": cfg.seasonal_seed_month,
        "seasonal_seed_day": cfg.seasonal_seed_day,
        "reservoir_force_scale": cfg.reservoir_force_scale,
        "virus_reservoir_force_scale": cfg.virus_reservoir_force_scale,
        "disease_seasonality_preset": cfg.disease_seasonality_preset,
        "virus_seasonal_peak_month": cfg.virus_seasonal_peak_month,
        "virus_seasonal_amplitude": cfg.virus_seasonal_amplitude,
        "virus_seasonal_floor": cfg.virus_seasonal_floor,
        "virus_aedes_scale": cfg.virus_aedes_scale,
        "virus_culex_scale": cfg.virus_culex_scale,
        "virus_other_vector_scale": cfg.virus_other_vector_scale,
        "virus_urban_preference": cfg.virus_urban_preference,
        "virus_wetland_preference": cfg.virus_wetland_preference,
        "virus_forest_preference": cfg.virus_forest_preference,
        "virus_rural_preference": cfg.virus_rural_preference,
        "virus_habitat_floor": cfg.virus_habitat_floor,
        "virus_precipitation_sensitivity": cfg.virus_precipitation_sensitivity,
        "virus_humidity_sensitivity": cfg.virus_humidity_sensitivity,
        "immunity_waning_rate": cfg.immunity_waning_rate,
        "host_replenishment_rate": cfg.host_replenishment_rate,
        "spatial_diffusion_rate": cfg.spatial_diffusion_rate,
        "host_transmission_coeff": cfg.host_transmission_coeff,
        "vector_transmission_coeff": cfg.vector_transmission_coeff,
        "host_transmission_scale": cfg.host_transmission_scale,
        "vector_transmission_scale": cfg.vector_transmission_scale,
        "climate_cache_dir": climate_cache_dir,
        "prefetch_windows": max(int(args.prefetch_windows), 0),
        "preprocess_workers": _resolve_preprocess_workers(args.preprocess_workers, args.preprocess_mode, len(window_starts), False),
        "preprocess_mode": args.preprocess_mode,
        "gpu_async_window_staging": bool(gpu_window_pipeline.enabled),
        "window_files": [],
        "checkpoint_files": [],
    }
    if online_calibrator is not None:
        manifest["online_calibration"] = online_calibrator.manifest_header()
    if args.resume_checkpoint:
        manifest["resumed_from"] = str(Path(args.resume_checkpoint).resolve())
    manifest["prebuilt_window_caches"] = _prebuild_window_cache(
        climate_files=climate_files,
        static_grid=static_grid,
        window_starts=window_starts,
        cfg=cfg,
        cache_dir=climate_cache_dir,
        preprocess_mode=args.preprocess_mode,
        prefetch_windows=args.prefetch_windows,
        multi_gpu=False,
        preprocess_workers=args.preprocess_workers,
    )
    pbar = tqdm(total=cfg.days, desc="Overall progress", unit="day") if TQDM_AVAILABLE else None

    def _load_window(start_day: int):
        return load_climate_window_cpu(
            climate_files=climate_files,
            static_grid=static_grid,
            start_day=start_day,
            window_days=min(cfg.climate_window_days, cfg.days - start_day),
            temperature_units=cfg.temperature_units,
            precipitation_units=cfg.precipitation_units,
            humidity_units=cfg.humidity_units,
            cache_dir=climate_cache_dir,
        )

    def _can_prefetch_window(start_day: int) -> bool:
        return _window_cache_ready(
            climate_files=climate_files,
            static_grid=static_grid,
            start_day=start_day,
            window_days=min(cfg.climate_window_days, cfg.days - start_day),
            cache_dir=climate_cache_dir,
            temperature_units=cfg.temperature_units,
            precipitation_units=cfg.precipitation_units,
            humidity_units=cfg.humidity_units,
        )

    cpu_window_iter = ClimateWindowPrefetcher(
        window_starts,
        _load_window,
        args.prefetch_windows,
        use_threads=True,
        can_prefetch_fn=_can_prefetch_window,
    )

    for local_window_index, (start_day, climate_window_cpu, climate_window) in enumerate(
        gpu_window_pipeline.iter_ready(cpu_window_iter)
    ):
        window_index = window_index_offset + local_window_index
        results = simulator.run_window(climate_window)
        payload: dict[str, np.ndarray] = {"dates": np.asarray(climate_window_cpu.dates)}
        payload.update(results)
        calibration_record = _apply_online_calibration_to_payload(payload, online_calibrator, window_index)
        file_name = writer.write_window(window_index, payload)
        window_entry = {
            "file": file_name,
            "start_date": climate_window_cpu.dates[0],
            "end_date": climate_window_cpu.dates[-1],
        }
        if calibration_record is not None:
            window_entry["online_calibration_scale"] = calibration_record["scale"]
        manifest["window_files"].append(window_entry)
        if pbar:
            pbar.update(len(climate_window_cpu.dates))
        if cfg.checkpoint_interval_windows and (local_window_index + 1) % cfg.checkpoint_interval_windows == 0:
            checkpoint_name = writer.write_checkpoint(
                window_index,
                simulator.export_checkpoint(
                    next_start_day=start_day + len(climate_window_cpu.dates),
                    next_window_index=window_index + 1,
                ),
            )
            manifest["checkpoint_files"].append(
                {
                    "file": checkpoint_name,
                    "next_start_day": start_day + len(climate_window_cpu.dates),
                    "next_window_index": window_index + 1,
                }
            )

    if pbar:
        pbar.close()
    summary = simulator.final_summary()
    if args.finalize_online_calibration:
        _finalize_online_calibration_windows(output_dir, manifest, online_calibrator)
    _attach_online_calibration_outputs(output_dir, summary, manifest, online_calibrator)
    _write_run_outputs(output_dir, summary, manifest, args)


def run_multi_gpu(args: argparse.Namespace) -> None:
    from .simulation import DistributedVectorizedABMSimulator

    cfg = build_distributed_config(args)
    cfg.validate()
    if args.resume_checkpoint:
        raise RuntimeError("Resuming from checkpoints is not yet supported for multi-GPU runs.")

    reducer = DistributedReducer(enabled=True)
    reducer.initialize()

    try:
        backend = get_multi_gpu_backend(
            gpu_ids=args.gpu_ids,
            grid_rows=args.grid_rows,
            grid_cols=args.grid_cols,
            force_gpu=True,
        )
        if not isinstance(backend, MultiGPUBackend):
            raise RuntimeError("Multi-GPU execution requested, but a multi-GPU backend was not created.")
        backend.activate_device()
        rank = backend.rank

        if not is_distributed_run():
            raise RuntimeError(
                "Multi-GPU execution requires one process per GPU. Launch with a distributed runner "
                "that sets LOCAL_RANK/RANK/WORLD_SIZE for each visible GPU."
            )
        if get_world_size() != backend.n_gpus:
            raise RuntimeError(
                f"WORLD_SIZE ({get_world_size()}) does not match visible GPU count ({backend.n_gpus}). "
                "Launch exactly one rank per visible GPU for this code path."
            )
        if reducer.world_size != backend.n_gpus:
            raise RuntimeError(
                f"torch.distributed world size ({reducer.world_size}) does not match visible GPU count ({backend.n_gpus})."
            )

        global_static_grid = load_static_grid(
            static_path=cfg.static_path,
            lat_min=cfg.lat_min,
            lat_max=cfg.lat_max,
            lon_min=cfg.lon_min,
            lon_max=cfg.lon_max,
            min_population=cfg.min_population,
            downsample_factor=args.downsample_factor,
        )

        climate_files = _select_climate_files_from_start(list_climate_files(cfg.climate_dir), cfg.start_date)
        validation = validate_inputs(global_static_grid, climate_files, requested_days=cfg.days)
        if cfg.days > len(climate_files):
            raise ValueError(f"Requested {cfg.days} days but only found {len(climate_files)} climate files.")

        grid_rows = args.grid_rows
        grid_cols = args.grid_cols
        if grid_cols is None and backend.n_gpus > 1:
            grid_cols = int(math.sqrt(backend.n_gpus))
            grid_rows = max(backend.n_gpus // max(grid_cols, 1), 1)
            if grid_rows * grid_cols != backend.n_gpus:
                grid_rows = 1
                grid_cols = backend.n_gpus
        elif grid_cols is None:
            grid_cols = 1

        spatial_decomp = SpatialDecomposition(
            n_gpus=backend.n_gpus,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            total_lats=global_static_grid.lat.size,
            total_lons=global_static_grid.lon.size,
            active_mask_2d=global_static_grid.active_mask_2d,
        )
        local_static_grid = partition_static_grid(global_static_grid, spatial_decomp, rank)

        output_root = Path(cfg.output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        climate_cache_dir = _resolve_climate_cache_dir(args, output_root)

        simulator = DistributedVectorizedABMSimulator(
            cfg=cfg,
            static_grid=local_static_grid,
            backend=backend,
            spatial_decomp=spatial_decomp,
            rank=rank,
        )
        online_calibrator = _build_online_window_calibrator(args) if rank == 0 else None

        active_cells_per_rank = reducer.gather_object(int(local_static_grid.population.size))
        writer = WindowWriter(output_root) if rank == 0 else None
        checkpoint_writer = WindowWriter(output_root / f"rank_{rank}") if cfg.checkpoint_interval_windows else None

        if rank == 0:
            input_audit = {
                "static": audit_static_dataset(cfg.static_path),
                "climate_sample": audit_climate_dataset(climate_files[0].path),
                "selected_start_date": climate_files[0].date.strftime("%Y-%m-%d"),
                "selected_end_date": climate_files[cfg.days - 1].date.strftime("%Y-%m-%d"),
            }
            write_metadata(output_root, cfg, global_static_grid, backend, validation, input_audit, simulator)

        manifest = {
            "backend": backend.name,
            "rank_count": backend.n_gpus,
            "grid_rows": grid_rows,
            "grid_cols": grid_cols,
            "days": cfg.days,
            "window_days": cfg.climate_window_days,
            "climate_memory_days": cfg.climate_memory_days,
            "temperature_units": cfg.temperature_units,
            "precipitation_units": cfg.precipitation_units,
            "humidity_units": cfg.humidity_units,
            "reporting_rate_scale": cfg.reporting_rate_scale,
            "disable_default_seeds": cfg.disable_default_seeds,
            "initial_state_path": cfg.initial_state_path,
            "initial_state_window_days": cfg.initial_state_window_days,
            "initial_vector_seed_rate": cfg.initial_vector_seed_rate,
            "initial_state_include_reported": cfg.initial_state_include_reported,
            "importation_targets": cfg.importation_targets,
            "importation_frequency": cfg.importation_frequency,
            "importation_fraction": cfg.importation_fraction,
            "seasonal_seed_cases": cfg.seasonal_seed_cases,
            "seasonal_seed_month": cfg.seasonal_seed_month,
            "seasonal_seed_day": cfg.seasonal_seed_day,
            "reservoir_force_scale": cfg.reservoir_force_scale,
            "virus_reservoir_force_scale": cfg.virus_reservoir_force_scale,
            "disease_seasonality_preset": cfg.disease_seasonality_preset,
            "virus_seasonal_peak_month": cfg.virus_seasonal_peak_month,
            "virus_seasonal_amplitude": cfg.virus_seasonal_amplitude,
            "virus_seasonal_floor": cfg.virus_seasonal_floor,
            "virus_aedes_scale": cfg.virus_aedes_scale,
            "virus_culex_scale": cfg.virus_culex_scale,
            "virus_other_vector_scale": cfg.virus_other_vector_scale,
            "virus_urban_preference": cfg.virus_urban_preference,
            "virus_wetland_preference": cfg.virus_wetland_preference,
            "virus_forest_preference": cfg.virus_forest_preference,
            "virus_rural_preference": cfg.virus_rural_preference,
            "virus_habitat_floor": cfg.virus_habitat_floor,
            "virus_precipitation_sensitivity": cfg.virus_precipitation_sensitivity,
            "virus_humidity_sensitivity": cfg.virus_humidity_sensitivity,
            "immunity_waning_rate": cfg.immunity_waning_rate,
            "host_replenishment_rate": cfg.host_replenishment_rate,
            "spatial_diffusion_rate": cfg.spatial_diffusion_rate,
            "host_transmission_coeff": cfg.host_transmission_coeff,
            "vector_transmission_coeff": cfg.vector_transmission_coeff,
            "host_transmission_scale": cfg.host_transmission_scale,
            "vector_transmission_scale": cfg.vector_transmission_scale,
            "climate_cache_dir": climate_cache_dir,
            "prefetch_windows": max(int(args.prefetch_windows), 0),
            "preprocess_mode": args.preprocess_mode,
            "active_cells_per_rank": active_cells_per_rank if rank == 0 else None,
            "window_files": [],
            "checkpoint_files": [],
        }
        if rank == 0 and online_calibrator is not None:
            manifest["online_calibration"] = online_calibrator.manifest_header()

        total_days = cfg.days
        window_starts = list(range(0, cfg.days, cfg.climate_window_days))
        manifest["preprocess_workers"] = _resolve_preprocess_workers(
            args.preprocess_workers,
            args.preprocess_mode,
            len(window_starts),
            True,
        )
        prebuilt_windows = _prebuild_window_cache(
            climate_files=climate_files,
            static_grid=local_static_grid,
            window_starts=window_starts,
            cfg=cfg,
            cache_dir=climate_cache_dir,
            preprocess_mode=args.preprocess_mode,
            prefetch_windows=args.prefetch_windows,
            multi_gpu=True,
            rank=rank,
            preprocess_workers=args.preprocess_workers,
        )
        if rank == 0:
            manifest["prebuilt_window_caches"] = prebuilt_windows

        def _load_local_window(start_day: int):
            return load_climate_window_cpu(
                climate_files=climate_files,
                static_grid=local_static_grid,
                start_day=start_day,
                window_days=min(cfg.climate_window_days, cfg.days - start_day),
                temperature_units=cfg.temperature_units,
                precipitation_units=cfg.precipitation_units,
                humidity_units=cfg.humidity_units,
                cache_dir=climate_cache_dir,
            )

        def _can_prefetch_window(start_day: int) -> bool:
            return _window_cache_ready(
                climate_files=climate_files,
                static_grid=local_static_grid,
                start_day=start_day,
                window_days=min(cfg.climate_window_days, cfg.days - start_day),
                cache_dir=climate_cache_dir,
                temperature_units=cfg.temperature_units,
                precipitation_units=cfg.precipitation_units,
                humidity_units=cfg.humidity_units,
            )

        pbar = tqdm(total=total_days, desc="Overall progress", unit="day") if TQDM_AVAILABLE and rank == 0 else None
        for local_window_index, (start_day, climate_window_cpu) in enumerate(
            ClimateWindowPrefetcher(
                window_starts,
                _load_local_window,
                args.prefetch_windows,
                use_threads=True,
                can_prefetch_fn=_can_prefetch_window,
            )
        ):
            window_index = local_window_index
            climate_window = climate_window_to_backend(climate_window_cpu, backend)
            local_results = simulator.run_window(climate_window)
            reduced_payload = reducer.reduce_window_payload(
                dates=climate_window_cpu.dates,
                local_results=local_results,
                active_cells=int(local_static_grid.population.size),
            )

            if checkpoint_writer is not None and cfg.checkpoint_interval_windows and (local_window_index + 1) % cfg.checkpoint_interval_windows == 0:
                checkpoint_writer.write_checkpoint(
                    window_index,
                    simulator.export_checkpoint(
                        next_start_day=start_day + len(climate_window_cpu.dates),
                        next_window_index=window_index + 1,
                    ),
                )

            if rank == 0 and writer is not None and reduced_payload is not None:
                calibration_record = _apply_online_calibration_to_payload(
                    reduced_payload,
                    online_calibrator,
                    window_index,
                )
                file_name = writer.write_window(window_index, reduced_payload)
                window_entry = {
                    "file": file_name,
                    "start_date": climate_window_cpu.dates[0],
                    "end_date": climate_window_cpu.dates[-1],
                }
                if calibration_record is not None:
                    window_entry["online_calibration_scale"] = calibration_record["scale"]
                manifest["window_files"].append(window_entry)
                if checkpoint_writer is not None and cfg.checkpoint_interval_windows and (local_window_index + 1) % cfg.checkpoint_interval_windows == 0:
                    manifest["checkpoint_files"].append(
                        {
                            "window_index": window_index,
                            "next_start_day": start_day + len(climate_window_cpu.dates),
                            "next_window_index": window_index + 1,
                            "files": [
                                f"rank_{worker_rank}/checkpoints/checkpoint_{window_index:04d}.npz"
                                for worker_rank in range(backend.n_gpus)
                            ],
                        }
                    )
                if pbar:
                    pbar.update(len(climate_window_cpu.dates))

        if pbar:
            pbar.close()

        reduced_summary = reducer.reduce_summary_payload(simulator.final_summary())
        if rank == 0 and reduced_summary is not None:
            if args.finalize_online_calibration:
                _finalize_online_calibration_windows(output_root, manifest, online_calibrator)
            _attach_online_calibration_outputs(output_root, reduced_summary, manifest, online_calibrator)
            _write_run_outputs(output_root, reduced_summary, manifest, args)
        reducer.barrier()
    finally:
        reducer.finalize()


def main() -> None:
    args = build_parser().parse_args()
    _validate_cli_args(args)

    if args.validate_only:
        run_validation_only(args)
        return

    if args.multi_gpu or is_distributed_run():
        run_multi_gpu(args)
    else:
        run_single_gpu(args)


if __name__ == "__main__":
    main()
