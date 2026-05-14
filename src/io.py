from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from netCDF4 import Dataset

from .backend import Backend
from .constants import (
    CLIMATE_VARIABLE_ALIASES,
    DEFAULT_CLIMATE_WINDOW_DAYS,
    LCCS_SNOW,
    LCCS_TO_LANDTYPE,
    LCCS_WATER,
    MAX_CLIMATE_WINDOW_DAYS,
    SUPPORTED_STATIC_VARIABLES,
)

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


CLIMATE_FILE_RE = re.compile(r"MERRA2_(\d+)\.statD_2d_slv_Nx\.(\d{8})_0_05\.nc4$")


@dataclass(frozen=True)
class ClimateFileRecord:
    path: Path
    stream_id: int
    date: datetime


@dataclass(frozen=True)
class StaticGrid:
    lat: np.ndarray
    lon: np.ndarray
    active_mask_2d: np.ndarray
    active_linear_idx: np.ndarray
    active_lat: np.ndarray
    active_lon: np.ndarray
    land_type: np.ndarray
    population: np.ndarray
    elevation_m: np.ndarray
    runoff_coeff: np.ndarray
    bbox: tuple[float, float, float, float]
    shape_2d: tuple[int, int]
    index_slices: tuple[slice, slice]
    downsample_factor: int = 1


@dataclass(frozen=True)
class ClimateWindow:
    dates: list[str]
    temperature_c: Any
    precipitation_mm: Any
    humidity_pct: Any


@dataclass
class ClimateWindowTransfer:
    climate_window: ClimateWindow
    stream: Any = None
    pinned_refs: tuple[np.ndarray, ...] = ()

    def wait(self) -> ClimateWindow:
        if self.stream is not None:
            self.stream.synchronize()
        self.pinned_refs = ()
        return self.climate_window


def _find_slice(values: np.ndarray, lower: float | None, upper: float | None) -> slice:
    if lower is None or upper is None:
        return slice(None)

    lo = min(lower, upper)
    hi = max(lower, upper)
    ascending = bool(values[0] < values[-1])
    work = values if ascending else values[::-1]

    start = int(np.searchsorted(work, lo, side="left"))
    stop = int(np.searchsorted(work, hi, side="right"))
    if start == stop:
        start = max(start - 1, 0)
        stop = min(stop + 1, work.size)

    if ascending:
        return slice(start, stop)

    n = work.size
    return slice(n - stop, n - start)


def _read_variable(dataset: Dataset, name: str, fill_value: float | int | None = None) -> np.ndarray:
    var = dataset.variables[name][:]
    if hasattr(var, "filled"):
        if fill_value is None:
            fill_value = np.nan if np.issubdtype(var.dtype, np.floating) else 0
        return np.asarray(var.filled(fill_value))
    return np.asarray(var)


def _normalize_population(population: np.ndarray) -> np.ndarray:
    fill_mask = population <= -9999.0
    population = np.where(fill_mask, np.nan, population)
    population = np.where(np.isfinite(population), np.maximum(population, 0.0), np.nan)
    return population.astype(np.float32)


def _normalize_elevation(elevation: np.ndarray) -> np.ndarray:
    elevation = np.where(np.isfinite(elevation), elevation, 0.0)
    return np.clip(elevation, -11000.0, 9000.0).astype(np.float32)


def _map_land_type(landcover: np.ndarray) -> np.ndarray:
    mapped = np.full_like(landcover, 2, dtype=np.uint8)
    for raw_value, land_type in LCCS_TO_LANDTYPE.items():
        mapped[landcover == raw_value] = land_type
    return mapped


def _estimate_runoff(population: np.ndarray, land_type: np.ndarray) -> np.ndarray:
    urban_like = np.isin(land_type, [1, 2, 5]).astype(np.float32)
    wet_like = np.isin(land_type, [3, 4, 8]).astype(np.float32)
    pop_norm = np.clip(population / 250000.0, 0.0, 1.0)
    runoff = 0.12 + 0.42 * urban_like + 0.16 * wet_like + 0.12 * pop_norm
    return np.clip(runoff, 0.05, 0.95).astype(np.float32)


def audit_static_dataset(static_path: str | Path) -> dict[str, Any]:
    static_path = Path(static_path)
    with Dataset(static_path) as dataset:
        dims = {name: len(dim) for name, dim in dataset.dimensions.items()}
        variables = {}
        for name, var in dataset.variables.items():
            variables[name] = {
                "dims": list(var.dimensions),
                "shape": list(var.shape),
                "dtype": str(var.dtype),
                "units": getattr(var, "units", None),
            }
    return {"path": str(static_path), "dims": dims, "variables": variables}


def audit_climate_dataset(climate_path: str | Path) -> dict[str, Any]:
    climate_path = Path(climate_path)
    with Dataset(climate_path) as dataset:
        dims = {name: len(dim) for name, dim in dataset.dimensions.items()}
        variables = {}
        for name, var in dataset.variables.items():
            variables[name] = {
                "dims": list(var.dimensions),
                "shape": list(var.shape),
                "dtype": str(var.dtype),
                "units": getattr(var, "units", None),
            }
    return {"path": str(climate_path), "dims": dims, "variables": variables}


def load_static_grid(
    static_path: str | Path,
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    min_population: float = 1.0,
    downsample_factor: int = 1,
) -> StaticGrid:
    static_path = Path(static_path)
    with Dataset(static_path) as dataset:
        missing = [name for name in SUPPORTED_STATIC_VARIABLES if name not in dataset.variables]
        if missing:
            raise KeyError(f"Static dataset is missing required variables: {missing}")

        lat = _read_variable(dataset, "lat").astype(np.float32)
        lon = _read_variable(dataset, "lon").astype(np.float32)
        lat_slice = _find_slice(lat, lat_min, lat_max)
        lon_slice = _find_slice(lon, lon_min, lon_max)

        sub_lat = lat[lat_slice]
        sub_lon = lon[lon_slice]
        population = _normalize_population(_read_variable(dataset, "population")[lat_slice, lon_slice])
        elevation = _normalize_elevation(_read_variable(dataset, "elevation")[lat_slice, lon_slice])
        landcover = _read_variable(dataset, "landcover", fill_value=0)[lat_slice, lon_slice].astype(np.uint8)

    if downsample_factor > 1:
        h, w = population.shape
        h_new, w_new = h // downsample_factor, w // downsample_factor
        
        sub_lat = sub_lat[:h_new * downsample_factor].reshape(h_new, downsample_factor).mean(axis=1)
        sub_lon = sub_lon[:w_new * downsample_factor].reshape(w_new, downsample_factor).mean(axis=1)
        
        population = population[:h_new * downsample_factor, :w_new * downsample_factor]
        population = population.reshape(h_new, downsample_factor, w_new, downsample_factor).sum(axis=(1,3))
        
        elevation = elevation[:h_new * downsample_factor, :w_new * downsample_factor]
        elevation = elevation.reshape(h_new, downsample_factor, w_new, downsample_factor).mean(axis=(1,3))
        
        landcover = landcover[:h_new * downsample_factor:downsample_factor, :w_new * downsample_factor:downsample_factor]
        
        actual_lat_start = lat_slice.start if lat_slice.start is not None else 0
        lat_slice = slice(actual_lat_start, actual_lat_start + h_new * downsample_factor)
        
        actual_lon_start = lon_slice.start if lon_slice.start is not None else 0
        lon_slice = slice(actual_lon_start, actual_lon_start + w_new * downsample_factor)

    land_type = _map_land_type(landcover)
    active_mask = np.isfinite(population) & (population >= float(min_population)) & ~np.isin(landcover, list(LCCS_WATER | LCCS_SNOW))
    active_linear_idx = np.flatnonzero(active_mask.reshape(-1))
    active_lat_idx, active_lon_idx = np.nonzero(active_mask)
    runoff = _estimate_runoff(np.nan_to_num(population, nan=0.0), land_type)
    bbox = (float(sub_lat.min()), float(sub_lat.max()), float(sub_lon.min()), float(sub_lon.max()))

    return StaticGrid(
        lat=sub_lat,
        lon=sub_lon,
        active_mask_2d=active_mask,
        active_linear_idx=active_linear_idx.astype(np.int64),
        active_lat=sub_lat[active_lat_idx].astype(np.float32),
        active_lon=sub_lon[active_lon_idx].astype(np.float32),
        land_type=land_type.reshape(-1)[active_linear_idx],
        population=np.nan_to_num(population, nan=0.0).reshape(-1)[active_linear_idx].astype(np.float32),
        elevation_m=elevation.reshape(-1)[active_linear_idx].astype(np.float32),
        runoff_coeff=runoff.reshape(-1)[active_linear_idx].astype(np.float32),
        bbox=bbox,
        shape_2d=active_mask.shape,
        index_slices=(lat_slice, lon_slice),
        downsample_factor=downsample_factor,
    )


def list_climate_files(climate_dir: str | Path) -> list[ClimateFileRecord]:
    records: list[ClimateFileRecord] = []
    for path in Path(climate_dir).glob("*.nc4"):
        match = CLIMATE_FILE_RE.match(path.name)
        if not match:
            continue
        records.append(
            ClimateFileRecord(
                path=path,
                stream_id=int(match.group(1)),
                date=datetime.strptime(match.group(2), "%Y%m%d"),
            )
        )

    if not records:
        raise FileNotFoundError(f"No climate nc4 files found in {climate_dir}")

    resolved: dict[datetime, ClimateFileRecord] = {}
    for record in sorted(records, key=lambda item: (item.date, item.stream_id)):
        previous = resolved.get(record.date)
        if previous is None or record.stream_id >= previous.stream_id:
            resolved[record.date] = record
    return [resolved[key] for key in sorted(resolved)]


def validate_inputs(static_grid: StaticGrid, climate_files: list[ClimateFileRecord], requested_days: int | None = None) -> dict[str, int | float]:
    if static_grid.population.size == 0:
        raise ValueError("Static subset produced zero active cells.")
    if math.isclose(float(static_grid.population.sum()), 0.0):
        raise ValueError("Static subset has no population in active cells.")

    selected = climate_files if requested_days is None else climate_files[:requested_days]
    for left, right in zip(selected[:-1], selected[1:]):
        if (right.date - left.date).days != 1:
            raise ValueError(f"Climate files are not contiguous after date sort: {left.date.date()} -> {right.date.date()}")

    return {
        "active_cells": int(static_grid.population.size),
        "total_population": float(static_grid.population.sum()),
        "climate_days_available": len(climate_files),
    }


def _resolve_climate_var(dataset: Dataset, canonical_name: str) -> str | None:
    for alias in CLIMATE_VARIABLE_ALIASES[canonical_name]:
        if alias in dataset.variables:
            return alias
    return None


def _derive_humidity_proxy(temp_c: np.ndarray, precip_mm: np.ndarray) -> np.ndarray:
    humidity = 52.0 + np.clip(precip_mm, 0.0, 30.0) * 1.4 - np.maximum(temp_c - 30.0, 0.0) * 0.9
    return np.clip(humidity, 15.0, 100.0).astype(np.float32)


def _resolve_units(dataset_units: str | None, override_units: str | None, variable_name: str) -> str:
    if override_units:
        return override_units
    if dataset_units:
        return str(dataset_units)
    raise ValueError(f"{variable_name} units are missing from the dataset and no explicit override was provided.")


def _normalize_temperature(temp_native: np.ndarray, units: str) -> np.ndarray:
    units_norm = units.strip().lower()
    filled = np.nan_to_num(temp_native, nan=298.15).astype(np.float32)
    if units_norm in {"c", "degc", "degree_celsius", "degrees_celsius", "celsius"}:
        return filled
    if units_norm in {"k", "kelvin"} or float(np.nanmax(filled)) > 200.0:
        return filled - 273.15
    raise ValueError(f"Unsupported temperature units: {units}")


def _normalize_precipitation(precip_native: np.ndarray, units: str) -> np.ndarray:
    units_norm = units.strip().lower()
    filled = np.nan_to_num(precip_native, nan=0.0).astype(np.float32)
    if any(token in units_norm for token in ("mm", "millimeter")):
        return np.maximum(filled, 0.0)
    if any(token in units_norm for token in ("m", "meter")):
        return np.maximum(filled, 0.0) * 1000.0
    raise ValueError(f"Unsupported precipitation units: {units}")


def _normalize_humidity(humidity_native: np.ndarray, units: str) -> np.ndarray:
    units_norm = units.strip().lower()
    filled = np.nan_to_num(humidity_native, nan=60.0).astype(np.float32)
    if units_norm in {"1", "fraction", "ratio"} or float(np.nanmax(filled)) <= 1.5:
        filled = filled * 100.0
        return np.clip(filled, 0.0, 100.0).astype(np.float32)
    if units_norm in {"%", "percent", "percentage", "relative_percent"}:
        return np.clip(filled, 0.0, 100.0).astype(np.float32)
    raise ValueError(f"Unsupported humidity units: {units}")


def _grid_cache_signature(static_grid: StaticGrid) -> str:
    digest = hashlib.sha1()
    digest.update(str(static_grid.index_slices).encode("utf-8"))
    digest.update(static_grid.active_linear_idx.tobytes())
    return digest.hexdigest()[:16]


def _cache_unit_tag(
    temperature_units: str | None,
    precipitation_units: str | None,
    humidity_units: str | None,
) -> str:
    return "__".join(
        [
            (temperature_units or "dataset").replace("/", "_"),
            (precipitation_units or "dataset").replace("/", "_"),
            (humidity_units or "dataset").replace("/", "_"),
        ]
    )


def _cache_file_path(
    cache_dir: str | Path,
    record: ClimateFileRecord,
    static_grid: StaticGrid,
    temperature_units: str | None,
    precipitation_units: str | None,
    humidity_units: str | None,
) -> Path:
    signature = _grid_cache_signature(static_grid)
    unit_tag = _cache_unit_tag(temperature_units, precipitation_units, humidity_units)
    return Path(cache_dir) / f"{record.date.strftime('%Y%m%d')}_{signature}_{unit_tag}.npz"


def _window_cache_file_path(
    cache_dir: str | Path,
    climate_files: list[ClimateFileRecord],
    static_grid: StaticGrid,
    start_day: int,
    window_days: int,
    temperature_units: str | None,
    precipitation_units: str | None,
    humidity_units: str | None,
) -> Path:
    selected = climate_files[start_day : start_day + window_days]
    if not selected:
        raise IndexError("Requested an empty climate window.")
    signature = _grid_cache_signature(static_grid)
    unit_tag = _cache_unit_tag(temperature_units, precipitation_units, humidity_units)
    start_stamp = selected[0].date.strftime("%Y%m%d")
    end_stamp = selected[-1].date.strftime("%Y%m%d")
    return Path(cache_dir) / f"window_{start_stamp}_{end_stamp}_{len(selected):03d}_{signature}_{unit_tag}.npz"


def climate_day_cache_path(
    record: ClimateFileRecord,
    static_grid: StaticGrid,
    cache_dir: str | Path,
    temperature_units: str | None = None,
    precipitation_units: str | None = None,
    humidity_units: str | None = None,
) -> Path:
    return _cache_file_path(
        cache_dir=cache_dir,
        record=record,
        static_grid=static_grid,
        temperature_units=temperature_units,
        precipitation_units=precipitation_units,
        humidity_units=humidity_units,
    )


def climate_window_cache_path(
    climate_files: list[ClimateFileRecord],
    static_grid: StaticGrid,
    cache_dir: str | Path,
    start_day: int,
    window_days: int,
    temperature_units: str | None = None,
    precipitation_units: str | None = None,
    humidity_units: str | None = None,
) -> Path:
    return _window_cache_file_path(
        cache_dir=cache_dir,
        climate_files=climate_files,
        static_grid=static_grid,
        start_day=start_day,
        window_days=window_days,
        temperature_units=temperature_units,
        precipitation_units=precipitation_units,
        humidity_units=humidity_units,
    )


def _climate_shard_paths(
    cache_dir: str | Path,
    static_grid: StaticGrid,
    temperature_units: str | None,
    precipitation_units: str | None,
    humidity_units: str | None,
) -> tuple[Path, Path]:
    signature = _grid_cache_signature(static_grid)
    unit_tag = _cache_unit_tag(temperature_units, precipitation_units, humidity_units)
    base = Path(cache_dir) / f"shard_{signature}_{unit_tag}"
    return base.with_suffix(".json"), base.with_suffix(".bin")


def _window_shard_key(start_day: int, window_days: int) -> str:
    return f"{int(start_day)}:{int(window_days)}"


def _read_shard_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_shard_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _shard_parts_dir(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / "_shard_parts"


def climate_window_shard_available(
    climate_files: list[ClimateFileRecord],
    static_grid: StaticGrid,
    cache_dir: str | Path,
    start_day: int,
    window_days: int,
    temperature_units: str | None = None,
    precipitation_units: str | None = None,
    humidity_units: str | None = None,
) -> bool:
    try:
        selected = climate_files[start_day : start_day + window_days]
        if not selected:
            return False
        manifest_path, data_path = _climate_shard_paths(
            cache_dir=cache_dir,
            static_grid=static_grid,
            temperature_units=temperature_units,
            precipitation_units=precipitation_units,
            humidity_units=humidity_units,
        )
        manifest = _read_shard_manifest(manifest_path)
        if manifest is None or not data_path.exists():
            return False
        return _window_shard_key(start_day, len(selected)) in manifest.get("windows", {})
    except IndexError:
        return False


def _atomic_savez(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            np.savez(handle, **payload)
    except FileExistsError:
        return


def _load_or_build_climate_day(
    record: ClimateFileRecord,
    static_grid: StaticGrid,
    temperature_units: str | None = None,
    precipitation_units: str | None = None,
    humidity_units: str | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_path = None if cache_dir is None else _cache_file_path(
        cache_dir,
        record,
        static_grid,
        temperature_units,
        precipitation_units,
        humidity_units,
    )
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as payload:
            return (
                np.asarray(payload["temperature_c"], dtype=np.float32),
                np.asarray(payload["precipitation_mm"], dtype=np.float32),
                np.asarray(payload["humidity_pct"], dtype=np.float32),
            )

    lat_slice, lon_slice = static_grid.index_slices
    flat_mask = static_grid.active_linear_idx
    with Dataset(record.path) as dataset:
        temp_var = _resolve_climate_var(dataset, "temperature")
        precip_var = _resolve_climate_var(dataset, "precipitation")
        humidity_var = _resolve_climate_var(dataset, "humidity")
        if temp_var is None or precip_var is None:
            raise KeyError(f"{record.path.name} is missing required climate variables.")

        temp_native = _read_variable(dataset, temp_var)[lat_slice, lon_slice].astype(np.float32)
        precip_native = _read_variable(dataset, precip_var)[lat_slice, lon_slice].astype(np.float32)
        temp_unit = _resolve_units(getattr(dataset.variables[temp_var], "units", None), temperature_units, "temperature")
        precip_unit = _resolve_units(getattr(dataset.variables[precip_var], "units", None), precipitation_units, "precipitation")
        temp_c = _normalize_temperature(temp_native, temp_unit)
        precip_mm = _normalize_precipitation(precip_native, precip_unit)

        if humidity_var is None:
            humidity_pct = _derive_humidity_proxy(temp_c, precip_mm)
        else:
            humidity_native = _read_variable(dataset, humidity_var)[lat_slice, lon_slice].astype(np.float32)
            humidity_unit = _resolve_units(getattr(dataset.variables[humidity_var], "units", None), humidity_units, "humidity")
            humidity_pct = _normalize_humidity(humidity_native, humidity_unit)
            
        downsample_factor = getattr(static_grid, "downsample_factor", 1)
        if downsample_factor > 1:
            h, w = temp_c.shape
            h_new, w_new = h // downsample_factor, w // downsample_factor
            
            temp_c = temp_c[:h_new * downsample_factor, :w_new * downsample_factor]
            temp_c = temp_c.reshape(h_new, downsample_factor, w_new, downsample_factor).mean(axis=(1,3))
            
            precip_mm = precip_mm[:h_new * downsample_factor, :w_new * downsample_factor]
            precip_mm = precip_mm.reshape(h_new, downsample_factor, w_new, downsample_factor).sum(axis=(1,3))
            
            humidity_pct = humidity_pct[:h_new * downsample_factor, :w_new * downsample_factor]
            humidity_pct = humidity_pct.reshape(h_new, downsample_factor, w_new, downsample_factor).mean(axis=(1,3))

    day_payload = {
        "temperature_c": temp_c.reshape(-1)[flat_mask].astype(np.float32),
        "precipitation_mm": precip_mm.reshape(-1)[flat_mask].astype(np.float32),
        "humidity_pct": humidity_pct.reshape(-1)[flat_mask].astype(np.float32),
    }
    if cache_path is not None and not cache_path.exists():
        _atomic_savez(cache_path, day_payload)

    return day_payload["temperature_c"], day_payload["precipitation_mm"], day_payload["humidity_pct"]


def _build_climate_window_cpu(
    selected: list[ClimateFileRecord],
    static_grid: StaticGrid,
    temperature_units: str | None = None,
    precipitation_units: str | None = None,
    humidity_units: str | None = None,
    cache_dir: str | Path | None = None,
) -> ClimateWindow:
    n_days = len(selected)
    n_cells = int(static_grid.active_linear_idx.size)
    temperature_c = np.empty((n_days, n_cells), dtype=np.float32)
    precipitation_mm = np.empty((n_days, n_cells), dtype=np.float32)
    humidity_pct = np.empty((n_days, n_cells), dtype=np.float32)
    dates: list[str] = []

    for day_idx, record in enumerate(selected):
        temp_c, precip_mm, humidity_pct_day = _load_or_build_climate_day(
            record=record,
            static_grid=static_grid,
            temperature_units=temperature_units,
            precipitation_units=precipitation_units,
            humidity_units=humidity_units,
            cache_dir=cache_dir,
        )
        temperature_c[day_idx] = temp_c
        precipitation_mm[day_idx] = precip_mm
        humidity_pct[day_idx] = humidity_pct_day
        dates.append(record.date.strftime("%Y%m%d"))

    return ClimateWindow(
        dates=dates,
        temperature_c=temperature_c,
        precipitation_mm=precipitation_mm,
        humidity_pct=humidity_pct,
    )


def _build_window_blob(
    climate_files: list[ClimateFileRecord],
    static_grid: StaticGrid,
    start_day: int,
    window_days: int,
    temperature_units: str | None = None,
    precipitation_units: str | None = None,
    humidity_units: str | None = None,
) -> tuple[ClimateWindow, np.ndarray]:
    selected = climate_files[start_day : start_day + window_days]
    if not selected:
        raise IndexError("Requested an empty climate window.")
    climate_window = _build_climate_window_cpu(
        selected=selected,
        static_grid=static_grid,
        temperature_units=temperature_units,
        precipitation_units=precipitation_units,
        humidity_units=humidity_units,
        cache_dir=None,
    )
    window_blob = np.ascontiguousarray(
        np.stack(
            [
                np.asarray(climate_window.temperature_c, dtype=np.float32),
                np.asarray(climate_window.precipitation_mm, dtype=np.float32),
                np.asarray(climate_window.humidity_pct, dtype=np.float32),
            ],
            axis=0,
        ),
        dtype=np.float32,
    )
    return climate_window, window_blob


def _prebuild_shard_chunk_worker(task: dict[str, Any]) -> dict[str, Any]:
    climate_files: list[ClimateFileRecord] = task["climate_files"]
    static_grid: StaticGrid = task["static_grid"]
    window_days = int(task["window_days"])
    temperature_units = task["temperature_units"]
    precipitation_units = task["precipitation_units"]
    humidity_units = task["humidity_units"]
    part_path = Path(task["part_path"])
    chunk_starts = [int(item) for item in task["window_starts"]]

    entries: list[dict[str, Any]] = []
    part_path.parent.mkdir(parents=True, exist_ok=True)
    with part_path.open("wb") as handle:
        for start_day in chunk_starts:
            local_window_days = min(window_days, len(climate_files) - start_day)
            if local_window_days <= 0:
                continue
            climate_window, window_blob = _build_window_blob(
                climate_files=climate_files,
                static_grid=static_grid,
                start_day=start_day,
                window_days=local_window_days,
                temperature_units=temperature_units,
                precipitation_units=precipitation_units,
                humidity_units=humidity_units,
            )
            part_offset_bytes = int(handle.tell())
            window_blob.tofile(handle)
            entries.append(
                {
                    "start_day": int(start_day),
                    "n_days": int(local_window_days),
                    "n_cells": int(static_grid.active_linear_idx.size),
                    "part_offset_bytes": part_offset_bytes,
                    "size_bytes": int(window_blob.nbytes),
                    "dates": list(climate_window.dates),
                }
            )
    return {"part_path": str(part_path), "entries": entries}


def load_climate_window_cpu(
    climate_files: list[ClimateFileRecord],
    static_grid: StaticGrid,
    start_day: int,
    window_days: int = DEFAULT_CLIMATE_WINDOW_DAYS,
    temperature_units: str | None = None,
    precipitation_units: str | None = None,
    humidity_units: str | None = None,
    cache_dir: str | Path | None = None,
) -> ClimateWindow:
    if window_days < 1 or window_days > MAX_CLIMATE_WINDOW_DAYS:
        raise ValueError(f"window_days must be in [1, {MAX_CLIMATE_WINDOW_DAYS}]")

    selected = climate_files[start_day : start_day + window_days]
    if not selected:
        raise IndexError("Requested an empty climate window.")

    shard_manifest_path = None
    shard_data_path = None
    if cache_dir is not None:
        shard_manifest_path, shard_data_path = _climate_shard_paths(
            cache_dir=cache_dir,
            static_grid=static_grid,
            temperature_units=temperature_units,
            precipitation_units=precipitation_units,
            humidity_units=humidity_units,
        )
        shard_manifest = _read_shard_manifest(shard_manifest_path)
        shard_key = _window_shard_key(start_day, len(selected))
        if shard_manifest is not None and shard_data_path.exists():
            entry = shard_manifest.get("windows", {}).get(shard_key)
            if entry is not None:
                n_days = int(entry["n_days"])
                n_cells = int(entry["n_cells"])
                offset_bytes = int(entry["offset_bytes"])
                window_blob = np.memmap(
                    shard_data_path,
                    dtype=np.float32,
                    mode="r",
                    offset=offset_bytes,
                    shape=(3, n_days, n_cells),
                    order="C",
                )
                return ClimateWindow(
                    dates=[str(item) for item in entry["dates"]],
                    temperature_c=window_blob[0],
                    precipitation_mm=window_blob[1],
                    humidity_pct=window_blob[2],
                )

    window_cache = None if cache_dir is None else _window_cache_file_path(
        cache_dir=cache_dir,
        climate_files=climate_files,
        static_grid=static_grid,
        start_day=start_day,
        window_days=window_days,
        temperature_units=temperature_units,
        precipitation_units=precipitation_units,
        humidity_units=humidity_units,
    )
    if window_cache is not None and window_cache.exists():
        with np.load(window_cache, allow_pickle=False) as payload:
            return ClimateWindow(
                dates=[str(item) for item in np.asarray(payload["dates"]).tolist()],
                temperature_c=np.asarray(payload["temperature_c"], dtype=np.float32),
                precipitation_mm=np.asarray(payload["precipitation_mm"], dtype=np.float32),
                humidity_pct=np.asarray(payload["humidity_pct"], dtype=np.float32),
            )

    climate_window = _build_climate_window_cpu(
        selected=selected,
        static_grid=static_grid,
        temperature_units=temperature_units,
        precipitation_units=precipitation_units,
        humidity_units=humidity_units,
        cache_dir=cache_dir,
    )
    if window_cache is not None and not window_cache.exists():
        _atomic_savez(
            window_cache,
            {
                "dates": np.asarray(climate_window.dates, dtype="U8"),
                "temperature_c": np.asarray(climate_window.temperature_c, dtype=np.float32),
                "precipitation_mm": np.asarray(climate_window.precipitation_mm, dtype=np.float32),
                "humidity_pct": np.asarray(climate_window.humidity_pct, dtype=np.float32),
            },
        )

    return climate_window


def prebuild_climate_windows(
    climate_files: list[ClimateFileRecord],
    static_grid: StaticGrid,
    window_starts: list[int],
    window_days: int = DEFAULT_CLIMATE_WINDOW_DAYS,
    temperature_units: str | None = None,
    precipitation_units: str | None = None,
    humidity_units: str | None = None,
    cache_dir: str | Path | None = None,
    progress_callback: Any | None = None,
) -> int:
    if cache_dir is None:
        return 0

    built = 0
    for start_day in window_starts:
        local_window_days = min(window_days, len(climate_files) - start_day)
        if local_window_days <= 0:
            continue
        window_cache = climate_window_cache_path(
            climate_files=climate_files,
            static_grid=static_grid,
            cache_dir=cache_dir,
            start_day=start_day,
            window_days=local_window_days,
            temperature_units=temperature_units,
            precipitation_units=precipitation_units,
            humidity_units=humidity_units,
        )
        if not window_cache.exists():
            load_climate_window_cpu(
                climate_files=climate_files,
                static_grid=static_grid,
                start_day=start_day,
                window_days=local_window_days,
                temperature_units=temperature_units,
                precipitation_units=precipitation_units,
                humidity_units=humidity_units,
                cache_dir=cache_dir,
            )
            built += 1
        if progress_callback is not None:
            progress_callback(start_day, local_window_days, window_cache)
    return built


def prebuild_climate_shard(
    climate_files: list[ClimateFileRecord],
    static_grid: StaticGrid,
    window_starts: list[int],
    window_days: int = DEFAULT_CLIMATE_WINDOW_DAYS,
    temperature_units: str | None = None,
    precipitation_units: str | None = None,
    humidity_units: str | None = None,
    cache_dir: str | Path | None = None,
    progress_callback: Any | None = None,
    num_workers: int = 1,
) -> int:
    if cache_dir is None:
        return 0

    manifest_path, data_path = _climate_shard_paths(
        cache_dir=cache_dir,
        static_grid=static_grid,
        temperature_units=temperature_units,
        precipitation_units=precipitation_units,
        humidity_units=humidity_units,
    )
    manifest = _read_shard_manifest(manifest_path)
    if manifest is None:
        manifest = {
            "version": 1,
            "dtype": "float32",
            "n_cells": int(static_grid.active_linear_idx.size),
            "windows": {},
        }
    else:
        manifest["windows"] = dict(manifest.get("windows", {}))
    data_path.parent.mkdir(parents=True, exist_ok=True)

    pending_starts: list[int] = []
    for start_day in window_starts:
        local_window_days = min(window_days, len(climate_files) - start_day)
        if local_window_days <= 0:
            continue
        shard_key = _window_shard_key(start_day, local_window_days)
        if shard_key in manifest["windows"]:
            if progress_callback is not None:
                progress_callback(start_day, local_window_days, data_path)
            continue
        pending_starts.append(start_day)

    if not pending_starts:
        return 0

    worker_count = max(int(num_workers), 1)
    built = 0
    if worker_count == 1 or len(pending_starts) == 1:
        with data_path.open("ab") as handle:
            for start_day in pending_starts:
                local_window_days = min(window_days, len(climate_files) - start_day)
                climate_window, window_blob = _build_window_blob(
                    climate_files=climate_files,
                    static_grid=static_grid,
                    start_day=start_day,
                    window_days=local_window_days,
                    temperature_units=temperature_units,
                    precipitation_units=precipitation_units,
                    humidity_units=humidity_units,
                )
                offset_bytes = int(handle.tell())
                window_blob.tofile(handle)
                manifest["windows"][_window_shard_key(start_day, local_window_days)] = {
                    "start_day": int(start_day),
                    "n_days": int(local_window_days),
                    "n_cells": int(static_grid.active_linear_idx.size),
                    "offset_bytes": offset_bytes,
                    "dates": list(climate_window.dates),
                }
                _write_shard_manifest(manifest_path, manifest)
                built += 1
                if progress_callback is not None:
                    progress_callback(start_day, local_window_days, data_path)
        return built

    chunk_count = min(worker_count, len(pending_starts))
    chunk_size = max(math.ceil(len(pending_starts) / chunk_count), 1)
    parts_dir = _shard_parts_dir(cache_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[dict[str, Any]] = []
    for chunk_index in range(chunk_count):
        chunk_starts = pending_starts[chunk_index * chunk_size : (chunk_index + 1) * chunk_size]
        if not chunk_starts:
            continue
        tasks.append(
            {
                "climate_files": climate_files,
                "static_grid": static_grid,
                "window_days": int(window_days),
                "temperature_units": temperature_units,
                "precipitation_units": precipitation_units,
                "humidity_units": humidity_units,
                "window_starts": chunk_starts,
                "part_path": str(parts_dir / f"part_{os.getpid()}_{chunk_index:03d}.bin"),
            }
        )

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        for result in executor.map(_prebuild_shard_chunk_worker, tasks):
            results.append(result)

    try:
        with data_path.open("ab") as handle:
            for result in results:
                part_path = Path(result["part_path"])
                offset_base = int(handle.tell())
                part_bytes = part_path.read_bytes()
                handle.write(part_bytes)
                for entry in result["entries"]:
                    shard_key = _window_shard_key(int(entry["start_day"]), int(entry["n_days"]))
                    manifest["windows"][shard_key] = {
                        "start_day": int(entry["start_day"]),
                        "n_days": int(entry["n_days"]),
                        "n_cells": int(entry["n_cells"]),
                        "offset_bytes": offset_base + int(entry["part_offset_bytes"]),
                        "dates": list(entry["dates"]),
                    }
                    built += 1
                    if progress_callback is not None:
                        progress_callback(int(entry["start_day"]), int(entry["n_days"]), data_path)
                _write_shard_manifest(manifest_path, manifest)
    finally:
        for task in tasks:
            part_path = Path(task["part_path"])
            try:
                part_path.unlink()
            except FileNotFoundError:
                pass
        try:
            parts_dir.rmdir()
        except OSError:
            pass

    return built


def _to_pinned_numpy(arr: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(arr, dtype=np.float32)
    if not CUPY_AVAILABLE or cp is None:
        return contiguous
    pinned_mem = cp.cuda.alloc_pinned_memory(contiguous.nbytes)
    pinned_arr = np.frombuffer(pinned_mem, dtype=contiguous.dtype, count=contiguous.size).reshape(contiguous.shape)
    np.copyto(pinned_arr, contiguous)
    return pinned_arr


def climate_window_to_backend_async(
    climate_window: ClimateWindow,
    backend: Backend,
    stream: Any = None,
    use_pinned_memory: bool = True,
) -> ClimateWindowTransfer:
    if not backend.gpu_enabled or not CUPY_AVAILABLE or cp is None:
        return ClimateWindowTransfer(
            climate_window=ClimateWindow(
                dates=climate_window.dates,
                temperature_c=backend.xp.asarray(climate_window.temperature_c, dtype=backend.xp.float32),
                precipitation_mm=backend.xp.asarray(climate_window.precipitation_mm, dtype=backend.xp.float32),
                humidity_pct=backend.xp.asarray(climate_window.humidity_pct, dtype=backend.xp.float32),
            )
        )

    xp = backend.xp
    transfer_stream = stream or xp.cuda.Stream(non_blocking=True)
    host_temp = _to_pinned_numpy(np.asarray(climate_window.temperature_c, dtype=np.float32)) if use_pinned_memory else np.ascontiguousarray(climate_window.temperature_c, dtype=np.float32)
    host_precip = _to_pinned_numpy(np.asarray(climate_window.precipitation_mm, dtype=np.float32)) if use_pinned_memory else np.ascontiguousarray(climate_window.precipitation_mm, dtype=np.float32)
    host_humidity = _to_pinned_numpy(np.asarray(climate_window.humidity_pct, dtype=np.float32)) if use_pinned_memory else np.ascontiguousarray(climate_window.humidity_pct, dtype=np.float32)

    device_temp = xp.empty(host_temp.shape, dtype=xp.float32)
    device_precip = xp.empty(host_precip.shape, dtype=xp.float32)
    device_humidity = xp.empty(host_humidity.shape, dtype=xp.float32)

    with transfer_stream:
        device_temp.set(host_temp, stream=transfer_stream)
        device_precip.set(host_precip, stream=transfer_stream)
        device_humidity.set(host_humidity, stream=transfer_stream)

    return ClimateWindowTransfer(
        climate_window=ClimateWindow(
            dates=climate_window.dates,
            temperature_c=device_temp,
            precipitation_mm=device_precip,
            humidity_pct=device_humidity,
        ),
        stream=transfer_stream,
        pinned_refs=(host_temp, host_precip, host_humidity),
    )


def climate_window_to_backend(climate_window: ClimateWindow, backend: Backend) -> ClimateWindow:
    transfer = climate_window_to_backend_async(climate_window, backend, use_pinned_memory=False)
    return transfer.wait()


def load_climate_window(
    climate_files: list[ClimateFileRecord],
    static_grid: StaticGrid,
    start_day: int,
    backend: Backend,
    window_days: int = DEFAULT_CLIMATE_WINDOW_DAYS,
    temperature_units: str | None = None,
    precipitation_units: str | None = None,
    humidity_units: str | None = None,
    cache_dir: str | Path | None = None,
) -> ClimateWindow:
    climate_window = load_climate_window_cpu(
        climate_files=climate_files,
        static_grid=static_grid,
        start_day=start_day,
        window_days=window_days,
        temperature_units=temperature_units,
        precipitation_units=precipitation_units,
        humidity_units=humidity_units,
        cache_dir=cache_dir,
    )
    return climate_window_to_backend(climate_window, backend)


def partition_static_grid(static_grid: StaticGrid, spatial_decomp: Any, rank: int) -> StaticGrid:
    lat_s, lat_e, lon_s, lon_e = spatial_decomp.get_partition(rank)
    local_mask = static_grid.active_mask_2d[lat_s:lat_e, lon_s:lon_e]
    local_linear_idx = np.flatnonzero(local_mask.reshape(-1))
    local_lat_idx, local_lon_idx = np.nonzero(local_mask)

    local_lat = static_grid.lat[lat_s:lat_e]
    local_lon = static_grid.lon[lon_s:lon_e]

    def _slice_active_values(values: np.ndarray, fill_value: float | int = 0) -> np.ndarray:
        full = np.full(static_grid.shape_2d, fill_value, dtype=values.dtype)
        full.reshape(-1)[static_grid.active_linear_idx] = values
        return full[lat_s:lat_e, lon_s:lon_e].reshape(-1)[local_linear_idx]

    global_lat_slice, global_lon_slice = static_grid.index_slices
    global_lat_start = global_lat_slice.start or 0
    global_lon_start = global_lon_slice.start or 0

    if local_linear_idx.size:
        bbox = (
            float(local_lat[local_lat_idx].min()),
            float(local_lat[local_lat_idx].max()),
            float(local_lon[local_lon_idx].min()),
            float(local_lon[local_lon_idx].max()),
        )
    elif local_lat.size and local_lon.size:
        bbox = (float(local_lat.min()), float(local_lat.max()), float(local_lon.min()), float(local_lon.max()))
    else:
        bbox = (0.0, 0.0, 0.0, 0.0)

    return StaticGrid(
        lat=local_lat,
        lon=local_lon,
        active_mask_2d=local_mask,
        active_linear_idx=local_linear_idx.astype(np.int64),
        active_lat=local_lat[local_lat_idx].astype(np.float32),
        active_lon=local_lon[local_lon_idx].astype(np.float32),
        land_type=_slice_active_values(static_grid.land_type, fill_value=0).astype(np.uint8),
        population=_slice_active_values(static_grid.population, fill_value=0.0).astype(np.float32),
        elevation_m=_slice_active_values(static_grid.elevation_m, fill_value=0.0).astype(np.float32),
        runoff_coeff=_slice_active_values(static_grid.runoff_coeff, fill_value=0.0).astype(np.float32),
        bbox=bbox,
        shape_2d=local_mask.shape,
        index_slices=(
            slice(global_lat_start + lat_s, global_lat_start + lat_e),
            slice(global_lon_start + lon_s, global_lon_start + lon_e),
        ),
    )


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_climate_window_distributed(
    climate_files: list[ClimateFileRecord],
    static_grid: StaticGrid,
    spatial_decomp: Any,
    rank: int,
    start_day: int,
    backend: Any,
    window_days: int = DEFAULT_CLIMATE_WINDOW_DAYS,
    temperature_units: str | None = None,
    precipitation_units: str | None = None,
    humidity_units: str | None = None,
    halo_width: int = 1,
) -> ClimateWindow:
    if window_days < 1 or window_days > MAX_CLIMATE_WINDOW_DAYS:
        raise ValueError(f"window_days must be in [1, {MAX_CLIMATE_WINDOW_DAYS}]")

    selected = climate_files[start_day : start_day + window_days]
    if not selected:
        raise IndexError("Requested an empty climate window.")

    lat_slice, lon_slice = static_grid.index_slices
    lat_s, lat_e, lon_s, lon_e = spatial_decomp.get_partition(rank)

    lat_start_with_halo = max(lat_s - halo_width, 0)
    lat_end_with_halo = min(lat_e + halo_width, static_grid.lat.size)
    lon_start_with_halo = max(lon_s - halo_width, 0)
    lon_end_with_halo = min(lon_e + halo_width, static_grid.lon.size)

    local_lat_slice = slice(lat_start_with_halo, lat_end_with_halo)
    local_lon_slice = slice(lon_start_with_halo, lon_end_with_halo)

    n_local_lats = lat_end_with_halo - lat_start_with_halo
    n_local_lons = lon_end_with_halo - lon_start_with_halo
    n_local_cells = n_local_lats * n_local_lons

    temp_days: list[np.ndarray] = []
    precip_days: list[np.ndarray] = []
    humidity_days: list[np.ndarray] = []
    dates: list[str] = []

    for record in selected:
        with Dataset(record.path) as dataset:
            temp_var = _resolve_climate_var(dataset, "temperature")
            precip_var = _resolve_climate_var(dataset, "precipitation")
            humidity_var = _resolve_climate_var(dataset, "humidity")
            if temp_var is None or precip_var is None:
                raise KeyError(f"{record.path.name} is missing required climate variables.")

            temp_native = _read_variable(dataset, temp_var)[local_lat_slice, local_lon_slice].astype(np.float32)
            precip_native = _read_variable(dataset, precip_var)[local_lat_slice, local_lon_slice].astype(np.float32)
            temp_unit = _resolve_units(getattr(dataset.variables[temp_var], "units", None), temperature_units, "temperature")
            precip_unit = _resolve_units(getattr(dataset.variables[precip_var], "units", None), precipitation_units, "precipitation")
            temp_c = _normalize_temperature(temp_native, temp_unit)
            precip_mm = _normalize_precipitation(precip_native, precip_unit)

            if humidity_var is None:
                humidity_pct = _derive_humidity_proxy(temp_c, precip_mm)
            else:
                humidity_native = _read_variable(dataset, humidity_var)[local_lat_slice, local_lon_slice].astype(np.float32)
                humidity_unit = _resolve_units(getattr(dataset.variables[humidity_var], "units", None), humidity_units, "humidity")
                humidity_pct = _normalize_humidity(humidity_native, humidity_unit)

        temp_days.append(temp_c.reshape(-1))
        precip_days.append(precip_mm.reshape(-1))
        humidity_days.append(humidity_pct.reshape(-1))
        dates.append(record.date.strftime("%Y%m%d"))

    return ClimateWindow(
        dates=dates,
        temperature_c=backend.xp.asarray(np.stack(temp_days, axis=0), dtype=backend.xp.float32),
        precipitation_mm=backend.xp.asarray(np.stack(precip_days, axis=0), dtype=backend.xp.float32),
        humidity_pct=backend.xp.asarray(np.stack(humidity_days, axis=0), dtype=backend.xp.float32),
    )


@dataclass(frozen=True)
class DistributedStaticGrid:
    lat: np.ndarray
    lon: np.ndarray
    active_mask_2d: np.ndarray
    active_linear_idx: np.ndarray
    active_lat: np.ndarray
    active_lon: np.ndarray
    land_type: np.ndarray
    population: np.ndarray
    elevation_m: np.ndarray
    runoff_coeff: np.ndarray
    bbox: tuple[float, float, float, float]
    shape_2d: tuple[int, int]
    index_slices: tuple[slice, slice]
    local_lat_slice: slice
    local_lon_slice: slice
    local_n_cells: int
    rank: int
    n_gpus: int


def load_static_grid_distributed(
    static_path: str | Path,
    spatial_decomp: Any,
    rank: int,
    lat_min: float | None = None,
    lat_max: float | None = None,
    lon_min: float | None = None,
    lon_max: float | None = None,
    min_population: float = 1.0,
) -> DistributedStaticGrid:
    static_path = Path(static_path)
    with Dataset(static_path) as dataset:
        missing = [name for name in SUPPORTED_STATIC_VARIABLES if name not in dataset.variables]
        if missing:
            raise KeyError(f"Static dataset is missing required variables: {missing}")

        lat = _read_variable(dataset, "lat").astype(np.float32)
        lon = _read_variable(dataset, "lon").astype(np.float32)
        lat_slice = _find_slice(lat, lat_min, lat_max)
        lon_slice = _find_slice(lon, lon_min, lon_max)

        sub_lat = lat[lat_slice]
        sub_lon = lon[lon_slice]
        population = _normalize_population(_read_variable(dataset, "population")[lat_slice, lon_slice])
        elevation = _normalize_elevation(_read_variable(dataset, "elevation")[lat_slice, lon_slice])
        landcover = _read_variable(dataset, "landcover", fill_value=0)[lat_slice, lon_slice].astype(np.uint8)

    land_type = _map_land_type(landcover)
    active_mask = np.isfinite(population) & (population >= float(min_population)) & ~np.isin(landcover, list(LCCS_WATER | LCCS_SNOW))

    lat_s, lat_e, lon_s, lon_e = spatial_decomp.get_partition(rank)
    global_lat_start = lat_slice.start or 0
    global_lon_start = lon_slice.start or 0

    local_lat_idx_start = max(lat_s - global_lat_start, 0)
    local_lat_idx_end = min(lat_e - global_lat_start, sub_lat.size)
    local_lon_idx_start = max(lon_s - global_lon_start, 0)
    local_lon_idx_end = min(lon_e - global_lon_start, sub_lon.size)

    local_active_mask = active_mask[local_lat_idx_start:local_lat_idx_end, local_lon_idx_start:local_lon_idx_end]

    active_linear_idx = np.flatnonzero(local_active_mask.reshape(-1))
    active_lat_idx, active_lon_idx = np.nonzero(local_active_mask)
    runoff = _estimate_runoff(np.nan_to_num(population, nan=0.0), land_type)

    local_population = np.nan_to_num(population, nan=0.0)[local_lat_idx_start:local_lat_idx_end, local_lon_idx_start:local_lon_idx_end]
    local_land_type = land_type[local_lat_idx_start:local_lat_idx_end, local_lon_idx_start:local_lon_idx_end]
    local_runoff = runoff[local_lat_idx_start:local_lat_idx_end, local_lon_idx_start:local_lon_idx_end]
    local_elevation = elevation[local_lat_idx_start:local_lat_idx_end, local_lon_idx_start:local_lon_idx_end]

    local_lat = sub_lat[local_lat_idx_start:local_lat_idx_end]
    local_lon = sub_lon[local_lon_idx_start:local_lon_idx_end]

    local_active_lat = local_lat[active_lat_idx].astype(np.float32)
    local_active_lon = local_lon[active_lon_idx].astype(np.float32)
    n_local_cells = len(active_linear_idx)

    bbox = (float(local_active_lat.min()) if n_local_cells > 0 else 0.0,
            float(local_active_lat.max()) if n_local_cells > 0 else 0.0,
            float(local_active_lon.min()) if n_local_cells > 0 else 0.0,
            float(local_active_lon.max()) if n_local_cells > 0 else 0.0)

    return DistributedStaticGrid(
        lat=local_lat,
        lon=local_lon,
        active_mask_2d=local_active_mask,
        active_linear_idx=active_linear_idx.astype(np.int64),
        active_lat=local_active_lat,
        active_lon=local_active_lon,
        land_type=local_land_type.reshape(-1)[active_linear_idx],
        population=local_population.reshape(-1)[active_linear_idx].astype(np.float32),
        elevation_m=local_elevation.reshape(-1)[active_linear_idx].astype(np.float32),
        runoff_coeff=local_runoff.reshape(-1)[active_linear_idx].astype(np.float32),
        bbox=bbox,
        shape_2d=local_active_mask.shape,
        index_slices=(slice(local_lat_idx_start, local_lat_idx_end), slice(local_lon_idx_start, local_lon_idx_end)),
        local_lat_slice=slice(local_lat_idx_start, local_lat_idx_end),
        local_lon_slice=slice(local_lon_idx_start, local_lon_idx_end),
        local_n_cells=n_local_cells,
        rank=rank,
        n_gpus=spatial_decomp.n_gpus,
    )
