from __future__ import annotations

import calendar
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from .backend import Backend
from .constants import (
    ADE_SEVERE_AMPLIFICATION,
    BASE_FATAL_FRACTION,
    BASE_SEVERE_FRACTION,
    CROSS_PROTECTION,
    DEFAULT_CLIMATE_WINDOW_DAYS,
    EIP_THERMAL_CONST,
    EIP_THRESHOLD_C,
    INCUBATION_DAYS,
    INFECTIOUS_DAYS,
    LANDTYPE_PREFERENCE,
    MAX_CLIMATE_WINDOW_DAYS,
    N_SPECIES,
    N_VIRUS,
    REPORTING_RATE,
    SEVERE_DAYS,
    SPECIES_NAMES,
    SPECIES_PARAMS,
    VECTOR_COMPETENCE,
    VIRUS_NAMES,
)
from .io import ClimateWindow, StaticGrid, dump_json
from .seed_data import DEFAULT_SEED_REPORTS
from .evaluation import load_validation_observations


@dataclass
class SimulationConfig:
    static_path: str = "data/static_layers_0_05.nc4"
    climate_dir: str = "data/climate"
    output_dir: str = "output/mosq_vectorized_gpu"
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None
    start_date: str | None = None
    days: int = 30
    climate_window_days: int = DEFAULT_CLIMATE_WINDOW_DAYS
    climate_memory_days: int = DEFAULT_CLIMATE_WINDOW_DAYS
    seed: int = 42
    use_gpu: bool = True
    host_scale: float = 1.0
    mosquito_scale: float = 0.03
    min_population: float = 1.0
    temperature_units: str | None = None
    precipitation_units: str | None = None
    humidity_units: str | None = None
    checkpoint_interval_windows: int = 0
    reporting_rate_scale: list[float] | None = None
    disable_default_seeds: bool = False
    initial_state_path: str | None = None
    initial_state_window_days: int = 365
    initial_vector_seed_rate: float = 1e-4
    initial_state_include_reported: bool = False
    importation_targets: str | None = None
    importation_frequency: str = "year"
    importation_fraction: float = 0.0
    seasonal_seed_cases: list[float] | None = None
    seasonal_seed_month: int = 1
    seasonal_seed_day: int = 1
    reservoir_force_scale: float = 0.0
    virus_reservoir_force_scale: list[float] | None = None
    disease_seasonality_preset: str = "none"
    virus_seasonal_peak_month: list[float] | None = None
    virus_seasonal_amplitude: list[float] | None = None
    virus_seasonal_floor: list[float] | None = None
    virus_aedes_scale: list[float] | None = None
    virus_culex_scale: list[float] | None = None
    virus_other_vector_scale: list[float] | None = None
    virus_urban_preference: list[float] | None = None
    virus_wetland_preference: list[float] | None = None
    virus_forest_preference: list[float] | None = None
    virus_rural_preference: list[float] | None = None
    virus_habitat_floor: list[float] | None = None
    virus_precipitation_sensitivity: list[float] | None = None
    virus_humidity_sensitivity: list[float] | None = None
    immunity_waning_rate: list[float] | None = None
    host_replenishment_rate: float = 0.0
    spatial_diffusion_rate: float = 0.0
    host_transmission_coeff: float = 0.11
    vector_transmission_coeff: float = 0.14
    host_transmission_scale: list[float] | None = None
    vector_transmission_scale: list[float] | None = None

    def validate(self) -> None:
        if self.days < 1:
            raise ValueError("days must be >= 1")
        if not 1 <= self.climate_window_days <= MAX_CLIMATE_WINDOW_DAYS:
            raise ValueError(f"climate_window_days must be in [1, {MAX_CLIMATE_WINDOW_DAYS}]")
        if not 1 <= self.climate_memory_days <= MAX_CLIMATE_WINDOW_DAYS:
            raise ValueError(f"climate_memory_days must be in [1, {MAX_CLIMATE_WINDOW_DAYS}]")
        if self.host_scale <= 0.0:
            raise ValueError("host_scale must be > 0")
        if self.mosquito_scale <= 0.0:
            raise ValueError("mosquito_scale must be > 0")
        if self.min_population < 0.0:
            raise ValueError("min_population must be >= 0")
        if self.checkpoint_interval_windows < 0:
            raise ValueError("checkpoint_interval_windows must be >= 0")
        if self.initial_state_window_days < 1:
            raise ValueError("initial_state_window_days must be >= 1")
        if self.reporting_rate_scale is not None:
            scale = np.asarray(self.reporting_rate_scale, dtype=np.float32).reshape(-1)
            if scale.size not in (1, N_VIRUS):
                raise ValueError(f"reporting_rate_scale must contain 1 or {N_VIRUS} values")
            if not np.all(np.isfinite(scale)) or np.any(scale < 0.0):
                raise ValueError("reporting_rate_scale values must be finite and >= 0")
        for field_name in (
            "host_transmission_scale",
            "vector_transmission_scale",
            "virus_reservoir_force_scale",
            "virus_aedes_scale",
            "virus_culex_scale",
            "virus_other_vector_scale",
            "virus_urban_preference",
            "virus_wetland_preference",
            "virus_forest_preference",
            "virus_rural_preference",
            "virus_habitat_floor",
            "virus_precipitation_sensitivity",
            "virus_humidity_sensitivity",
            "immunity_waning_rate",
        ):
            scale_values = getattr(self, field_name)
            if scale_values is None:
                continue
            scale = np.asarray(scale_values, dtype=np.float32).reshape(-1)
            if scale.size not in (1, N_VIRUS):
                raise ValueError(f"{field_name} must contain 1 or {N_VIRUS} values")
            if not np.all(np.isfinite(scale)) or np.any(scale < 0.0):
                raise ValueError(f"{field_name} values must be finite and >= 0")
        if self.disease_seasonality_preset not in {"none", "ecology"}:
            raise ValueError("disease_seasonality_preset must be one of: none, ecology")
        if self.virus_seasonal_peak_month is not None:
            values = np.asarray(self.virus_seasonal_peak_month, dtype=np.float32).reshape(-1)
            if values.size not in (1, N_VIRUS):
                raise ValueError(f"virus_seasonal_peak_month must contain 1 or {N_VIRUS} values")
            if not np.all(np.isfinite(values)) or np.any(values < 1.0) or np.any(values > 12.0):
                raise ValueError("virus_seasonal_peak_month values must be finite and in [1, 12]")
        for field_name in ("virus_seasonal_amplitude", "virus_seasonal_floor"):
            values = getattr(self, field_name)
            if values is None:
                continue
            scale = np.asarray(values, dtype=np.float32).reshape(-1)
            if scale.size not in (1, N_VIRUS):
                raise ValueError(f"{field_name} must contain 1 or {N_VIRUS} values")
            if not np.all(np.isfinite(scale)) or np.any(scale < 0.0):
                raise ValueError(f"{field_name} values must be finite and >= 0")
        if self.importation_frequency not in {"year", "month", "day"}:
            raise ValueError("importation_frequency must be one of: year, month, day")
        for name in (
            "initial_vector_seed_rate",
            "importation_fraction",
            "reservoir_force_scale",
            "host_replenishment_rate",
            "spatial_diffusion_rate",
            "host_transmission_coeff",
            "vector_transmission_coeff",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.spatial_diffusion_rate > 1.0:
            raise ValueError("spatial_diffusion_rate must be <= 1")
        if not 1 <= self.seasonal_seed_month <= 12:
            raise ValueError("seasonal_seed_month must be in [1, 12]")
        if not 1 <= self.seasonal_seed_day <= 31:
            raise ValueError("seasonal_seed_day must be in [1, 31]")
        if self.seasonal_seed_cases is not None:
            cases = np.asarray(self.seasonal_seed_cases, dtype=np.float32).reshape(-1)
            if cases.size not in (1, N_VIRUS):
                raise ValueError(f"seasonal_seed_cases must contain 1 or {N_VIRUS} values")
            if not np.all(np.isfinite(cases)) or np.any(cases < 0.0):
                raise ValueError("seasonal_seed_cases values must be finite and >= 0")


@dataclass
class SimulationState:
    host_population: Any
    host_s: Any
    host_e: Any
    host_i: Any
    host_severe: Any
    host_r: Any
    host_d: Any
    mosquito_aquatic: Any
    mosquito_adult: Any
    mosquito_exposed: Any
    mosquito_infectious: Any
    water_level: Any
    cumulative_reported: Any
    climate_temperature_hist: Any
    climate_precip_hist: Any
    climate_humidity_hist: Any
    climate_hist_cursor: int
    climate_hist_count: int


class WindowWriter:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_dir = self.output_dir / "chunks"
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.files: list[str] = []

    def write_window(self, window_index: int, payload: dict[str, np.ndarray]) -> str:
        target = self.chunk_dir / f"window_{window_index:04d}.npz"
        np.savez(target, **payload)
        self.files.append(target.name)
        return target.name

    def write_checkpoint(self, window_index: int, payload: dict[str, np.ndarray]) -> str:
        target = self.checkpoint_dir / f"checkpoint_{window_index:04d}.npz"
        np.savez(target, **payload)
        return target.name

    @staticmethod
    def read_checkpoint(path: str | Path) -> dict[str, np.ndarray]:
        with np.load(Path(path), allow_pickle=False) as payload:
            return {key: payload[key] for key in payload.files}


class VectorizedABMSimulator:
    def __init__(self, cfg: SimulationConfig, static_grid: StaticGrid, backend: Backend):
        self.cfg = cfg
        self.cfg.validate()
        self.grid = static_grid
        self.backend = backend
        self.xp = backend.xp
        self.rng = self._build_rng()

        xp = self.xp
        self.land_type = xp.asarray(static_grid.land_type, dtype=xp.int32)
        self.runoff_coeff = xp.asarray(static_grid.runoff_coeff, dtype=xp.float32)
        self.land_pref = xp.asarray(LANDTYPE_PREFERENCE, dtype=xp.float32)
        self.active_lat = xp.asarray(static_grid.active_lat, dtype=xp.float32)

        self.species_tmin = xp.asarray([item.t_min for item in SPECIES_PARAMS], dtype=xp.float32)
        self.species_topt = xp.asarray([item.t_opt for item in SPECIES_PARAMS], dtype=xp.float32)
        self.species_tmax = xp.asarray([item.t_max for item in SPECIES_PARAMS], dtype=xp.float32)
        self.species_bite_target = xp.asarray([item.bite_rate_25c for item in SPECIES_PARAMS], dtype=xp.float32)
        self.species_mortality = xp.asarray([item.mortality_base for item in SPECIES_PARAMS], dtype=xp.float32)
        self.species_fertility_target = xp.asarray([item.fertility for item in SPECIES_PARAMS], dtype=xp.float32)
        self.species_maturation_target = xp.asarray([item.maturation_rate for item in SPECIES_PARAMS], dtype=xp.float32)
        self.species_k = xp.asarray([item.carrying_scale for item in SPECIES_PARAMS], dtype=xp.float32)

        self.species_briere_peak = xp.asarray(
            [self._briere_peak(item.t_min, item.t_max) for item in SPECIES_PARAMS],
            dtype=xp.float32,
        )
        self.species_bite_scale = xp.asarray(
            [self._briere_scale(item.t_min, item.t_max, item.bite_rate_25c) for item in SPECIES_PARAMS],
            dtype=xp.float32,
        )
        self.species_fertility_scale = xp.asarray(
            [self._briere_scale(item.t_min, item.t_max, item.fertility) for item in SPECIES_PARAMS],
            dtype=xp.float32,
        )
        self.species_maturation_scale = xp.asarray(
            [self._briere_scale(item.t_min, item.t_max, item.maturation_rate) for item in SPECIES_PARAMS],
            dtype=xp.float32,
        )
        self.thermal_left_width = xp.maximum(self.species_topt - self.species_tmin, 1.0)
        self.thermal_right_width = xp.maximum(self.species_tmax - self.species_topt, 1.0)

        self.vector_comp = xp.asarray(VECTOR_COMPETENCE, dtype=xp.float32)
        self.vector_species_scale = xp.asarray(self._vector_species_scale_array(cfg), dtype=xp.float32)
        self.effective_vector_comp = self.vector_comp * self.vector_species_scale
        self.eip_const = xp.asarray(EIP_THERMAL_CONST, dtype=xp.float32)
        self.eip_threshold = xp.asarray(EIP_THRESHOLD_C, dtype=xp.float32)
        self.incubation_days = xp.asarray(INCUBATION_DAYS, dtype=xp.float32)
        self.infectious_days = xp.asarray(INFECTIOUS_DAYS, dtype=xp.float32)
        self.severe_days = xp.asarray(SEVERE_DAYS, dtype=xp.float32)
        self.base_severe = xp.asarray(BASE_SEVERE_FRACTION, dtype=xp.float32)
        self.base_fatal = xp.asarray(BASE_FATAL_FRACTION, dtype=xp.float32)
        self.base_reporting_rate = xp.asarray(REPORTING_RATE, dtype=xp.float32)
        self.reporting_rate_scale = xp.asarray(
            self._virus_scale_array(cfg.reporting_rate_scale),
            dtype=xp.float32,
        )
        self.reporting_rate = self.base_reporting_rate * self.reporting_rate_scale
        self.host_transmission_scale = xp.asarray(
            self._virus_scale_array(cfg.host_transmission_scale),
            dtype=xp.float32,
        )
        self.vector_transmission_scale = xp.asarray(
            self._virus_scale_array(cfg.vector_transmission_scale),
            dtype=xp.float32,
        )
        (
            seasonal_peak_day,
            seasonal_amplitude,
            seasonal_floor,
        ) = self._seasonality_arrays(cfg)
        self.virus_seasonal_peak_day = xp.asarray(seasonal_peak_day, dtype=xp.float32)
        self.virus_seasonal_amplitude = xp.asarray(seasonal_amplitude, dtype=xp.float32)
        self.virus_seasonal_floor = xp.asarray(seasonal_floor, dtype=xp.float32)
        self.seasonality_enabled = bool(np.any(seasonal_amplitude > 0.0))
        (
            self.habitat_multiplier,
            self.virus_precipitation_sensitivity,
            self.virus_humidity_sensitivity,
        ) = self._build_ecology_modifiers(cfg, static_grid)
        self.ecology_climate_enabled = bool(
            cfg.disease_seasonality_preset == "ecology"
            or cfg.virus_precipitation_sensitivity is not None
            or cfg.virus_humidity_sensitivity is not None
        )
        self.cross_protection = xp.asarray(CROSS_PROTECTION, dtype=xp.float32)
        self.ade_amplification = xp.asarray(ADE_SEVERE_AMPLIFICATION, dtype=xp.float32)
        seasonal_seed_cases = np.asarray(cfg.seasonal_seed_cases or [0.0], dtype=np.float32).reshape(-1)
        if seasonal_seed_cases.size == 1:
            seasonal_seed_cases = np.repeat(seasonal_seed_cases, N_VIRUS)
        self.seasonal_seed_cases_np = seasonal_seed_cases.astype(np.float32)
        self.seasonal_seed_cases = xp.asarray(seasonal_seed_cases, dtype=xp.float32)
        reservoir_force = self._reservoir_force_array(cfg)
        self.reservoir_force_enabled = bool(np.any(reservoir_force > 0.0))
        self.reservoir_force = xp.asarray(reservoir_force, dtype=xp.float32)
        self.importation_schedule = self._load_importation_schedule(cfg.importation_targets, cfg.importation_frequency)
        self.importation_weights = self._build_importation_weights()
        neighbor_src, neighbor_dst, neighbor_counts = self._build_neighbor_edges()
        self.neighbor_src = xp.asarray(neighbor_src, dtype=xp.int64)
        self.neighbor_dst = xp.asarray(neighbor_dst, dtype=xp.int64)
        self.neighbor_counts = xp.asarray(neighbor_counts, dtype=xp.float32)

        self.history_size = int(cfg.climate_memory_days)
        self.initial_state_summary: dict[str, Any] = {
            "default_seeds_applied": False,
            "initial_state_path": cfg.initial_state_path,
            "rows_applied": 0,
            "host_exposed": np.zeros(N_VIRUS, dtype=np.float64),
            "host_infectious": np.zeros(N_VIRUS, dtype=np.float64),
            "host_severe": np.zeros(N_VIRUS, dtype=np.float64),
            "infectious_vectors": np.zeros(N_VIRUS, dtype=np.float64),
        }
        self.state = self._initialize_state()
        if not cfg.disable_default_seeds:
            self._seed_default_reports()
            self.initial_state_summary["default_seeds_applied"] = True
        if cfg.initial_state_path is not None:
            self._seed_initial_state(cfg.initial_state_path)

    def _build_rng(self) -> Any:
        if self.backend.gpu_enabled:
            return self.xp.random.RandomState(self.cfg.seed)
        return np.random.default_rng(self.cfg.seed)

    def _random_uniform(self, shape: tuple[int, ...]) -> Any:
        if self.backend.gpu_enabled:
            return self.xp.asarray(self.rng.random_sample(shape), dtype=self.xp.float32)
        return self.xp.asarray(self.rng.random(shape), dtype=self.xp.float32)

    @staticmethod
    def _virus_scale_array(values: list[float] | None) -> np.ndarray:
        if values is None:
            return np.ones(N_VIRUS, dtype=np.float32)
        scale = np.asarray(values, dtype=np.float32).reshape(-1)
        if scale.size == 1:
            scale = np.repeat(scale, N_VIRUS)
        return scale.astype(np.float32)

    @staticmethod
    def _virus_value_array(values: list[float] | None, default: np.ndarray | float) -> np.ndarray:
        if values is None:
            if isinstance(default, np.ndarray):
                return default.astype(np.float32)
            return np.full(N_VIRUS, float(default), dtype=np.float32)
        parsed = np.asarray(values, dtype=np.float32).reshape(-1)
        if parsed.size == 1:
            parsed = np.repeat(parsed, N_VIRUS)
        if parsed.size != N_VIRUS:
            raise ValueError(f"Expected 1 or {N_VIRUS} values, got {parsed.size}")
        return parsed.astype(np.float32)

    @staticmethod
    def _month_to_peak_day(month_values: np.ndarray) -> np.ndarray:
        peak_days = []
        for raw_month in month_values:
            month = int(round(float(raw_month)))
            month = min(max(month, 1), 12)
            days_before = sum(calendar.monthrange(2021, item)[1] for item in range(1, month))
            peak_days.append(days_before + calendar.monthrange(2021, month)[1] / 2.0)
        return np.asarray(peak_days, dtype=np.float32)

    @classmethod
    def _seasonality_arrays(cls, cfg: SimulationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if cfg.disease_seasonality_preset == "ecology":
            peak_month = np.asarray([9.0, 9.0, 1.0, 8.0, 8.0], dtype=np.float32)
            amplitude = np.asarray([0.55, 0.60, 0.65, 0.95, 0.70], dtype=np.float32)
            floor = np.asarray([0.25, 0.20, 0.15, 0.03, 0.15], dtype=np.float32)
        else:
            peak_month = np.asarray([1.0] * N_VIRUS, dtype=np.float32)
            amplitude = np.zeros(N_VIRUS, dtype=np.float32)
            floor = np.zeros(N_VIRUS, dtype=np.float32)

        peak_month = cls._virus_value_array(cfg.virus_seasonal_peak_month, peak_month)
        amplitude = cls._virus_value_array(cfg.virus_seasonal_amplitude, amplitude)
        floor = cls._virus_value_array(cfg.virus_seasonal_floor, floor)
        return cls._month_to_peak_day(peak_month), amplitude, floor

    @staticmethod
    def _reservoir_force_array(cfg: SimulationConfig) -> np.ndarray:
        if cfg.virus_reservoir_force_scale is not None:
            return VectorizedABMSimulator._virus_value_array(cfg.virus_reservoir_force_scale, 0.0)
        reservoir_force = np.zeros(N_VIRUS, dtype=np.float32)
        reservoir_force[3] = float(cfg.reservoir_force_scale)
        reservoir_force[4] = float(cfg.reservoir_force_scale)
        return reservoir_force

    @staticmethod
    def _vector_species_scale_array(cfg: SimulationConfig) -> np.ndarray:
        if cfg.disease_seasonality_preset == "ecology":
            aedes = np.asarray([1.40, 1.45, 1.20, 0.05, 0.05], dtype=np.float32)
            culex = np.asarray([0.05, 0.05, 0.20, 1.70, 1.70], dtype=np.float32)
            other = np.asarray([0.10, 0.10, 0.70, 0.30, 0.50], dtype=np.float32)
        else:
            aedes = np.ones(N_VIRUS, dtype=np.float32)
            culex = np.ones(N_VIRUS, dtype=np.float32)
            other = np.ones(N_VIRUS, dtype=np.float32)
        aedes = VectorizedABMSimulator._virus_value_array(cfg.virus_aedes_scale, aedes)
        culex = VectorizedABMSimulator._virus_value_array(cfg.virus_culex_scale, culex)
        other = VectorizedABMSimulator._virus_value_array(cfg.virus_other_vector_scale, other)
        scale = np.ones((N_SPECIES, N_VIRUS), dtype=np.float32)
        scale[0] = aedes
        scale[1] = aedes
        scale[2] = culex
        scale[3] = other
        scale[4] = other
        return scale

    def _build_ecology_modifiers(
        self,
        cfg: SimulationConfig,
        static_grid: StaticGrid,
    ) -> tuple[Any, Any, Any]:
        xp = self.xp
        if cfg.disease_seasonality_preset == "ecology":
            urban_pref = np.asarray([1.00, 1.00, 0.35, 0.05, 0.05], dtype=np.float32)
            wetland_pref = np.asarray([0.20, 0.20, 0.20, 0.70, 1.00], dtype=np.float32)
            forest_pref = np.asarray([0.05, 0.05, 1.00, 0.20, 0.30], dtype=np.float32)
            rural_pref = np.asarray([0.10, 0.10, 0.30, 0.40, 0.90], dtype=np.float32)
            habitat_floor = np.asarray([0.30, 0.25, 0.20, 0.08, 0.15], dtype=np.float32)
            precip_sens = np.asarray([0.35, 0.40, 0.25, 0.20, 0.35], dtype=np.float32)
            humidity_sens = np.asarray([0.30, 0.35, 0.20, 0.10, 0.20], dtype=np.float32)
        else:
            urban_pref = np.zeros(N_VIRUS, dtype=np.float32)
            wetland_pref = np.zeros(N_VIRUS, dtype=np.float32)
            forest_pref = np.zeros(N_VIRUS, dtype=np.float32)
            rural_pref = np.zeros(N_VIRUS, dtype=np.float32)
            habitat_floor = np.ones(N_VIRUS, dtype=np.float32)
            precip_sens = np.zeros(N_VIRUS, dtype=np.float32)
            humidity_sens = np.zeros(N_VIRUS, dtype=np.float32)

        urban_pref = self._virus_value_array(cfg.virus_urban_preference, urban_pref)
        wetland_pref = self._virus_value_array(cfg.virus_wetland_preference, wetland_pref)
        forest_pref = self._virus_value_array(cfg.virus_forest_preference, forest_pref)
        rural_pref = self._virus_value_array(cfg.virus_rural_preference, rural_pref)
        habitat_floor = np.clip(self._virus_value_array(cfg.virus_habitat_floor, habitat_floor), 0.0, 1.0)
        precip_sens = self._virus_value_array(cfg.virus_precipitation_sensitivity, precip_sens)
        humidity_sens = self._virus_value_array(cfg.virus_humidity_sensitivity, humidity_sens)

        weight_sum = urban_pref + wetland_pref + forest_pref + rural_pref
        if not np.any(weight_sum > 0.0):
            return (
                xp.ones((int(static_grid.population.size), N_VIRUS), dtype=xp.float32),
                xp.asarray(precip_sens, dtype=xp.float32),
                xp.asarray(humidity_sens, dtype=xp.float32),
            )

        land_type = np.asarray(static_grid.land_type)
        population = np.asarray(static_grid.population, dtype=np.float32)
        runoff = np.asarray(static_grid.runoff_coeff, dtype=np.float32)
        pop_norm = np.clip(np.sqrt(np.maximum(population, 0.0)) / 500.0, 0.0, 1.0)
        urban_like = np.isin(land_type, [1, 2, 5]).astype(np.float32)
        wet_like = np.isin(land_type, [3, 4, 8, 11]).astype(np.float32)
        vegetation_like = np.isin(land_type, [3, 4, 6, 7, 8, 9, 10]).astype(np.float32)

        urban_score = np.clip(0.65 * urban_like + 0.35 * pop_norm, 0.0, 1.0)
        wetland_score = np.clip(0.70 * wet_like + 0.30 * runoff, 0.0, 1.0)
        forest_score = np.clip(0.65 * vegetation_like * (1.0 - pop_norm) + 0.35 * (1.0 - urban_like) * (1.0 - runoff), 0.0, 1.0)
        rural_score = np.clip((1.0 - urban_like) * (0.50 + 0.50 * pop_norm), 0.0, 1.0)

        score = (
            urban_score[:, None] * urban_pref[None, :]
            + wetland_score[:, None] * wetland_pref[None, :]
            + forest_score[:, None] * forest_pref[None, :]
            + rural_score[:, None] * rural_pref[None, :]
        ) / np.maximum(weight_sum[None, :], 1e-6)
        habitat_multiplier = np.where(
            weight_sum[None, :] > 0.0,
            habitat_floor[None, :] + (1.0 - habitat_floor[None, :]) * np.clip(score, 0.0, 1.0),
            1.0,
        )
        return (
            xp.asarray(habitat_multiplier, dtype=xp.float32),
            xp.asarray(precip_sens, dtype=xp.float32),
            xp.asarray(humidity_sens, dtype=xp.float32),
        )

    def set_reporting_rate_scale(self, values: list[float] | np.ndarray) -> None:
        scale = np.asarray(values, dtype=np.float32).reshape(-1)
        if scale.size == 1:
            scale = np.repeat(scale, N_VIRUS)
        if scale.shape != (N_VIRUS,):
            raise ValueError(f"reporting_rate_scale must contain 1 or {N_VIRUS} values")
        if not np.all(np.isfinite(scale)) or np.any(scale < 0.0):
            raise ValueError("reporting_rate_scale values must be finite and >= 0")
        self.reporting_rate_scale = self.xp.asarray(scale, dtype=self.xp.float32)
        self.reporting_rate = self.base_reporting_rate * self.reporting_rate_scale

    def current_reporting_rate_scale(self) -> list[float]:
        return [
            float(value)
            for value in np.asarray(self.backend.asnumpy(self.reporting_rate_scale), dtype=np.float32)
        ]

    def _load_importation_schedule(self, path: str | None, frequency: str) -> dict[str, np.ndarray]:
        if path is None or self.cfg.importation_fraction <= 0.0:
            return {}
        observations = load_validation_observations(path, frequency=frequency)
        schedule: dict[str, np.ndarray] = {}
        for obs in observations:
            values = schedule.setdefault(obs.period, np.zeros(N_VIRUS, dtype=np.float32))
            values[obs.virus_index] += np.float32(obs.observed_cases)
        return schedule

    def _build_importation_weights(self) -> Any:
        xp = self.xp
        n_cells = int(self.grid.population.size)
        if n_cells == 0:
            return xp.zeros((0, N_VIRUS), dtype=xp.float32)
        population_weight = xp.sqrt(xp.asarray(self.grid.population, dtype=xp.float32) + 1.0)
        habitat_strength = self.land_pref[self.land_type]
        vector_suitability = xp.maximum(habitat_strength @ self.effective_vector_comp, 0.0) * self.habitat_multiplier
        weights = vector_suitability * population_weight[:, None]
        fallback = population_weight / xp.maximum(xp.sum(population_weight), 1e-6)
        totals = xp.sum(weights, axis=0)
        weights = xp.where(totals[None, :] > 0.0, weights / xp.maximum(totals[None, :], 1e-6), fallback[:, None])
        return weights.astype(xp.float32)

    def _build_neighbor_edges(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        active_linear = np.asarray(self.grid.active_linear_idx, dtype=np.int64)
        n_cells = int(active_linear.size)
        if n_cells == 0:
            return (
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32),
            )
        rows, cols = self.grid.shape_2d
        full_to_active = {int(full_idx): active_idx for active_idx, full_idx in enumerate(active_linear)}
        src: list[int] = []
        dst: list[int] = []
        for active_idx, full_idx in enumerate(active_linear):
            row = int(full_idx // cols)
            col = int(full_idx % cols)
            for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                n_row = row + d_row
                n_col = col + d_col
                if not (0 <= n_row < rows and 0 <= n_col < cols):
                    continue
                neighbor = full_to_active.get(n_row * cols + n_col)
                if neighbor is None:
                    continue
                src.append(active_idx)
                dst.append(neighbor)
        counts = np.zeros(n_cells, dtype=np.float32)
        if dst:
            np.add.at(counts, np.asarray(dst, dtype=np.int64), 1.0)
        return np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64), counts

    @staticmethod
    def _date_from_climate_label(date_label: str | None) -> datetime | None:
        if date_label is None:
            return None
        text = str(date_label)
        for fmt in ("%Y%m%d", "%Y-%m-%d"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _period_key(date_value: datetime, frequency: str) -> str:
        if frequency == "year":
            return date_value.strftime("%Y")
        if frequency == "month":
            return date_value.strftime("%Y-%m")
        return date_value.strftime("%Y-%m-%d")

    @staticmethod
    def _period_days(date_value: datetime, frequency: str) -> int:
        if frequency == "year":
            return 366 if calendar.isleap(date_value.year) else 365
        if frequency == "month":
            return calendar.monthrange(date_value.year, date_value.month)[1]
        return 1

    def _virus_seasonal_multiplier(self, date_label: str | None) -> Any:
        xp = self.xp
        if not self.seasonality_enabled:
            return xp.ones(N_VIRUS, dtype=xp.float32)
        date_value = self._date_from_climate_label(date_label)
        if date_value is None:
            return xp.ones(N_VIRUS, dtype=xp.float32)
        day_of_year = xp.asarray(float(date_value.timetuple().tm_yday), dtype=xp.float32)
        phase = (
            2.0
            * xp.asarray(np.pi, dtype=xp.float32)
            * (day_of_year - self.virus_seasonal_peak_day)
            / xp.asarray(365.25, dtype=xp.float32)
        )
        raw = 1.0 + self.virus_seasonal_amplitude * xp.cos(phase)
        return xp.maximum(raw, self.virus_seasonal_floor).astype(xp.float32)

    def _virus_climate_multiplier(self, rolling_precip: Any, rolling_humidity: Any) -> Any:
        xp = self.xp
        if not self.ecology_climate_enabled:
            return xp.ones((rolling_precip.shape[0], N_VIRUS), dtype=xp.float32)
        precip_signal = xp.clip(rolling_precip / 120.0, 0.0, 2.0) - 0.35
        humidity_signal = xp.clip((rolling_humidity - 55.0) / 45.0, -1.0, 1.0)
        multiplier = (
            1.0
            + precip_signal[:, None] * self.virus_precipitation_sensitivity[None, :]
            + humidity_signal[:, None] * self.virus_humidity_sensitivity[None, :]
        )
        return xp.clip(multiplier, 0.05, 3.0).astype(xp.float32)

    def _external_exposed_for_date(self, date_label: str | None) -> Any:
        xp = self.xp
        daily = np.zeros(N_VIRUS, dtype=np.float32)
        date_value = self._date_from_climate_label(date_label)
        if date_value is None:
            return xp.asarray(daily, dtype=xp.float32)
        if self.importation_schedule:
            period = self._period_key(date_value, self.cfg.importation_frequency)
            observed = self.importation_schedule.get(period)
            if observed is not None:
                daily += observed * np.float32(self.cfg.importation_fraction / self._period_days(date_value, self.cfg.importation_frequency))
        if (
            self.cfg.seasonal_seed_cases is not None
            and date_value.month == self.cfg.seasonal_seed_month
            and date_value.day == self.cfg.seasonal_seed_day
        ):
            daily += self.seasonal_seed_cases_np
        return xp.asarray(daily, dtype=xp.float32)

    def _mix_neighbor_pressure(self, values: Any) -> Any:
        if self.cfg.spatial_diffusion_rate <= 0.0 or self.neighbor_src.size == 0:
            return values
        xp = self.xp
        neighbor_sum = xp.zeros_like(values)
        xp.add.at(neighbor_sum, self.neighbor_dst, values[self.neighbor_src])
        neighbor_mean = neighbor_sum / xp.maximum(self.neighbor_counts[:, None], 1.0)
        rate = xp.asarray(self.cfg.spatial_diffusion_rate, dtype=xp.float32)
        return values * (1.0 - rate) + neighbor_mean * rate

    @staticmethod
    def _briere_peak(t_min: float, t_max: float) -> float:
        temps = np.linspace(t_min + 1e-3, t_max - 1e-3, 512, dtype=np.float32)
        raw = temps * np.maximum(temps - t_min, 0.0) * np.sqrt(np.maximum(t_max - temps, 0.0))
        return float(np.maximum(raw.max(), 1e-6))

    @staticmethod
    def _briere_scale(t_min: float, t_max: float, target_rate: float, reference_temp: float = 25.0) -> float:
        reference = float(np.clip(reference_temp, t_min + 0.5, t_max - 0.5))
        denominator = reference * max(reference - t_min, 1e-6) * np.sqrt(max(t_max - reference, 1e-6))
        return float(target_rate / max(denominator, 1e-6))

    def _initialize_state(self) -> SimulationState:
        xp = self.xp
        n_cells = self.grid.population.size
        host_population = xp.asarray(self.grid.population * self.cfg.host_scale, dtype=xp.float32)
        host_s = xp.repeat(host_population[:, None], N_VIRUS, axis=1)
        host_e = xp.zeros((n_cells, N_VIRUS), dtype=xp.float32)
        host_i = xp.zeros((n_cells, N_VIRUS), dtype=xp.float32)
        host_severe = xp.zeros((n_cells, N_VIRUS), dtype=xp.float32)
        host_r = xp.zeros((n_cells, N_VIRUS), dtype=xp.float32)
        host_d = xp.zeros((n_cells, N_VIRUS), dtype=xp.float32)

        habitat_strength = self.land_pref[self.land_type]
        adult_jitter = 0.92 + 0.16 * self._random_uniform((n_cells, N_SPECIES))
        carrying_base = xp.sqrt(host_population[:, None] + 1.0) * habitat_strength * self.cfg.mosquito_scale * 120.0 * adult_jitter
        mosquito_adult = xp.maximum(carrying_base, 0.0).astype(xp.float32)
        mosquito_aquatic = (mosquito_adult * (1.20 + 0.25 * self._random_uniform((n_cells, N_SPECIES)))).astype(xp.float32)
        mosquito_exposed = xp.zeros((n_cells, N_SPECIES, N_VIRUS), dtype=xp.float32)
        mosquito_infectious = xp.zeros((n_cells, N_SPECIES, N_VIRUS), dtype=xp.float32)
        water_jitter = (self._random_uniform((n_cells,)) - 0.5) * 0.06
        water_level = xp.clip(0.18 + 0.55 * self.runoff_coeff + water_jitter, 0.0, 1.5).astype(xp.float32)
        cumulative_reported = xp.zeros(N_VIRUS, dtype=xp.float32)

        hist_shape = (self.history_size, n_cells)
        climate_temperature_hist = xp.zeros(hist_shape, dtype=xp.float32)
        climate_precip_hist = xp.zeros(hist_shape, dtype=xp.float32)
        climate_humidity_hist = xp.zeros(hist_shape, dtype=xp.float32)

        return SimulationState(
            host_population=host_population,
            host_s=host_s,
            host_e=host_e,
            host_i=host_i,
            host_severe=host_severe,
            host_r=host_r,
            host_d=host_d,
            mosquito_aquatic=mosquito_aquatic,
            mosquito_adult=mosquito_adult,
            mosquito_exposed=mosquito_exposed,
            mosquito_infectious=mosquito_infectious,
            water_level=water_level,
            cumulative_reported=cumulative_reported,
            climate_temperature_hist=climate_temperature_hist,
            climate_precip_hist=climate_precip_hist,
            climate_humidity_hist=climate_humidity_hist,
            climate_hist_cursor=0,
            climate_hist_count=0,
        )

    @staticmethod
    def _normalise_initial_key(value: str) -> str:
        return str(value).strip().lower().replace("-", "_").replace(" ", "_")

    @staticmethod
    def _normalise_virus_name(value: Any) -> str:
        normalized = VectorizedABMSimulator._normalise_initial_key(str(value))
        aliases = {
            "yellowfever": "yellow_fever",
            "yellow_fever_virus": "yellow_fever",
            "westnile": "west_nile",
            "west_nile_virus": "west_nile",
            "japaneseencephalitis": "japanese_encephalitis",
            "japanese_encephalitis_virus": "japanese_encephalitis",
            "je": "japanese_encephalitis",
        }
        return aliases.get(normalized, aliases.get(normalized.replace("_", ""), normalized))

    @staticmethod
    def _virus_index_from_initial_value(value: Any) -> int:
        text = str(value).strip()
        if text == "":
            raise ValueError("initial state row has an empty virus value")
        if text.isdigit():
            index = int(text)
            if 0 <= index < N_VIRUS:
                return index
            raise ValueError(f"virus index must be in [0, {N_VIRUS - 1}], got {index}")
        normalized = VectorizedABMSimulator._normalise_virus_name(text)
        for index, name in enumerate(VIRUS_NAMES):
            if normalized == VectorizedABMSimulator._normalise_virus_name(name):
                return index
        raise ValueError(f"Unknown virus name in initial state: {value!r}")

    @staticmethod
    def _initial_float(
        row: dict[str, Any],
        keys: tuple[str, ...],
        default: float | None = None,
        allow_negative: bool = False,
    ) -> float | None:
        for key in keys:
            value = row.get(key)
            if value is None or str(value).strip() == "":
                continue
            text = str(value).strip().replace(",", "")
            try:
                number = float(text)
            except ValueError as exc:
                raise ValueError(f"Initial state value {key!r} must be numeric, got {value!r}") from exc
            if not np.isfinite(number) or (number < 0.0 and not allow_negative):
                requirement = "finite" if allow_negative else "finite and >= 0"
                raise ValueError(f"Initial state value {key!r} must be {requirement}, got {number!r}")
            return number
        return default

    @staticmethod
    def _initial_date(value: Any) -> datetime | None:
        text = str(value).strip()
        if text == "":
            return None
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y-%m", "%Y/%m", "%Y"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    def _initial_window_days(self, row: dict[str, Any]) -> int:
        explicit = self._initial_float(row, ("window_days", "days", "duration_days"), None)
        if explicit is not None:
            return max(int(round(explicit)), 1)

        start = self._initial_date(row.get("period_start") or row.get("start_date"))
        end = self._initial_date(row.get("period_end") or row.get("end_date"))
        if start is not None and end is not None:
            return max((end - start).days + 1, 1)

        period = row.get("period") or row.get("year")
        period_dt = self._initial_date(period) if period is not None else None
        if period_dt is not None:
            text = str(period).strip()
            if len(text) == 4:
                return 366 if calendar.isleap(period_dt.year) else 365
            if len(text) in (7,):
                return calendar.monthrange(period_dt.year, period_dt.month)[1]
        return int(self.cfg.initial_state_window_days)

    def _load_initial_state_rows(self, path: str | Path) -> list[dict[str, Any]]:
        state_path = Path(path)
        if not state_path.exists():
            raise FileNotFoundError(f"Initial state file does not exist: {state_path}")
        if state_path.suffix.lower() != ".csv":
            raise ValueError("Initial state currently supports CSV files only.")
        with state_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("Initial state CSV requires a header row.")
            rows: list[dict[str, Any]] = []
            for row in reader:
                if not any(value is not None and str(value).strip() != "" for value in row.values()):
                    continue
                rows.append({self._normalise_initial_key(key): value for key, value in row.items()})
            return rows

    def _row_coordinate_cell(self, row: dict[str, Any]) -> tuple[int | None, bool]:
        lat = self._initial_float(row, ("lat", "latitude", "y"), None, allow_negative=True)
        lon = self._initial_float(row, ("lon", "longitude", "x"), None, allow_negative=True)
        if lat is None and lon is None:
            return None, False
        if lat is None or lon is None:
            raise ValueError("Initial state rows must provide both lat and lon when using coordinates.")
        if not self._seed_in_bbox(float(lat), float(lon)):
            return None, True
        return self._nearest_active_cell(float(lat), float(lon)), True

    def _initial_cell_weights(self, row: dict[str, Any], virus: int) -> tuple[Any | None, bool]:
        xp = self.xp
        n_cells = int(self.grid.population.size)
        if n_cells == 0:
            return None, True
        cell_index, has_coordinates = self._row_coordinate_cell(row)
        if has_coordinates:
            if cell_index is None:
                return None, True
            weights = xp.zeros(n_cells, dtype=xp.float32)
            weights[cell_index] = 1.0
            return weights, False
        return self.importation_weights[:, virus], False

    def _apply_initial_host_state(
        self,
        virus: int,
        weights: Any,
        exposed: float,
        infectious: float,
        severe: float,
    ) -> tuple[float, float, float]:
        xp = self.xp
        st = self.state
        add_e = weights * xp.asarray(exposed, dtype=xp.float32)
        add_i = weights * xp.asarray(infectious, dtype=xp.float32)
        add_s = weights * xp.asarray(severe, dtype=xp.float32)
        requested = add_e + add_i + add_s
        scale = xp.minimum(1.0, st.host_s[:, virus] / xp.maximum(requested, 1e-6))
        add_e = add_e * scale
        add_i = add_i * scale
        add_s = add_s * scale
        total = add_e + add_i + add_s
        st.host_s[:, virus] = xp.maximum(st.host_s[:, virus] - total, 0.0)
        st.host_e[:, virus] = st.host_e[:, virus] + add_e
        st.host_i[:, virus] = st.host_i[:, virus] + add_i
        st.host_severe[:, virus] = st.host_severe[:, virus] + add_s
        return (
            float(np.sum(self.backend.asnumpy(add_e))),
            float(np.sum(self.backend.asnumpy(add_i))),
            float(np.sum(self.backend.asnumpy(add_s))),
        )

    def _apply_initial_vectors(
        self,
        row: dict[str, Any],
        virus: int,
        total_vectors: float | None,
        vector_seed_rate: float,
    ) -> float:
        if total_vectors is not None and total_vectors <= 0.0:
            return 0.0
        if total_vectors is None and vector_seed_rate <= 0.0:
            return 0.0
        xp = self.xp
        st = self.state
        n_cells = int(self.grid.population.size)
        if n_cells == 0:
            return 0.0

        occupancy = xp.sum(st.mosquito_exposed + st.mosquito_infectious, axis=2)
        clean_adults = xp.clip(st.mosquito_adult - occupancy, 0.0, None)
        suitability = clean_adults * self.vector_comp[None, :, virus]

        cell_index, has_coordinates = self._row_coordinate_cell(row)
        if has_coordinates:
            if cell_index is None:
                return 0.0
            mask = xp.zeros((n_cells, N_SPECIES), dtype=xp.float32)
            mask[cell_index, :] = 1.0
            suitability = suitability * mask

        if total_vectors is None:
            add = suitability * xp.asarray(vector_seed_rate, dtype=xp.float32)
        else:
            total_suitability = xp.sum(suitability)
            if float(self.backend.asnumpy(total_suitability)) <= 0.0:
                return 0.0
            add = suitability / xp.maximum(total_suitability, 1e-6) * xp.asarray(total_vectors, dtype=xp.float32)
        add = xp.minimum(add, clean_adults)
        st.mosquito_infectious[:, :, virus] = st.mosquito_infectious[:, :, virus] + add
        return float(np.sum(self.backend.asnumpy(add)))

    def _seed_initial_state(self, path: str | Path) -> None:
        rows = self._load_initial_state_rows(path)
        for row in rows:
            virus_value = row.get("virus") or row.get("virus_name") or row.get("disease") or row.get("pathogen")
            if virus_value is None or str(virus_value).strip() == "":
                raise ValueError(f"Initial state row is missing a virus column: {row!r}")
            virus = self._virus_index_from_initial_value(virus_value)
            weights, skipped = self._initial_cell_weights(row, virus)
            if skipped or weights is None:
                continue

            direct_host_values = any(
                row.get(key) is not None and str(row.get(key)).strip() != ""
                for key in (
                    "initial_exposed",
                    "exposed",
                    "e0",
                    "initial_infectious",
                    "infectious",
                    "i0",
                    "initial_severe",
                    "severe",
                    "severe0",
                )
            )
            if direct_host_values:
                exposed = self._initial_float(row, ("initial_exposed", "exposed", "e0"), 0.0) or 0.0
                infectious = self._initial_float(row, ("initial_infectious", "infectious", "i0"), 0.0) or 0.0
                severe = self._initial_float(row, ("initial_severe", "severe", "severe0"), 0.0) or 0.0
                reported_cases = self._initial_float(
                    row,
                    ("reported_cases", "observed_cases", "reported", "cases", "target"),
                    0.0,
                ) or 0.0
            else:
                reported_cases = self._initial_float(
                    row,
                    ("reported_cases", "observed_cases", "reported", "cases", "target"),
                    0.0,
                ) or 0.0
                reporting_rate = self._initial_float(row, ("reporting_rate", "report_rate"), None)
                if reporting_rate is None:
                    reporting_rate = float(np.asarray(self.backend.asnumpy(self.reporting_rate))[virus])
                reporting_rate = max(float(reporting_rate), 1e-6)
                window_days = self._initial_window_days(row)
                daily_infections = reported_cases / reporting_rate / max(float(window_days), 1.0)
                exposed = daily_infections * float(INCUBATION_DAYS[virus])
                infectious = daily_infections * float(INFECTIOUS_DAYS[virus])
                severe = daily_infections * float(BASE_SEVERE_FRACTION[virus]) * float(SEVERE_DAYS[virus])

            applied_e, applied_i, applied_severe = self._apply_initial_host_state(
                virus=virus,
                weights=weights,
                exposed=exposed,
                infectious=infectious,
                severe=severe,
            )

            vector_seed_rate = self._initial_float(row, ("vector_seed_rate", "infectious_vector_rate"), None)
            if vector_seed_rate is None:
                vector_seed_rate = float(self.cfg.initial_vector_seed_rate)
            total_vectors = self._initial_float(
                row,
                ("initial_infectious_vectors", "infectious_vectors", "vector_infectious", "v0"),
                None,
            )
            applied_vectors = self._apply_initial_vectors(row, virus, total_vectors, float(vector_seed_rate))

            if self.cfg.initial_state_include_reported and reported_cases > 0.0:
                self.state.cumulative_reported[virus] += np.float32(reported_cases)

            self.initial_state_summary["rows_applied"] = int(self.initial_state_summary["rows_applied"]) + 1
            self.initial_state_summary["host_exposed"][virus] += applied_e
            self.initial_state_summary["host_infectious"][virus] += applied_i
            self.initial_state_summary["host_severe"][virus] += applied_severe
            self.initial_state_summary["infectious_vectors"][virus] += applied_vectors

    def _seed_default_reports(self) -> None:
        if self.grid.population.size == 0:
            return
        xp = self.xp
        for record in DEFAULT_SEED_REPORTS:
            if not self._seed_in_bbox(record["lat"], record["lon"]):
                continue

            cell_index = self._nearest_active_cell(record["lat"], record["lon"])
            virus = int(record["virus"])
            focal_species = int(record["species"])
            cases = float(record["initial_reported_cases"])

            current_s = float(self.backend.asnumpy(self.state.host_s[cell_index, virus]))
            exposed = min(cases, current_s)
            self.state.host_s[cell_index, virus] -= exposed
            self.state.host_e[cell_index, virus] += exposed

            vector_weights = xp.asarray(VECTOR_COMPETENCE[:, virus], dtype=xp.float32) + 0.02
            vector_weights[focal_species] += 0.30
            vector_weights = vector_weights * (0.90 + 0.20 * self._random_uniform((N_SPECIES,)))
            vector_weights = vector_weights / xp.maximum(xp.sum(vector_weights), 1e-6)
            base_infectious = max(16.0, np.sqrt(cases) * 2.0)
            self.state.mosquito_infectious[cell_index, :, virus] += base_infectious * vector_weights
            self.state.cumulative_reported[virus] += exposed

    def _seed_in_bbox(self, lat: float, lon: float) -> bool:
        lat_lo, lat_hi, lon_lo, lon_hi = self.grid.bbox
        return lat_lo <= lat <= lat_hi and lon_lo <= lon <= lon_hi

    def _nearest_active_cell(self, lat: float, lon: float) -> int:
        lat_delta = self.grid.active_lat - np.float32(lat)
        lon_delta = self.grid.active_lon - np.float32(lon)
        return int(np.argmin(lat_delta * lat_delta + lon_delta * lon_delta))

    def _species_temperature_response(self, temp_c: Any) -> Any:
        xp = self.xp
        temp = temp_c[:, None]
        raw = temp * xp.maximum(temp - self.species_tmin[None, :], 0.0) * xp.sqrt(xp.maximum(self.species_tmax[None, :] - temp, 0.0))
        response = raw / xp.maximum(self.species_briere_peak[None, :], 1e-6)
        valid = (temp > self.species_tmin[None, :]) & (temp < self.species_tmax[None, :])
        return xp.where(valid, xp.clip(response, 0.0, 1.0), 0.0).astype(xp.float32)

    def _species_briere_rate(self, temp_c: Any, scale: Any) -> Any:
        xp = self.xp
        temp = temp_c[:, None]
        raw = scale[None, :] * temp * xp.maximum(temp - self.species_tmin[None, :], 0.0) * xp.sqrt(xp.maximum(self.species_tmax[None, :] - temp, 0.0))
        valid = (temp > self.species_tmin[None, :]) & (temp < self.species_tmax[None, :])
        return xp.where(valid, xp.maximum(raw, 0.0), 0.0).astype(xp.float32)

    def _species_humidity_response(self, humidity_pct: Any) -> Any:
        xp = self.xp
        return xp.clip((humidity_pct[:, None] - 20.0) / 60.0, 0.1, 1.2).astype(xp.float32)

    def _species_thermal_mortality(self, temp_c: Any, humidity_pct: Any) -> Any:
        xp = self.xp
        temp = temp_c[:, None]
        cold = xp.maximum(self.species_topt[None, :] - temp, 0.0) / self.thermal_left_width[None, :]
        heat = xp.maximum(temp - self.species_topt[None, :], 0.0) / self.thermal_right_width[None, :]
        thermal_penalty = 1.0 + 1.8 * cold * cold + 2.4 * heat * heat
        dryness_stress = xp.maximum(0.0, 45.0 - humidity_pct[:, None]) * 0.0015
        return xp.clip(self.species_mortality[None, :] * thermal_penalty + dryness_stress, 0.0, 0.95).astype(xp.float32)

    def _update_climate_history(self, temp_c: Any, precip_mm: Any, humidity_pct: Any) -> tuple[Any, Any, Any]:
        st = self.state
        idx = st.climate_hist_cursor
        st.climate_temperature_hist[idx] = temp_c
        st.climate_precip_hist[idx] = precip_mm
        st.climate_humidity_hist[idx] = humidity_pct
        st.climate_hist_cursor = (idx + 1) % self.history_size
        st.climate_hist_count = min(st.climate_hist_count + 1, self.history_size)

        hist_len = st.climate_hist_count
        hist_slice = slice(0, hist_len)
        rolling_temp = self.xp.mean(st.climate_temperature_hist[hist_slice], axis=0)
        rolling_precip = self.xp.sum(st.climate_precip_hist[hist_slice], axis=0)
        rolling_humidity = self.xp.mean(st.climate_humidity_hist[hist_slice], axis=0)
        return rolling_temp.astype(self.xp.float32), rolling_precip.astype(self.xp.float32), rolling_humidity.astype(self.xp.float32)

    def step_day(self, temp_c: Any, precip_mm: Any, humidity_pct: Any, date_label: str | None = None) -> dict[str, Any]:
        xp = self.xp
        st = self.state

        if st.host_population.size == 0:
            return {
                "new_exposed": xp.zeros(N_VIRUS, dtype=xp.float32),
                "new_infectious": xp.zeros(N_VIRUS, dtype=xp.float32),
                "new_severe": xp.zeros(N_VIRUS, dtype=xp.float32),
                "new_deaths": xp.zeros(N_VIRUS, dtype=xp.float32),
                "reported": xp.zeros(N_VIRUS, dtype=xp.float32),
                "adult_vectors": xp.zeros(N_SPECIES, dtype=xp.float32),
                "infectious_vectors": xp.zeros(N_VIRUS, dtype=xp.float32),
                "water_level_mean": xp.asarray(0.0, dtype=xp.float32),
                "temp_mean": xp.asarray(0.0, dtype=xp.float32),
                "precip_sum": xp.asarray(0.0, dtype=xp.float32),
            }

        if self.cfg.immunity_waning_rate is not None:
            waned = st.host_r * xp.asarray(self.cfg.immunity_waning_rate, dtype=xp.float32)
            st.host_r = xp.maximum(st.host_r - waned, 0.0)
            st.host_s = xp.minimum(st.host_s + waned, st.host_population[:, None])
        if self.cfg.host_replenishment_rate > 0.0:
            replenished = st.host_d * xp.asarray(self.cfg.host_replenishment_rate, dtype=xp.float32)
            st.host_d = xp.maximum(st.host_d - replenished, 0.0)
            st.host_s = xp.minimum(st.host_s + replenished, st.host_population[:, None])

        rolling_temp, rolling_precip, rolling_humidity = self._update_climate_history(temp_c, precip_mm, humidity_pct)
        seasonal_multiplier = self._virus_seasonal_multiplier(date_label)
        ecology_multiplier = (
            self.habitat_multiplier
            * self._virus_climate_multiplier(rolling_precip, rolling_humidity)
            * seasonal_multiplier[None, :]
        )
        temp_response = self._species_temperature_response(rolling_temp)
        humidity_response = self._species_humidity_response(rolling_humidity)
        bite_rates = self._species_briere_rate(rolling_temp, self.species_bite_scale) * humidity_response
        fertility_rates = self._species_briere_rate(rolling_temp, self.species_fertility_scale)
        maturation_rates = self._species_briere_rate(rolling_temp, self.species_maturation_scale)

        evaporation = xp.clip(0.0045 * xp.maximum(rolling_temp + 5.0, 0.0) * (100.0 - rolling_humidity) / 100.0, 0.0, 0.35)
        runoff_input = rolling_precip * (0.010 + 0.018 * self.runoff_coeff)
        seepage = st.water_level * (0.025 + 0.035 * self.runoff_coeff)
        st.water_level = xp.clip(st.water_level + runoff_input - seepage - evaporation, 0.0, 1.5)

        breeding_multiplier = xp.clip(0.15 + st.water_level[:, None] + xp.clip(rolling_precip[:, None] / 100.0, 0.0, 0.5), 0.08, 2.20)
        carrying = (
            xp.sqrt(st.host_population[:, None] + 1.0)
            * self.land_pref[self.land_type]
            * self.species_k[None, :]
            * breeding_multiplier
            * self.cfg.mosquito_scale
            * 140.0
        )
        carrying = xp.maximum(carrying, 1.0)

        egg_laying = st.mosquito_adult * fertility_rates * breeding_multiplier
        aquatic_decay = st.mosquito_aquatic * xp.clip(0.04 + xp.maximum(0.0, 16.0 - rolling_temp[:, None]) * 0.003, 0.03, 0.22)
        st.mosquito_aquatic = xp.maximum(st.mosquito_aquatic + egg_laying - aquatic_decay, 0.0)

        matured = xp.minimum(st.mosquito_aquatic, st.mosquito_aquatic * maturation_rates)
        st.mosquito_aquatic = xp.maximum(st.mosquito_aquatic - matured, 0.0)

        daily_mortality = self._species_thermal_mortality(rolling_temp, rolling_humidity)
        survival = 1.0 - daily_mortality

        st.mosquito_adult = xp.maximum(st.mosquito_adult * survival + matured, 0.0)
        st.mosquito_exposed = st.mosquito_exposed * survival[:, :, None]
        st.mosquito_infectious = st.mosquito_infectious * survival[:, :, None]

        occupancy = xp.sum(st.mosquito_exposed + st.mosquito_infectious, axis=2)
        scale_to_capacity = xp.minimum(1.0, carrying / xp.maximum(st.mosquito_adult, 1e-6))
        st.mosquito_adult = st.mosquito_adult * scale_to_capacity
        st.mosquito_exposed = st.mosquito_exposed * scale_to_capacity[:, :, None]
        st.mosquito_infectious = st.mosquito_infectious * scale_to_capacity[:, :, None]

        temp_above_threshold = xp.maximum(rolling_temp[:, None, None] - self.eip_threshold[None, :, :], 0.0)
        eip_days = xp.where(temp_above_threshold > 0.0, self.eip_const[None, :, :] / xp.maximum(temp_above_threshold, 1e-3), xp.inf)
        exposed_to_infectious = st.mosquito_exposed * xp.clip(1.0 / eip_days, 0.0, 1.0)
        st.mosquito_exposed = xp.maximum(st.mosquito_exposed - exposed_to_infectious, 0.0)
        st.mosquito_infectious = st.mosquito_infectious + exposed_to_infectious

        infectious_pressure = xp.sum(st.mosquito_infectious * bite_rates[:, :, None] * self.effective_vector_comp[None, :, :], axis=1)
        infectious_pressure = self._mix_neighbor_pressure(infectious_pressure)

        live_hosts = xp.maximum(st.host_population - xp.sum(st.host_d, axis=1), 1.0)
        active_host_cases = st.host_e + st.host_i + st.host_severe
        free_hosts = xp.clip(live_hosts - xp.sum(active_host_cases, axis=1), 0.0, None)
        recovered_fraction = st.host_r / live_hosts[:, None]
        susceptibility_modifier = xp.clip(1.0 - recovered_fraction @ self.cross_protection, 0.2, 1.0)
        effective_sus = xp.minimum(st.host_s, free_hosts[:, None]) * susceptibility_modifier

        host_foi = xp.clip(
            infectious_pressure / live_hosts[:, None]
            * self.cfg.host_transmission_coeff
            * self.host_transmission_scale[None, :]
            * ecology_multiplier,
            0.0,
            0.95,
        )
        new_host_exposed = effective_sus * host_foi
        external_exposed_total = self._external_exposed_for_date(date_label)
        new_host_exposed = new_host_exposed + self.importation_weights * external_exposed_total[None, :]
        new_host_exposed = xp.minimum(new_host_exposed, st.host_s)
        exposure_scale = xp.minimum(1.0, free_hosts[:, None] / xp.maximum(xp.sum(new_host_exposed, axis=1, keepdims=True), 1e-6))
        new_host_exposed = new_host_exposed * exposure_scale

        host_e_before = st.host_e
        st.host_s = xp.maximum(st.host_s - new_host_exposed, 0.0)
        st.host_e = st.host_e + new_host_exposed

        exposed_to_infectious_hosts = host_e_before * xp.clip(1.0 / self.incubation_days[None, :], 0.0, 1.0)
        st.host_e = xp.maximum(st.host_e - exposed_to_infectious_hosts, 0.0)
        host_i_before = st.host_i
        st.host_i = st.host_i + exposed_to_infectious_hosts

        secondary_risk = recovered_fraction @ self.ade_amplification
        severe_fraction = xp.clip(self.base_severe[None, :] * (1.0 + secondary_risk), 0.0, 0.95)
        infectious_resolve = host_i_before * xp.clip(1.0 / self.infectious_days[None, :], 0.0, 1.0)
        new_severe = infectious_resolve * severe_fraction
        new_recovered = infectious_resolve - new_severe
        st.host_i = xp.maximum(st.host_i - infectious_resolve, 0.0)
        host_severe_before = st.host_severe
        st.host_severe = st.host_severe + new_severe
        st.host_r = st.host_r + new_recovered

        severe_resolve = host_severe_before * xp.clip(1.0 / self.severe_days[None, :], 0.0, 1.0)
        fatal_fraction = xp.clip(self.base_fatal[None, :] * (1.0 + 0.5 * secondary_risk), 0.0, 0.98)
        new_deaths = severe_resolve * fatal_fraction
        recovered_from_severe = severe_resolve - new_deaths
        st.host_severe = xp.maximum(st.host_severe - severe_resolve, 0.0)
        st.host_r = st.host_r + recovered_from_severe
        st.host_d = st.host_d + new_deaths

        host_infectious = st.host_i + st.host_severe
        occupancy = xp.sum(st.mosquito_exposed + st.mosquito_infectious, axis=2)
        clean_adults = xp.clip(st.mosquito_adult - occupancy, 0.0, None)
        vector_foi = (
            host_infectious[:, None, :] / live_hosts[:, None, None]
        ) * bite_rates[:, :, None] * self.effective_vector_comp[None, :, :] * self.cfg.vector_transmission_coeff * self.vector_transmission_scale[None, None, :] * ecology_multiplier[:, None, :]
        if self.reservoir_force_enabled:
            reservoir_foi = (
                self.reservoir_force[None, None, :]
                * temp_response[:, :, None]
                * humidity_response[:, :, None]
                * self.effective_vector_comp[None, :, :]
                * ecology_multiplier[:, None, :]
            )
            vector_foi = vector_foi + reservoir_foi
        new_vector_exposed = clean_adults[:, :, None] * xp.clip(vector_foi, 0.0, 0.95)
        vector_scale = xp.minimum(1.0, clean_adults[:, :, None] / xp.maximum(xp.sum(new_vector_exposed, axis=2, keepdims=True), 1e-6))
        new_vector_exposed = new_vector_exposed * vector_scale
        st.mosquito_exposed = st.mosquito_exposed + new_vector_exposed

        reported_today = xp.sum((exposed_to_infectious_hosts + new_severe) * self.reporting_rate[None, :], axis=0)
        st.cumulative_reported = st.cumulative_reported + reported_today

        return {
            "new_exposed": xp.sum(new_host_exposed, axis=0),
            "new_infectious": xp.sum(exposed_to_infectious_hosts, axis=0),
            "new_severe": xp.sum(new_severe, axis=0),
            "new_deaths": xp.sum(new_deaths, axis=0),
            "reported": reported_today,
            "adult_vectors": xp.sum(st.mosquito_adult, axis=0),
            "infectious_vectors": xp.sum(st.mosquito_infectious, axis=(0, 1)),
            "water_level_mean": xp.mean(st.water_level),
            "temp_mean": xp.mean(rolling_temp),
            "precip_sum": xp.sum(rolling_precip),
        }

    def run_window(self, climate_window: ClimateWindow) -> dict[str, np.ndarray]:
        xp = self.xp
        n_days = len(climate_window.dates)
        daily = {
            "new_exposed": xp.zeros((n_days, N_VIRUS), dtype=xp.float32),
            "new_infectious": xp.zeros((n_days, N_VIRUS), dtype=xp.float32),
            "new_severe": xp.zeros((n_days, N_VIRUS), dtype=xp.float32),
            "new_deaths": xp.zeros((n_days, N_VIRUS), dtype=xp.float32),
            "reported": xp.zeros((n_days, N_VIRUS), dtype=xp.float32),
            "adult_vectors": xp.zeros((n_days, N_SPECIES), dtype=xp.float32),
            "infectious_vectors": xp.zeros((n_days, N_VIRUS), dtype=xp.float32),
            "water_level_mean": xp.zeros(n_days, dtype=xp.float32),
            "temp_mean": xp.zeros(n_days, dtype=xp.float32),
            "precip_sum": xp.zeros(n_days, dtype=xp.float32),
        }
        day_iter = range(n_days)
        for offset in day_iter:
            metrics = self.step_day(
                climate_window.temperature_c[offset],
                climate_window.precipitation_mm[offset],
                climate_window.humidity_pct[offset],
                climate_window.dates[offset],
            )
            for key, value in metrics.items():
                daily[key][offset] = value
        return {key: self.backend.asnumpy(value) for key, value in daily.items()}

    def export_checkpoint(self, next_start_day: int, next_window_index: int) -> dict[str, np.ndarray]:
        st = self.state
        payload = {
            "next_start_day": np.asarray(next_start_day, dtype=np.int32),
            "next_window_index": np.asarray(next_window_index, dtype=np.int32),
            "host_population": self.backend.asnumpy(st.host_population),
            "host_s": self.backend.asnumpy(st.host_s),
            "host_e": self.backend.asnumpy(st.host_e),
            "host_i": self.backend.asnumpy(st.host_i),
            "host_severe": self.backend.asnumpy(st.host_severe),
            "host_r": self.backend.asnumpy(st.host_r),
            "host_d": self.backend.asnumpy(st.host_d),
            "mosquito_aquatic": self.backend.asnumpy(st.mosquito_aquatic),
            "mosquito_adult": self.backend.asnumpy(st.mosquito_adult),
            "mosquito_exposed": self.backend.asnumpy(st.mosquito_exposed),
            "mosquito_infectious": self.backend.asnumpy(st.mosquito_infectious),
            "water_level": self.backend.asnumpy(st.water_level),
            "cumulative_reported": self.backend.asnumpy(st.cumulative_reported),
            "climate_temperature_hist": self.backend.asnumpy(st.climate_temperature_hist),
            "climate_precip_hist": self.backend.asnumpy(st.climate_precip_hist),
            "climate_humidity_hist": self.backend.asnumpy(st.climate_humidity_hist),
            "climate_hist_cursor": np.asarray(st.climate_hist_cursor, dtype=np.int32),
            "climate_hist_count": np.asarray(st.climate_hist_count, dtype=np.int32),
        }
        return payload

    def load_checkpoint(self, payload: dict[str, np.ndarray]) -> tuple[int, int]:
        xp = self.xp
        st = self.state
        st.host_population = xp.asarray(payload["host_population"], dtype=xp.float32)
        st.host_s = xp.asarray(payload["host_s"], dtype=xp.float32)
        st.host_e = xp.asarray(payload["host_e"], dtype=xp.float32)
        st.host_i = xp.asarray(payload["host_i"], dtype=xp.float32)
        st.host_severe = xp.asarray(payload["host_severe"], dtype=xp.float32)
        st.host_r = xp.asarray(payload["host_r"], dtype=xp.float32)
        st.host_d = xp.asarray(payload["host_d"], dtype=xp.float32)
        st.mosquito_aquatic = xp.asarray(payload["mosquito_aquatic"], dtype=xp.float32)
        st.mosquito_adult = xp.asarray(payload["mosquito_adult"], dtype=xp.float32)
        st.mosquito_exposed = xp.asarray(payload["mosquito_exposed"], dtype=xp.float32)
        st.mosquito_infectious = xp.asarray(payload["mosquito_infectious"], dtype=xp.float32)
        st.water_level = xp.asarray(payload["water_level"], dtype=xp.float32)
        st.cumulative_reported = xp.asarray(payload["cumulative_reported"], dtype=xp.float32)
        st.climate_temperature_hist = xp.asarray(payload["climate_temperature_hist"], dtype=xp.float32)
        st.climate_precip_hist = xp.asarray(payload["climate_precip_hist"], dtype=xp.float32)
        st.climate_humidity_hist = xp.asarray(payload["climate_humidity_hist"], dtype=xp.float32)
        st.climate_hist_cursor = int(payload["climate_hist_cursor"])
        st.climate_hist_count = int(payload["climate_hist_count"])
        return int(payload["next_start_day"]), int(payload["next_window_index"])

    def initial_state_report(self) -> dict[str, Any]:
        return {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in self.initial_state_summary.items()
        }

    def final_summary(self) -> dict[str, Any]:
        st = self.state
        return {
            "virus_names": VIRUS_NAMES,
            "species_names": SPECIES_NAMES,
            "initial_state": self.initial_state_report(),
            "reported_cases_total": self.backend.asnumpy(st.cumulative_reported).tolist(),
            "dead_total": self.backend.asnumpy(self.xp.sum(st.host_d, axis=0)).tolist(),
            "infectious_humans_total": self.backend.asnumpy(self.xp.sum(st.host_i + st.host_severe, axis=0)).tolist(),
            "adult_vectors_total": self.backend.asnumpy(self.xp.sum(st.mosquito_adult, axis=0)).tolist(),
            "infectious_vectors_total": self.backend.asnumpy(self.xp.sum(st.mosquito_infectious, axis=(0, 1))).tolist(),
        }

    def estimate_state_bytes(self) -> int:
        n_cells = int(self.grid.population.size)
        float32_bytes = 4
        total_floats = (
            n_cells
            + n_cells * N_VIRUS * 6
            + n_cells * N_SPECIES * 2
            + n_cells * N_SPECIES * N_VIRUS * 2
            + n_cells * self.history_size * 3
            + N_VIRUS
        )
        return total_floats * float32_bytes


def write_metadata(
    output_dir: str | Path,
    cfg: SimulationConfig,
    static_grid: StaticGrid,
    backend: Backend,
    validation: dict[str, Any],
    input_audit: dict[str, Any],
    simulator: VectorizedABMSimulator,
) -> None:
    payload = {
        "config": asdict(cfg),
        "backend": {"name": backend.name, "gpu_enabled": backend.gpu_enabled},
        "grid": {
            "bbox": static_grid.bbox,
            "shape_2d": list(static_grid.shape_2d),
            "active_cells": int(static_grid.population.size),
            "total_population": float(static_grid.population.sum()),
        },
        "validation": validation,
        "input_audit": input_audit,
        "virus_names": VIRUS_NAMES,
        "species_names": SPECIES_NAMES,
        "seed_reports": DEFAULT_SEED_REPORTS,
        "initial_state": simulator.initial_state_report(),
        "estimated_state_memory_gib": round(simulator.estimate_state_bytes() / 1024**3, 3),
        "notes": {
            "vectorization": "All host, vector, and climate updates are tensorized over active cells/species/viruses.",
            "chunked_climate_io": f"Climate is loaded in windows of {cfg.climate_window_days} day(s).",
            "climate_memory_days": cfg.climate_memory_days,
            "temperature_driver": "Briere thermal response",
            "mortality_driver": "U-shaped thermal mortality with humidity stress",
            "checkpoint_interval_windows": cfg.checkpoint_interval_windows,
            "reporting_rate_scale": cfg.reporting_rate_scale,
            "external_importation": (
                "Optional observed-case schedule can inject a configurable fraction as external exposed hosts "
                "to keep pathogens from disappearing solely because initial chains burn out."
            ),
            "seasonal_seeding": "Optional once-per-year exposed-host pulses can represent seasonal introductions.",
            "reservoir_force": "Optional non-human reservoir force can be set globally for west_nile/japanese_encephalitis or per virus.",
            "disease_seasonality": {
                "preset": cfg.disease_seasonality_preset,
                "virus_seasonal_peak_month": cfg.virus_seasonal_peak_month,
                "virus_seasonal_amplitude": cfg.virus_seasonal_amplitude,
                "virus_seasonal_floor": cfg.virus_seasonal_floor,
            },
            "host_turnover": "Optional immunity waning and death replacement return hosts to susceptible compartments.",
            "spatial_diffusion": "Optional neighbor mixing smooths infectious vector pressure across adjacent active cells.",
            "transmission_coefficients": {
                "host_transmission_coeff": cfg.host_transmission_coeff,
                "vector_transmission_coeff": cfg.vector_transmission_coeff,
                "host_transmission_scale": cfg.host_transmission_scale,
                "vector_transmission_scale": cfg.vector_transmission_scale,
            },
        },
    }
    dump_json(Path(output_dir) / "metadata.json", payload)


class DistributedVectorizedABMSimulator(VectorizedABMSimulator):
    def __init__(
        self,
        cfg: SimulationConfig,
        static_grid: StaticGrid,
        backend: MultiGPUBackend,
        spatial_decomp: Any,
        rank: int,
    ):
        self.spatial_decomp = spatial_decomp
        self.rank = rank
        self.n_gpus = getattr(backend, "n_gpus", 1)
        super().__init__(cfg, static_grid, backend)

    def _build_rng(self) -> Any:
        if self.backend.gpu_enabled:
            return self.xp.random.RandomState(self.cfg.seed + self.rank)
        return np.random.default_rng(self.cfg.seed + self.rank)
