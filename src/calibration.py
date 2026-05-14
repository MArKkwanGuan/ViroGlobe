from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .constants import N_VIRUS, VIRUS_NAMES


TARGET_VALUE_KEYS = (
    "target_reported",
    "target_cases",
    "reported_cases_total",
    "reported_cases",
    "reported",
    "cases",
    "observed",
    "target",
)
VIRUS_NAME_KEYS = ("virus", "virus_name", "disease", "pathogen", "name")
VIRUS_INDEX_KEYS = ("virus_index", "index", "id")


@dataclass(frozen=True)
class CalibrationTarget:
    virus_index: int
    virus_name: str
    target_reported: float


def _normalize_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_virus_name(value: str) -> str:
    normalized = _normalize_key(value)
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


def _virus_index_from_value(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        index = int(value)
        if 0 <= index < N_VIRUS:
            return index
        raise ValueError(f"virus index must be in [0, {N_VIRUS - 1}], got {index}")

    text = str(value).strip()
    if text == "":
        raise ValueError("virus value is empty")
    if text.isdigit():
        return _virus_index_from_value(int(text))

    normalized = _normalize_virus_name(text)
    for index, name in enumerate(VIRUS_NAMES):
        if normalized == _normalize_virus_name(name):
            return index
    raise ValueError(f"Unknown virus name: {value!r}. Expected one of: {', '.join(VIRUS_NAMES)}")


def _coerce_target_value(value: Any) -> float:
    try:
        target = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Calibration target must be numeric, got {value!r}") from exc
    if not math.isfinite(target) or target < 0.0:
        raise ValueError(f"Calibration target must be finite and >= 0, got {target!r}")
    return target


def _normalise_record(record: dict[str, Any]) -> dict[str, Any]:
    return {_normalize_key(key): value for key, value in record.items()}


def _record_to_target(record: dict[str, Any]) -> CalibrationTarget:
    normalized = _normalise_record(record)

    virus_index: int | None = None
    for key in VIRUS_INDEX_KEYS:
        if key in normalized and str(normalized[key]).strip() != "":
            virus_index = _virus_index_from_value(normalized[key])
            break
    if virus_index is None:
        for key in VIRUS_NAME_KEYS:
            if key in normalized and str(normalized[key]).strip() != "":
                virus_index = _virus_index_from_value(normalized[key])
                break
    if virus_index is None:
        raise ValueError(f"Calibration target row is missing a virus column: {record!r}")

    target_value: float | None = None
    for key in TARGET_VALUE_KEYS:
        if key in normalized and str(normalized[key]).strip() != "":
            target_value = _coerce_target_value(normalized[key])
            break
    if target_value is None:
        raise ValueError(f"Calibration target row is missing a target value column: {record!r}")

    return CalibrationTarget(
        virus_index=virus_index,
        virus_name=VIRUS_NAMES[virus_index],
        target_reported=target_value,
    )


def _rows_from_json(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        if all(not isinstance(item, dict) for item in raw):
            if len(raw) != N_VIRUS:
                raise ValueError(f"JSON calibration target list must contain {N_VIRUS} values.")
            return [
                {"virus": VIRUS_NAMES[index], "target_reported": value}
                for index, value in enumerate(raw)
            ]
        return [dict(item) for item in raw]

    if not isinstance(raw, dict):
        raise ValueError("JSON calibration targets must be an object or list.")

    if "targets" in raw:
        return _rows_from_json(raw["targets"])

    if "reported_cases_total" in raw and not any(key in raw for key in VIRUS_NAME_KEYS):
        value = raw["reported_cases_total"]
        if isinstance(value, dict):
            return [{"virus": virus, "target_reported": target} for virus, target in value.items()]
        return _rows_from_json(value)

    if any(_normalize_key(key) in VIRUS_NAME_KEYS for key in raw):
        return [dict(raw)]

    rows = []
    for virus, target in raw.items():
        rows.append({"virus": virus, "target_reported": target})
    return rows


def _load_json_targets(path: Path) -> list[CalibrationTarget]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [_record_to_target(row) for row in _rows_from_json(raw)]


def _load_csv_targets(path: Path) -> list[CalibrationTarget]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV calibration targets require a header row.")
        targets = []
        for row in reader:
            if not any(value is not None and str(value).strip() != "" for value in row.values()):
                continue
            targets.append(_record_to_target(row))
        return targets


def load_calibration_targets(path: str | Path) -> list[CalibrationTarget]:
    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Calibration target file does not exist: {target_path}")

    if target_path.suffix.lower() == ".json":
        targets = _load_json_targets(target_path)
    else:
        targets = _load_csv_targets(target_path)

    if not targets:
        raise ValueError(f"Calibration target file is empty: {target_path}")

    aggregate = np.zeros(N_VIRUS, dtype=np.float64)
    active = np.zeros(N_VIRUS, dtype=bool)
    for target in targets:
        aggregate[target.virus_index] += target.target_reported
        active[target.virus_index] = True

    return [
        CalibrationTarget(index, VIRUS_NAMES[index], float(aggregate[index]))
        for index in range(N_VIRUS)
        if active[index]
    ]


def _summary_reported_vector(summary: dict[str, Any]) -> np.ndarray:
    values = np.asarray(summary.get("reported_cases_total"), dtype=np.float64)
    if values.shape != (N_VIRUS,):
        raise ValueError(f"summary.reported_cases_total must contain {N_VIRUS} values.")
    if not np.all(np.isfinite(values)):
        raise ValueError("summary.reported_cases_total contains non-finite values.")
    return np.maximum(values, 0.0)


def build_calibration_result(
    summary: dict[str, Any],
    targets: list[CalibrationTarget],
    min_scale: float = 0.0,
    max_scale: float = 100.0,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    if min_scale < 0.0 or not math.isfinite(min_scale):
        raise ValueError("calibration min scale must be finite and >= 0")
    if max_scale < min_scale or not math.isfinite(max_scale):
        raise ValueError("calibration max scale must be finite and >= min scale")

    predicted = _summary_reported_vector(summary)
    target_values = np.full(N_VIRUS, np.nan, dtype=np.float64)
    scales = np.ones(N_VIRUS, dtype=np.float64)
    rows: list[dict[str, Any]] = []

    for target in targets:
        index = int(target.virus_index)
        observed = float(target.target_reported)
        predicted_value = float(predicted[index])
        target_values[index] = observed

        if predicted_value <= epsilon:
            raw_scale = 1.0 if observed <= epsilon else max_scale
            status = "target_and_prediction_zero" if observed <= epsilon else "prediction_zero_clamped"
        else:
            raw_scale = observed / predicted_value
            status = "ok"

        scale = float(np.clip(raw_scale, min_scale, max_scale))
        if status == "ok" and scale != raw_scale:
            status = "clamped"

        scales[index] = scale
        calibrated = predicted_value * scale
        rows.append(
            {
                "virus": VIRUS_NAMES[index],
                "target_reported": observed,
                "predicted_reported": predicted_value,
                "scale": scale,
                "calibrated_reported": calibrated,
                "residual_before": predicted_value - observed,
                "residual_after": calibrated - observed,
                "status": status,
            }
        )

    calibrated_values = predicted * scales
    active_mask = np.isfinite(target_values)
    before_abs = float(np.sum(np.abs(predicted[active_mask] - target_values[active_mask])))
    after_abs = float(np.sum(np.abs(calibrated_values[active_mask] - target_values[active_mask])))

    return {
        "method": "multiplicative_reported_cases",
        "virus_names": list(VIRUS_NAMES),
        "reporting_rate_scale": [float(value) for value in scales],
        "min_scale": float(min_scale),
        "max_scale": float(max_scale),
        "predicted_reported": {
            name: float(predicted[index])
            for index, name in enumerate(VIRUS_NAMES)
        },
        "calibrated_reported": {
            name: float(calibrated_values[index])
            for index, name in enumerate(VIRUS_NAMES)
        },
        "targets": {
            VIRUS_NAMES[index]: float(target_values[index])
            for index in range(N_VIRUS)
            if active_mask[index]
        },
        "per_virus": rows,
        "metrics": {
            "target_total": float(np.sum(target_values[active_mask])) if np.any(active_mask) else 0.0,
            "predicted_total": float(np.sum(predicted[active_mask])) if np.any(active_mask) else 0.0,
            "calibrated_total": float(np.sum(calibrated_values[active_mask])) if np.any(active_mask) else 0.0,
            "absolute_error_before": before_abs,
            "absolute_error_after": after_abs,
        },
        "notes": [
            "Scales are multiplicative factors for reported cases by virus.",
            "Use --reporting-rate-scale with these values to apply the same reporting-rate scale in a future run.",
        ],
    }


def build_calibrated_summary(summary: dict[str, Any], calibration_result: dict[str, Any]) -> dict[str, Any]:
    calibrated = dict(summary)
    scales = np.asarray(calibration_result["reporting_rate_scale"], dtype=np.float64)
    reported = _summary_reported_vector(summary)
    calibrated["reported_cases_total_uncalibrated"] = [float(value) for value in reported]
    calibrated["reported_cases_total_calibrated"] = [float(value) for value in reported * scales]
    calibrated["calibration"] = {
        "method": calibration_result["method"],
        "reporting_rate_scale": calibration_result["reporting_rate_scale"],
        "targets": calibration_result["targets"],
        "metrics": calibration_result["metrics"],
    }
    return calibrated


def apply_calibration_to_windows(
    output_dir: str | Path,
    manifest: dict[str, Any],
    reporting_rate_scale: list[float],
) -> list[str]:
    output_path = Path(output_dir)
    scales = np.asarray(reporting_rate_scale, dtype=np.float32)
    if scales.shape != (N_VIRUS,):
        raise ValueError(f"reporting_rate_scale must contain {N_VIRUS} values.")

    updated_files: list[str] = []
    for window in manifest.get("window_files", []):
        file_name = window["file"] if isinstance(window, dict) else str(window)
        chunk_path = output_path / "chunks" / file_name
        with np.load(chunk_path, allow_pickle=False) as payload:
            data = {key: payload[key] for key in payload.files}
        if "reported" not in data:
            continue
        data["reported_calibrated"] = np.asarray(data["reported"], dtype=np.float32) * scales[None, :]
        np.savez(chunk_path, **data)
        updated_files.append(file_name)
    return updated_files
