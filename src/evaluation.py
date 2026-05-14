from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .constants import N_VIRUS, VIRUS_NAMES


OBSERVED_VALUE_KEYS = (
    "observed_cases",
    "observed_reported",
    "reported_cases",
    "reported",
    "cases",
    "actual",
    "value",
    "target",
)
VIRUS_NAME_KEYS = ("virus", "virus_name", "disease", "pathogen", "name")
VIRUS_INDEX_KEYS = ("virus_index", "index", "id")
SUPPORTED_FREQUENCIES = ("year", "month", "day")


@dataclass(frozen=True)
class ValidationObservation:
    virus_index: int
    virus_name: str
    period: str
    observed_cases: float


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


def _coerce_observed_value(value: Any) -> float:
    try:
        observed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Observed cases must be numeric, got {value!r}") from exc
    if not math.isfinite(observed) or observed < 0.0:
        raise ValueError(f"Observed cases must be finite and >= 0, got {observed!r}")
    return observed


def _normalise_record(record: dict[str, Any]) -> dict[str, Any]:
    return {_normalize_key(key): value for key, value in record.items()}


def _parse_date(value: Any) -> datetime:
    text = str(value).strip()
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y/%m", "%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"Could not parse date or period value: {value!r}")


def _period_from_date(value: Any, frequency: str) -> str:
    dt = _parse_date(value)
    if frequency == "year":
        return dt.strftime("%Y")
    if frequency == "month":
        return dt.strftime("%Y-%m")
    if frequency == "day":
        return dt.strftime("%Y-%m-%d")
    raise ValueError(f"Unsupported validation frequency: {frequency}")


def _period_from_record(record: dict[str, Any], frequency: str) -> str:
    normalized = _normalise_record(record)
    if "period" in normalized and str(normalized["period"]).strip() != "":
        return _period_from_date(normalized["period"], frequency)
    if "date" in normalized and str(normalized["date"]).strip() != "":
        return _period_from_date(normalized["date"], frequency)
    if "period_start" in normalized and str(normalized["period_start"]).strip() != "":
        return _period_from_date(normalized["period_start"], frequency)

    year_value = normalized.get("year")
    month_value = normalized.get("month")
    day_value = normalized.get("day")
    if year_value is None or str(year_value).strip() == "":
        raise ValueError(f"Validation row is missing year/date/period: {record!r}")

    year = int(float(str(year_value).strip()))
    if frequency == "year":
        return f"{year:04d}"

    if month_value is None or str(month_value).strip() == "":
        raise ValueError(f"Validation row needs a month column for monthly/daily validation: {record!r}")
    month = int(float(str(month_value).strip()))
    if not 1 <= month <= 12:
        raise ValueError(f"month must be in [1, 12], got {month}")
    if frequency == "month":
        return f"{year:04d}-{month:02d}"

    if day_value is None or str(day_value).strip() == "":
        raise ValueError(f"Validation row needs a day column for daily validation: {record!r}")
    day = int(float(str(day_value).strip()))
    return datetime(year, month, day).strftime("%Y-%m-%d")


def _record_to_observation(record: dict[str, Any], frequency: str) -> ValidationObservation:
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
        raise ValueError(f"Validation row is missing a virus column: {record!r}")

    observed_cases: float | None = None
    for key in OBSERVED_VALUE_KEYS:
        if key in normalized and str(normalized[key]).strip() != "":
            observed_cases = _coerce_observed_value(normalized[key])
            break
    if observed_cases is None:
        raise ValueError(f"Validation row is missing an observed cases column: {record!r}")

    return ValidationObservation(
        virus_index=virus_index,
        virus_name=VIRUS_NAMES[virus_index],
        period=_period_from_record(record, frequency),
        observed_cases=observed_cases,
    )


def _rows_from_json(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [dict(item) for item in raw]
    if not isinstance(raw, dict):
        raise ValueError("JSON validation targets must be an object or list.")
    if "observations" in raw:
        return _rows_from_json(raw["observations"])
    if "targets" in raw:
        return _rows_from_json(raw["targets"])

    rows: list[dict[str, Any]] = []
    for virus, periods in raw.items():
        if not isinstance(periods, dict):
            raise ValueError("JSON validation mapping must be {virus: {period: observed_cases}}.")
        for period, observed in periods.items():
            rows.append({"virus": virus, "period": period, "observed_cases": observed})
    return rows


def _load_json_observations(path: Path, frequency: str) -> list[ValidationObservation]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [_record_to_observation(row, frequency) for row in _rows_from_json(raw)]


def _load_csv_observations(path: Path, frequency: str) -> list[ValidationObservation]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV validation targets require a header row.")
        observations = []
        for row in reader:
            if not any(value is not None and str(value).strip() != "" for value in row.values()):
                continue
            observations.append(_record_to_observation(row, frequency))
        return observations


def load_validation_observations(path: str | Path, frequency: str = "year") -> list[ValidationObservation]:
    if frequency not in SUPPORTED_FREQUENCIES:
        raise ValueError(f"validation frequency must be one of: {', '.join(SUPPORTED_FREQUENCIES)}")

    target_path = Path(path)
    if not target_path.exists():
        raise FileNotFoundError(f"Validation target file does not exist: {target_path}")

    if target_path.suffix.lower() == ".json":
        observations = _load_json_observations(target_path, frequency)
    else:
        observations = _load_csv_observations(target_path, frequency)
    if not observations:
        raise ValueError(f"Validation target file is empty: {target_path}")

    aggregate: dict[tuple[int, str], float] = {}
    for obs in observations:
        key = (obs.virus_index, obs.period)
        aggregate[key] = aggregate.get(key, 0.0) + obs.observed_cases

    return [
        ValidationObservation(index, VIRUS_NAMES[index], period, observed)
        for (index, period), observed in sorted(aggregate.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


def _manifest_window_files(output_dir: Path, manifest: dict[str, Any]) -> list[Path]:
    files: list[Path] = []
    for entry in manifest.get("window_files", []):
        file_name = entry["file"] if isinstance(entry, dict) else str(entry)
        files.append(output_dir / "chunks" / file_name)
    if not files:
        raise ValueError(f"Manifest contains no window files: {output_dir / 'manifest.json'}")
    return files


def aggregate_window_predictions(
    output_dir: str | Path,
    manifest: dict[str, Any],
    prediction_key: str = "reported",
    frequency: str = "year",
) -> tuple[dict[tuple[int, str], float], set[str]]:
    if frequency not in SUPPORTED_FREQUENCIES:
        raise ValueError(f"validation frequency must be one of: {', '.join(SUPPORTED_FREQUENCIES)}")

    output_path = Path(output_dir)
    predictions: dict[tuple[int, str], float] = {}
    periods: set[str] = set()
    for chunk_path in _manifest_window_files(output_path, manifest):
        if not chunk_path.exists():
            raise FileNotFoundError(f"Window output is missing: {chunk_path}")
        with np.load(chunk_path, allow_pickle=False) as payload:
            if prediction_key not in payload.files:
                raise KeyError(f"{chunk_path.name} does not contain prediction key {prediction_key!r}")
            dates = np.asarray(payload["dates"]).tolist()
            predicted = np.asarray(payload[prediction_key], dtype=np.float64)

        if predicted.ndim != 2 or predicted.shape[1] != N_VIRUS:
            raise ValueError(f"{chunk_path.name}:{prediction_key} must have shape (days, {N_VIRUS})")
        if len(dates) != predicted.shape[0]:
            raise ValueError(f"{chunk_path.name}: dates length does not match {prediction_key} rows")

        for day_index, raw_date in enumerate(dates):
            period = _period_from_date(raw_date, frequency)
            periods.add(period)
            for virus_index in range(N_VIRUS):
                key = (virus_index, period)
                predictions[key] = predictions.get(key, 0.0) + float(predicted[day_index, virus_index])
    return predictions, periods


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    return numerator / denominator


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "observed_total": 0.0,
            "predicted_total": 0.0,
            "error_total": 0.0,
            "bias": None,
            "mae": None,
            "rmse": None,
            "mape": None,
            "smape": None,
            "pearson_r": None,
        }

    observed = np.asarray([row["observed_cases"] for row in rows], dtype=np.float64)
    predicted = np.asarray([row["predicted_cases"] for row in rows], dtype=np.float64)
    errors = predicted - observed
    abs_errors = np.abs(errors)
    n = int(len(rows))

    nonzero_observed = observed > 0.0
    mape = None
    if np.any(nonzero_observed):
        mape = float(np.mean(abs_errors[nonzero_observed] / observed[nonzero_observed]) * 100.0)

    smape_denom = np.abs(observed) + np.abs(predicted)
    smape_mask = smape_denom > 0.0
    smape = None
    if np.any(smape_mask):
        smape = float(np.mean(2.0 * abs_errors[smape_mask] / smape_denom[smape_mask]) * 100.0)

    pearson = None
    if n >= 2 and float(np.std(observed)) > 0.0 and float(np.std(predicted)) > 0.0:
        pearson = float(np.corrcoef(observed, predicted)[0, 1])

    return {
        "n": n,
        "observed_total": float(np.sum(observed)),
        "predicted_total": float(np.sum(predicted)),
        "error_total": float(np.sum(errors)),
        "bias": float(np.mean(errors)),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "mape": mape,
        "smape": smape,
        "pearson_r": pearson,
    }


def build_evaluation_result(
    observations: list[ValidationObservation],
    predictions: dict[tuple[int, str], float],
    predicted_periods: set[str],
    prediction_key: str,
    frequency: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for obs in observations:
        predicted = float(predictions.get((obs.virus_index, obs.period), 0.0))
        error = predicted - obs.observed_cases
        percent_error = _safe_divide(error * 100.0, obs.observed_cases)
        rows.append(
            {
                "virus": obs.virus_name,
                "period": obs.period,
                "observed_cases": float(obs.observed_cases),
                "predicted_cases": predicted,
                "error": float(error),
                "absolute_error": float(abs(error)),
                "percent_error": percent_error,
                "status": "ok" if obs.period in predicted_periods else "no_prediction_period",
            }
        )

    by_virus = {}
    for virus_name in VIRUS_NAMES:
        virus_rows = [row for row in rows if row["virus"] == virus_name]
        if virus_rows:
            by_virus[virus_name] = _metric_summary(virus_rows)

    return {
        "method": "historical_backtest",
        "frequency": frequency,
        "prediction_key": prediction_key,
        "virus_names": list(VIRUS_NAMES),
        "rows": rows,
        "summary": {
            "overall": _metric_summary(rows),
            "by_virus": by_virus,
            "status_counts": {
                status: sum(1 for row in rows if row["status"] == status)
                for status in sorted({row["status"] for row in rows})
            },
        },
        "notes": [
            "error = predicted_cases - observed_cases",
            "mape ignores rows with observed_cases equal to zero",
            "status no_prediction_period means the observation period is outside the simulated dates",
        ],
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_evaluation_outputs(
    output_dir: str | Path,
    evaluation: dict[str, Any],
    output_prefix: str = "validation",
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_name = f"{output_prefix}.csv"
    summary_name = f"{output_prefix}_summary.json"
    csv_path = output_path / csv_name
    summary_path = output_path / summary_name

    fieldnames = [
        "virus",
        "period",
        "observed_cases",
        "predicted_cases",
        "error",
        "absolute_error",
        "percent_error",
        "status",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in evaluation["rows"]:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})

    summary_payload = {
        key: value
        for key, value in evaluation.items()
        if key != "rows"
    }
    summary_path.write_text(json.dumps(_json_ready(summary_payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return {"csv": csv_name, "summary": summary_name}


def evaluate_output(
    output_dir: str | Path,
    manifest: dict[str, Any],
    observations: list[ValidationObservation],
    prediction_key: str = "reported",
    frequency: str = "year",
) -> dict[str, Any]:
    predictions, predicted_periods = aggregate_window_predictions(
        output_dir=output_dir,
        manifest=manifest,
        prediction_key=prediction_key,
        frequency=frequency,
    )
    return build_evaluation_result(
        observations=observations,
        predictions=predictions,
        predicted_periods=predicted_periods,
        prediction_key=prediction_key,
        frequency=frequency,
    )
