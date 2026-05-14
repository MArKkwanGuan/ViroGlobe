from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timedelta
import posixpath
from pathlib import Path
import re
import sys
from typing import Any
import zipfile
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = REPO_ROOT / "data" / "virs"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "historical_validation.csv"

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
XML_NS = {"main": MAIN_NS}

SUPPORTED_VIRUSES = {
    "dengue",
    "zika",
    "yellow_fever",
    "west_nile",
    "japanese_encephalitis",
}


def _normalize_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _parse_number(value: Any) -> float:
    if value is None:
        return 0.0
    text = str(value).strip().replace(",", "")
    if text in {"", "---", "*", "nan", "None"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _parse_year(value: Any) -> int | None:
    text = str(value).strip()
    if not text or text in {"---", "*"}:
        return None
    match = re.search(r"(?:19|20)\d{2}", text)
    if match:
        return int(match.group(0))
    try:
        number = float(text)
    except ValueError:
        return None
    if 1800 <= number <= 2200 and number.is_integer():
        return int(number)
    if number > 20000:
        try:
            return (datetime(1899, 12, 30) + timedelta(days=number)).year
        except OverflowError:
            return None
    return None


def _within_year_range(year: int, min_year: int | None, max_year: int | None) -> bool:
    if min_year is not None and year < min_year:
        return False
    if max_year is not None and year > max_year:
        return False
    return True


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    values: list[str] = []
    for item in root.findall("main:si", XML_NS):
        values.append("".join(node.text or "" for node in item.findall(".//main:t", XML_NS)))
    return values


def _xlsx_sheet_paths(archive: zipfile.ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_targets = {rel.attrib["Id"]: rel.attrib["Target"] for rel in relationships}
    paths: dict[str, str] = {}
    sheets = workbook.find("main:sheets", XML_NS)
    if sheets is None:
        return paths
    for sheet in sheets:
        name = sheet.attrib["name"]
        relationship_id = sheet.attrib[f"{{{REL_NS}}}id"]
        target = rel_targets[relationship_id]
        if target.startswith("/"):
            path = target.lstrip("/")
        else:
            path = posixpath.normpath(posixpath.join("xl", target))
        paths[name] = path
    return paths


def _xlsx_col_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for ch in letters:
        index = index * 26 + ord(ch.upper()) - ord("A") + 1
    return max(index - 1, 0)


def _xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value = cell.find("main:v", XML_NS)
    if cell_type == "s" and value is not None:
        return shared_strings[int(value.text or "0")]
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//main:t", XML_NS))
    return "" if value is None else str(value.text or "")


def _xlsx_rows(path: Path, sheet_name: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings = _xlsx_shared_strings(archive)
        sheet_paths = _xlsx_sheet_paths(archive)
        if sheet_name not in sheet_paths:
            raise ValueError(f"{path} does not contain sheet {sheet_name!r}")
        root = ET.fromstring(archive.read(sheet_paths[sheet_name]))

        rows: list[list[str]] = []
        for row in root.findall(".//main:sheetData/main:row", XML_NS):
            values: list[str] = []
            for cell in row.findall("main:c", XML_NS):
                index = _xlsx_col_index(cell.attrib.get("r", "A1"))
                while len(values) < index:
                    values.append("")
                values.append(_xlsx_cell_value(cell, shared_strings))
            if any(str(value).strip() for value in values):
                rows.append(values)

    if not rows:
        return []
    header = [_normalize_key(value) for value in rows[0]]
    records: list[dict[str, str]] = []
    for values in rows[1:]:
        record = {}
        for index, key in enumerate(header):
            if key:
                record[key] = values[index] if index < len(values) else ""
        records.append(record)
    return records


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return [{_normalize_key(key): value for key, value in row.items()} for row in reader]


def _should_include_region(row: dict[str, str], target_region: str | None) -> bool:
    if not target_region or target_region.lower() == "global":
        return True
    
    region_map = {
        "africa": ["africa", "afr"],
        "americas": ["americas", "amr", "paho"],
        "europe": ["europe", "eur"],
        "southeast_asia": ["south-east asia", "sear"],
        "western_pacific": ["western pacific", "wpr"],
        "eastern_mediterranean": ["eastern mediterranean", "emr"],
    }
    
    target_tags = region_map.get(target_region.lower(), [target_region.lower()])
    
    # Check common regional columns across different file formats
    potential_keys = ["parentlocation", "parentlocationcode", "who_region", "who_region_long", "region"]
    for key in potential_keys:
        val = str(row.get(key, "")).strip().lower()
        if val in target_tags:
            return True
            
    # Special case for ECDC (West Nile) - it's all Europe
    if "regioncode" in row and "regionname" in row: # ECDC format
        return target_region.lower() == "europe"
        
    # Special case for PAHO (Zika) - it's all Americas
    if "country_or_subregion" in row and "in_/_out_of_subregions" in row:
        return target_region.lower() == "americas"

    return False


def _add_total(
    totals: dict[tuple[str, int], dict[str, Any]],
    virus: str,
    year: int | None,
    value: Any,
    source: Path,
    min_year: int | None,
    max_year: int | None,
    row: dict[str, str],
    target_region: str | None,
) -> None:
    if virus not in SUPPORTED_VIRUSES or year is None:
        return
    if not _within_year_range(year, min_year, max_year):
        return
    if not _should_include_region(row, target_region):
        return
        
    key = (virus, year)
    if key not in totals:
        totals[key] = {
            "virus": virus,
            "year": year,
            "observed_cases": 0.0,
            "source": source.name,
        }
    totals[key]["observed_cases"] += _parse_number(value)


def _convert_dengue(
    source_dir: Path,
    totals: dict[tuple[str, int], dict[str, Any]],
    min_year: int | None,
    max_year: int | None,
    target_region: str | None,
) -> int:
    converted = 0
    for path in source_dir.rglob("dengue-global-data-*.xlsx"):
        for row in _xlsx_rows(path, "data"):
            year = _parse_year(row.get("date_lab") or row.get("date"))
            _add_total(totals, "dengue", year, row.get("cases"), path, min_year, max_year, row, target_region)
            converted += 1
    return converted


def _convert_zika(
    source_dir: Path,
    totals: dict[tuple[str, int], dict[str, Any]],
    min_year: int | None,
    max_year: int | None,
    zika_measure: str,
    target_region: str | None,
) -> int:
    measure_name = "Confirmed" if zika_measure == "confirmed" else "Total Cases (b)"
    converted = 0
    for path in source_dir.rglob("W_By_Last_Available_EpiWeek_data.csv"):
        for row in _csv_rows(path):
            if row.get("measure_names") != measure_name:
                continue
            year = _parse_year(row.get("year"))
            _add_total(totals, "zika", year, row.get("measure_values"), path, min_year, max_year, row, target_region)
            converted += 1
    return converted


def _convert_who_csv(
    source_dir: Path,
    file_name: str,
    virus: str,
    totals: dict[tuple[str, int], dict[str, Any]],
    min_year: int | None,
    max_year: int | None,
    target_region: str | None,
) -> int:
    converted = 0
    for path in source_dir.rglob(file_name):
        for row in _csv_rows(path):
            if row.get("period_type") and row.get("period_type") != "Year":
                continue
            if row.get("location_type") and row.get("location_type") != "Country":
                continue
            year = _parse_year(row.get("period"))
            value = row.get("factvaluenumeric") or row.get("value")
            _add_total(totals, virus, year, value, path, min_year, max_year, row, target_region)
            converted += 1
    return converted


def _convert_japanese_encephalitis_xlsx(
    source_dir: Path,
    totals: dict[tuple[str, int], dict[str, Any]],
    min_year: int | None,
    max_year: int | None,
    target_region: str | None,
) -> int:
    converted = 0
    # Note: Japanese Encephalitis XLSX currently only has a reliable 'Global' row for totals.
    # If a region is requested, we skip this file unless we implement a country-to-region mapping.
    if target_region and target_region.lower() != "global":
        return 0
        
    for path in source_dir.rglob("Japanese Encephalitis (JE) reported cases*.xlsx"):
        for row in _xlsx_rows(path, "Sheet1"):
            country = str(row.get("country_/_region", "")).strip().lower()
            if country != "global":
                continue
            for key, value in row.items():
                year = _parse_year(key)
                if year is None:
                    continue
                _add_total(totals, "japanese_encephalitis", year, value, path, min_year, max_year, row, target_region)
                converted += 1
    return converted


def _convert_west_nile_ecdc(
    source_dir: Path,
    totals: dict[tuple[str, int], dict[str, Any]],
    min_year: int | None,
    max_year: int | None,
    target_region: str | None,
) -> int:
    converted = 0
    for path in source_dir.rglob("ECDC_surveillance_data_West_Nile_virus_infection.csv"):
        for row in _csv_rows(path):
            if row.get("population") != "All cases" or row.get("indicator") != "Reported cases":
                continue
            year = _parse_year(row.get("time"))
            _add_total(totals, "west_nile", year, row.get("numvalue"), path, min_year, max_year, row, target_region)
            converted += 1
    return converted


def build_validation_targets(
    source_dir: Path,
    min_year: int | None = None,
    max_year: int | None = None,
    zika_measure: str = "total",
    target_region: str | None = "global",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    totals: dict[tuple[str, int], dict[str, Any]] = {}
    japanese_encephalitis_rows = _convert_who_csv(
        source_dir,
        "Japanese Encephalitis (JE).csv",
        "japanese_encephalitis",
        totals,
        min_year,
        max_year,
        target_region,
    )
    counts = {
        "dengue_rows": _convert_dengue(source_dir, totals, min_year, max_year, target_region),
        "zika_rows": _convert_zika(source_dir, totals, min_year, max_year, zika_measure, target_region),
        "yellow_fever_rows": _convert_who_csv(source_dir, "yellowfever.csv", "yellow_fever", totals, min_year, max_year, target_region),
        "japanese_encephalitis_rows": japanese_encephalitis_rows,
        "japanese_encephalitis_xlsx_rows": 0
        if japanese_encephalitis_rows
        else _convert_japanese_encephalitis_xlsx(source_dir, totals, min_year, max_year, target_region),
        "west_nile_rows": _convert_west_nile_ecdc(source_dir, totals, min_year, max_year, target_region),
    }

    rows = [totals[key] for key in sorted(totals, key=lambda item: (item[0], item[1]))]
    return rows, counts


def write_validation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["virus", "year", "observed_cases", "source"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            value = float(row["observed_cases"])
            out = dict(row)
            out["observed_cases"] = str(int(value)) if value.is_integer() else f"{value:.6g}"
            writer.writerow(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert bundled disease history files into validation targets.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR), help="Input root such as data/virs or data/virs_new")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    parser.add_argument("--min-year", type=int, default=None)
    parser.add_argument("--max-year", type=int, default=None)
    parser.add_argument(
        "--zika-measure",
        choices=("total", "confirmed"),
        default="total",
        help="Use PAHO Total Cases (b) or Confirmed for zika",
    )
    parser.add_argument(
        "--region",
        choices=("global", "africa", "americas", "europe", "southeast_asia", "western_pacific", "eastern_mediterranean"),
        default="global",
        help="Filter data by region/continent",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_dir = Path(args.source_dir)
    if not source_dir.is_absolute():
        source_dir = (REPO_ROOT / source_dir).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = (REPO_ROOT / output).resolve()

    rows, counts = build_validation_targets(
        source_dir=source_dir,
        min_year=args.min_year,
        max_year=args.max_year,
        zika_measure=args.zika_measure,
        target_region=args.region,
    )
    if not rows:
        raise RuntimeError(f"No validation rows were produced from {source_dir} for region {args.region}")
    write_validation_csv(output, rows)

    print(f"Wrote {len(rows)} validation rows to {output} (Region: {args.region})")
    for key, value in counts.items():
        print(f"{key}: {value}", file=sys.stderr)


if __name__ == "__main__":
    main()
