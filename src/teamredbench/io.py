from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from teamredbench.benchmarks.base import BenchmarkRecord


def _serialize_record(record: BenchmarkRecord) -> dict[str, object]:
    return {
        "benchmark": record.benchmark,
        "case": record.case,
        "dtype": record.dtype,
        "status": record.status,
        "error": record.error,
        "metrics": record.metrics,
        "raw_metrics": record.raw_metrics,
        "metadata": record.metadata,
    }


def _flatten_record(record: BenchmarkRecord) -> dict[str, object]:
    row: dict[str, object] = {
        "benchmark": record.benchmark,
        "case": record.case,
        "dtype": record.dtype,
        "status": record.status,
        "error": record.error or "",
    }
    for key, value in sorted(record.metrics.items()):
        row[f"metric_{key}"] = value
    for key, value in sorted(record.raw_metrics.items()):
        row[f"raw_{key}"] = value
    for key, value in sorted(record.metadata.items()):
        row[f"meta_{key}"] = value
    return row


def write_results(
    records: Iterable[BenchmarkRecord],
    suite_name: str,
    output_dir: Path,
    formats: list[str],
    run_metadata: dict[str, object] | None = None,
) -> dict[str, Path]:
    records_list = list(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outputs: dict[str, Path] = {}

    if "json" in formats:
        json_path = output_dir / f"{suite_name}-{timestamp}.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump([_serialize_record(record) for record in records_list], handle, indent=2)
        outputs["json"] = json_path

    if "csv" in formats:
        csv_path = output_dir / f"{suite_name}-{timestamp}.csv"
        flat_rows = [_flatten_record(record) for record in records_list]
        fieldnames = sorted({key for row in flat_rows for key in row})
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_rows)
        outputs["csv"] = csv_path

    metadata_path = output_dir / f"{suite_name}-{timestamp}.metadata.json"
    outputs["metadata"] = metadata_path
    status_counts: dict[str, int] = {}
    for record in records_list:
        status_counts[record.status] = status_counts.get(record.status, 0) + 1
    metadata_payload = {
        "schema_version": 1,
        "suite_name": suite_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "record_count": len(records_list),
        "record_status_counts": status_counts,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "run": run_metadata or {},
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_payload, handle, indent=2)

    return outputs
