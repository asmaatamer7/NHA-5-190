"""Run the unchanged per-image LAB CLI in verified, resumable parallel chunks."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from extract_training_features import digest, write_json


def extract(manifest, config, output, repo_root, *, workers=6, chunk_size=2048, resume=False):
    manifest, config, output, repo_root = map(Path, (manifest, config, output, repo_root))
    if workers < 1 or chunk_size < 1:
        raise ValueError("Workers and chunk size must be positive")
    if output.exists() and (not resume or not (output / "run.json").is_file()):
        raise ValueError("Existing output requires --resume and its original run.json")
    inputs = [
        manifest,
        config,
        repo_root / "scripts/extract_color_palettes.py",
        repo_root / "src/wardiq/clothing/color_extraction.py",
    ]
    if any(
        output.resolve() == p.resolve() or output.resolve() in p.resolve().parents for p in inputs
    ):
        raise ValueError("Color output overlaps an input")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({r["item_id"] for r in rows}) != len(rows):
        raise ValueError("Expected nonempty, unique training items")
    if any(r.get("target_split") != "train" for r in rows):
        raise ValueError("Color extraction accepts training items only")
    signature = {
        "manifest_sha256": digest(manifest),
        "config_sha256": digest(config),
        "extractor_runner_sha256": digest(inputs[2]),
        "implementation_sha256": digest(inputs[3]),
        "item_count": len(rows),
        "chunk_size": chunk_size,
    }
    if output.exists() and json.loads((output / "run.json").read_text()) != signature:
        raise ValueError("Resume inputs/configuration differ from the original run")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run.json", signature)
    jobs = []
    for start in range(0, len(rows), chunk_size):
        directory = output / "chunks" / f"{start:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        chunk_manifest = directory / "manifest.csv"
        subset = rows[start : start + chunk_size]
        with chunk_manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(subset)
        jobs.append((directory, chunk_manifest, subset))

    def run_chunk(job):
        directory, chunk_manifest, subset = job
        receipt = directory / "verified.json"
        files = ["palettes.jsonl", "coverage.json"]
        result_dir = directory / "results"
        if receipt.is_file():
            saved = json.loads(receipt.read_text())
            if saved["manifest_sha256"] != digest(chunk_manifest) or any(
                digest(result_dir / name) != checksum for name, checksum in saved["outputs"].items()
            ):
                raise ValueError(f"Changed completed color chunk: {directory.name}")
        else:
            with (directory / "extraction.log").open("w", encoding="utf-8") as log:
                command = [
                    sys.executable,
                    str(inputs[2].resolve()),
                    "--manifest",
                    str(chunk_manifest.resolve()),
                    "--config",
                    str(config.resolve()),
                    "--output-dir",
                    str(result_dir.resolve()),
                    "--image-root",
                    str(repo_root.resolve()),
                ]
                proc = subprocess.run(
                    command, cwd=repo_root, stdout=log, stderr=subprocess.STDOUT, check=False
                )
            if proc.returncode:
                raise ValueError(f"Color chunk failed: {directory.name}; see extraction.log")
        coverage = json.loads((result_dir / "coverage.json").read_text(encoding="utf-8"))
        if (
            coverage["input_rows"] != len(subset)
            or coverage["successful"] != len(subset)
            or coverage["failed"]
            or coverage["excluded"]
            or coverage["manifest_sha256"] != digest(chunk_manifest)
            or coverage["config_sha256"] != signature["config_sha256"]
            or coverage["implementation_sha256"] != signature["implementation_sha256"]
            or coverage["runner_sha256"] != signature["extractor_runner_sha256"]
        ):
            raise ValueError(f"Invalid color chunk coverage/provenance: {directory.name}")
        with (result_dir / "palettes.jsonl").open(encoding="utf-8") as stream:
            result_ids = [json.loads(line)["item_id"] for line in stream]
        if result_ids != [r["item_id"] for r in subset]:
            raise ValueError(f"Color chunk row alignment mismatch: {directory.name}")
        write_json(
            receipt,
            {
                "manifest_sha256": digest(chunk_manifest),
                "outputs": {name: digest(result_dir / name) for name in files},
            },
        )
        return coverage

    completed, reports = 0, {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(run_chunk, job): job for job in jobs}
        for future in as_completed(pending):
            directory, _, subset = pending[future]
            reports[directory.name] = future.result()
            completed += len(subset)
            print(f"LAB: {completed}/{len(rows)} verified", flush=True)
    with (output / "palettes.jsonl").open("w", encoding="utf-8", newline="\n") as merged:
        for directory, _, _ in jobs:
            with (directory / "results/palettes.jsonl").open(encoding="utf-8") as stream:
                shutil.copyfileobj(stream, merged)
    for path, expected in zip(
        inputs,
        (
            signature["manifest_sha256"],
            signature["config_sha256"],
            signature["extractor_runner_sha256"],
            signature["implementation_sha256"],
        ),
    ):
        if digest(path) != expected:
            raise ValueError(f"Color input/code changed during extraction: {path}")
    first = reports[jobs[0][0].name]
    preview = jobs[0][0] / "results/palette_contact_sheet.png"
    if preview.exists():
        shutil.copyfile(preview, output / "palette_contact_sheet.png")
    report = {
        **signature,
        "schema_version": "wardiq.parallel-color-run.v1",
        "orchestrator_sha256": digest(__file__),
        "scope": first["scope"],
        "input_rows": len(rows),
        "successful": len(rows),
        "failed": 0,
        "excluded": 0,
        "numerical_threads_per_worker": 1,
        "workers": workers,
        "full_dataset_coverage_claimed": False,
        "review_status": "pending_visual_review",
        "background_risk_records": sum(r["background_risk_records"] for r in reports.values()),
        "source_records": "unchanged per-image algorithm; input-order chunk merge",
        "palettes_sha256": digest(output / "palettes.jsonl"),
        "chunks": {directory.name: digest(directory / "verified.json") for directory, _, _ in jobs},
    }
    write_json(output / "coverage.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            extract(
                args.manifest,
                args.config,
                args.output,
                args.repo_root,
                workers=args.workers,
                chunk_size=args.chunk_size,
                resume=args.resume,
            ),
            indent=2,
        )
    )
