"""Test parallel ordering, checkpoint integrity and failure accounting."""

import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/m3"))
import extract_training_colors as colors  # noqa: E402


def setup(tmp_path, monkeypatch, fail=False):
    repo = tmp_path / "repo"
    cli = repo / "scripts/extract_color_palettes.py"
    implementation = repo / "src/wardiq/clothing/color_extraction.py"
    for path in (cli, implementation):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# producer fixture")
    config = tmp_path / "config.yaml"
    config.write_text("scope: fixture\n")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("item_id,target_split\na,train\nb,train\nc,train\nd,train\n")
    calls = []

    def fake_cli(command, **kwargs):
        source = Path(command[command.index("--manifest") + 1])
        output = Path(command[command.index("--output-dir") + 1])
        calls.append(source)
        if fail:
            return SimpleNamespace(returncode=1)
        with source.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if rows[0]["item_id"] == "a":
            time.sleep(0.02)  # Force completion order to differ from input order.
        output.mkdir()
        (output / "palettes.jsonl").write_text(
            "".join(json.dumps({"item_id": r["item_id"]}) + "\n" for r in rows)
        )
        (output / "coverage.json").write_text(
            json.dumps(
                {
                    "input_rows": len(rows),
                    "successful": len(rows),
                    "failed": 0,
                    "excluded": 0,
                    "manifest_sha256": colors.digest(source),
                    "config_sha256": colors.digest(config),
                    "implementation_sha256": colors.digest(implementation),
                    "runner_sha256": colors.digest(cli),
                    "scope": "fixture",
                    "background_risk_records": 0,
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(colors.subprocess, "run", fake_cli)
    return manifest, config, tmp_path / "out", repo, calls


def test_parallel_merge_preserves_order_and_reuses_verified_chunks(tmp_path, monkeypatch):
    manifest, config, output, repo, calls = setup(tmp_path, monkeypatch)
    result = colors.extract(manifest, config, output, repo, workers=2, chunk_size=2)
    assert result["successful"] == 4 and result["failed"] == 0
    records = [json.loads(line) for line in (output / "palettes.jsonl").read_text().splitlines()]
    assert [r["item_id"] for r in records] == ["a", "b", "c", "d"]
    colors.extract(manifest, config, output, repo, workers=1, chunk_size=2, resume=True)
    assert len(calls) == 2


@pytest.mark.parametrize("case", ["tamper", "changed_input", "unrelated"])
def test_unsafe_resume_fails(tmp_path, monkeypatch, case):
    manifest, config, output, repo, _ = setup(tmp_path, monkeypatch)
    colors.extract(manifest, config, output, repo, workers=2, chunk_size=2)
    if case == "tamper":
        (output / "chunks/000000/results/palettes.jsonl").write_text("corrupt")
    elif case == "changed_input":
        manifest.write_text(manifest.read_text().replace("a,train", "other,train"))
    elif case == "unrelated":
        output = tmp_path / "unrelated"
        output.mkdir()
    with pytest.raises(ValueError):
        colors.extract(manifest, config, output, repo, workers=2, chunk_size=2, resume=True)


def test_worker_failure_never_produces_successful_global_report(tmp_path, monkeypatch):
    manifest, config, output, repo, _ = setup(tmp_path, monkeypatch, fail=True)
    with pytest.raises(ValueError, match="Color chunk failed"):
        colors.extract(manifest, config, output, repo, workers=2, chunk_size=2)
    assert not (output / "coverage.json").exists()
