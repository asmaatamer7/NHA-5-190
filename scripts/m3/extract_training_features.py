"""Checkpointed batches around the existing, unchanged pinned CLIP extractor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def extract(manifest, output, *, chunk_size=2048, batch_size=32, resume=False):
    import pandas as pd

    from wardiq.representation.clip_embeddings import (
        CLIP_MODEL_ID,
        CLIP_MODEL_REVISION,
        ClipEmbeddingMetadata,
        extract_clip_embeddings,
        load_usable_manifest,
        validate_embedding_artifact,
        validate_embedding_inventory,
    )

    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    if chunk_size < 1 or batch_size < 1:
        raise ValueError("Chunk and batch sizes must be positive")
    if output.exists() and not resume:
        raise ValueError("Output exists; use --resume with the unchanged manifest")
    if output.exists() and resume and not (output / "run.json").is_file():
        raise ValueError("Resume requires the original run.json; use a new output directory")
    if output == manifest or output in manifest.parents:
        raise ValueError("Output overlaps manifest")
    rows = load_usable_manifest(manifest)
    if set(rows["target_split"]) != {"train"}:
        raise ValueError("Training extraction refuses non-training rows")
    if "image_sha256" not in rows:
        raise ValueError("Prepared image hashes are required")
    for row in rows.itertuples():
        if digest(row.resolved_image_path) != row.image_sha256:
            raise ValueError(f"Prepared image checksum mismatch: {row.item_id}")
    signature = {
        "manifest_sha256": digest(manifest),
        "items": len(rows),
        "chunk_size": chunk_size,
        "batch_size": batch_size,
        "model": CLIP_MODEL_ID,
        "revision": CLIP_MODEL_REVISION,
    }
    output.mkdir(parents=True, exist_ok=True)
    state = output / "run.json"
    if state.exists() and json.loads(state.read_text()) != signature:
        raise ValueError("Resume configuration differs from the original run")
    write_json(state, signature)
    chunks = []
    for start in range(0, len(rows), chunk_size):
        directory = output / "chunks" / f"{start:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        source = directory / "manifest.csv"
        subset = rows.iloc[start : start + chunk_size].copy()
        subset["image_path"] = subset["resolved_image_path"]
        subset.to_csv(source, index=False)
        receipt = directory / "verified.json"
        if receipt.is_file():
            saved = json.loads(receipt.read_text())
            if saved["input_sha256"] != digest(source):
                raise ValueError(f"Changed chunk manifest: {start}")
            for name, checksum in saved["outputs"].items():
                if digest(directory / name) != checksum:
                    raise ValueError(f"Changed completed chunk: {start}: {name}")
        else:
            extract_clip_embeddings(source, directory, batch_size=batch_size)
            files = ["clip_embeddings.npy", "clip_inventory.csv", "metadata.json"]
            write_json(
                receipt,
                {
                    "input_sha256": digest(source),
                    "outputs": {n: digest(directory / n) for n in files},
                },
            )
        vectors = np.load(directory / "clip_embeddings.npy", mmap_mode="r", allow_pickle=False)
        validate_embedding_artifact(vectors, ClipEmbeddingMetadata(), len(subset))
        inventory = pd.read_csv(directory / "clip_inventory.csv")
        validate_embedding_inventory(inventory, len(subset))
        if inventory["item_id"].tolist() != subset["item_id"].tolist():
            raise ValueError(f"Chunk item alignment mismatch: {start}")
        metadata = json.loads((directory / "metadata.json").read_text())
        if (
            metadata["model_id"] != CLIP_MODEL_ID
            or metadata["model_revision"] != CLIP_MODEL_REVISION
        ):
            raise ValueError("Chunk model provenance mismatch")
        chunks.append((start, directory, len(subset)))
        print(f"CLIP: {start + len(subset)}/{len(rows)} verified", flush=True)
    temporary = output / "clip_embeddings.tmp.npy"
    merged = np.lib.format.open_memmap(
        temporary, mode="w+", dtype="float32", shape=(len(rows), 512)
    )
    for start, directory, count in chunks:
        merged[start : start + count] = np.load(
            directory / "clip_embeddings.npy", allow_pickle=False
        )
    merged.flush()
    validate_embedding_artifact(merged, ClipEmbeddingMetadata(), len(rows))
    del merged
    for row in rows.itertuples():
        if digest(row.resolved_image_path) != row.image_sha256:
            raise ValueError(f"Image changed during extraction: {row.item_id}")
    if digest(manifest) != signature["manifest_sha256"]:
        raise ValueError("Manifest changed during extraction")
    temporary.replace(output / "clip_embeddings.npy")
    inventory = rows.copy()
    inventory.insert(0, "embedding_row_index", np.arange(len(rows)))
    validate_embedding_inventory(inventory, len(rows))
    inventory.to_csv(output / "clip_inventory.csv", index=False)
    metadata.update(
        item_count=len(rows),
        source_manifest=str(manifest),
        manifest_sha256=signature["manifest_sha256"],
        embedding_file="clip_embeddings.npy",
        inventory_file="clip_inventory.csv",
        runner_sha256=digest(__file__),
        extraction_mode="verified_checkpointed_chunks",
    )
    write_json(output / "metadata.json", metadata)
    write_json(
        output / "complete.json",
        {
            **signature,
            "outputs": {
                name: digest(output / name)
                for name in ("clip_embeddings.npy", "clip_inventory.csv", "metadata.json")
            },
        },
    )
    return metadata


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--chunk-size", default=2048, type=int)
    parser.add_argument("--threads", default=6, type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    extract(
        args.manifest,
        args.output,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
