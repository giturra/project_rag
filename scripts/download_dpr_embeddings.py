#!/usr/bin/env python3
"""
download_dpr_embeddings_parquet.py

Works with datasets>=4.* (no dataset scripts).

It loads the DPR passage parquet shards directly from the HF repo:
  hf://datasets/facebook/wiki_dpr@main/data/psgs_w100/<subset>/*.parquet

Subset mapping (from your old config string):
  psgs_w100.nq.*        -> data/psgs_w100/nq
  psgs_w100.multiset.*  -> data/psgs_w100/multiset
  psgs_w100.no_embeddings.* -> data/psgs_w100/no_embeddings  (NO embeddings column)

Examples:
  # preview
  python download_dpr_embeddings_parquet.py --config psgs_w100.nq.exact --max_rows 3

  # export embeddings + meta shards
  python download_dpr_embeddings_parquet.py \
    --config psgs_w100.nq.exact \
    --out_dir data/wiki_dpr_nq \
    --shard_size 200000 \
    --write_embeddings --write_meta
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ExportStats:
    rows_processed: int = 0
    shards_written: int = 0
    last_shard_rows: int = 0


def set_hf_cache_dirs(hf_home: Optional[str], hf_datasets_cache: Optional[str]) -> None:
    if hf_home:
        os.environ["HF_HOME"] = hf_home
    if hf_datasets_cache:
        os.environ["HF_DATASETS_CACHE"] = hf_datasets_cache


def parse_config_to_subset_dir(config: str) -> str:
    """
    Convert 'psgs_w100.nq.exact' -> 'nq'
            'psgs_w100.multiset.compressed' -> 'multiset'
            'psgs_w100.no_embeddings' / 'psgs_w100.no_embeddings.no_index' -> 'no_embeddings'
    """
    parts = config.split(".")
    if len(parts) < 2:
        raise ValueError(f"Unexpected config: {config}")

    if parts[0] != "psgs_w100":
        raise ValueError(f"Only psgs_w100 supported here, got: {parts[0]}")

    subset = parts[1]
    if subset not in {"nq", "multiset", "no_embeddings"}:
        raise ValueError(
            f"Unsupported subset '{subset}'. Use nq|multiset|no_embeddings. "
            f"Got config={config}"
        )
    return subset


def load_wiki_dpr_parquet(
    config: str,
    split: str = "train",
    hf_home: Optional[str] = None,
    hf_datasets_cache: Optional[str] = None,
):
    """
    Loads the parquet shards directly (no dataset script).
    """
    from datasets import load_dataset

    set_hf_cache_dirs(hf_home, hf_datasets_cache)

    subset = parse_config_to_subset_dir(config)
    # The repo uses train-xxxxx-of-yyyyy.parquet under this directory.
    data_files = f"hf://datasets/facebook/wiki_dpr@main/data/psgs_w100/{subset}/{split}-*.parquet"

    ds = load_dataset(
        "parquet",
        data_files=data_files,
        split="train",  # parquet builder uses 'train' as the split name
    )
    return ds


def infer_embedding_key(example: Dict[str, Any]) -> Optional[str]:
    if "embeddings" in example:
        return "embeddings"
    for k, v in example.items():
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], (int, float)):
            return k
    return None


def export_shards(
    ds,
    out_dir: str,
    shard_size: int = 200_000,
    max_rows: Optional[int] = None,
    write_embeddings: bool = True,
    write_meta: bool = True,
    embedding_key: Optional[str] = None,
) -> ExportStats:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if len(ds) == 0:
        raise ValueError("Dataset is empty for the given config/split.")

    if embedding_key is None:
        embedding_key = infer_embedding_key(ds[0])

    if write_embeddings:
        if not embedding_key:
            raise ValueError(
                "Could not infer embedding key. If you loaded 'no_embeddings' subset, "
                "disable --write_embeddings."
            )
        if embedding_key not in ds.column_names:
            raise ValueError(f"Embedding key '{embedding_key}' not in columns: {ds.column_names}")

    all_cols = list(ds.column_names)
    meta_cols = [c for c in all_cols if c != embedding_key] if embedding_key else all_cols

    stats = ExportStats()
    emb_buf: List[np.ndarray] = []
    meta_buf: List[Dict[str, Any]] = []

    def flush(shard_idx: int) -> int:
        nonlocal emb_buf, meta_buf, stats
        n = len(meta_buf) if write_meta else (len(emb_buf) if write_embeddings else 0)
        if n == 0:
            return 0

        shard_prefix = out_path / f"shard_{shard_idx:05d}"

        if write_embeddings:
            emb_mat = np.stack(emb_buf, axis=0).astype(np.float32, copy=False)
            np.save(str(shard_prefix) + "_embeddings.npy", emb_mat)

        if write_meta:
            meta_file = str(shard_prefix) + "_meta.jsonl"
            with open(meta_file, "w", encoding="utf-8") as f:
                for m in meta_buf:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")

        stats.shards_written += 1
        stats.last_shard_rows = n
        emb_buf = []
        meta_buf = []
        return n

    shard_idx = 0
    total = len(ds) if max_rows is None else min(len(ds), max_rows)

    for i in range(total):
        ex = ds[i]

        if write_embeddings:
            emb_buf.append(np.asarray(ex[embedding_key], dtype=np.float32))

        if write_meta:
            meta_buf.append({k: ex[k] for k in meta_cols})

        stats.rows_processed += 1

        if stats.rows_processed % shard_size == 0:
            flush(shard_idx)
            shard_idx += 1

    if (write_meta and meta_buf) or (write_embeddings and emb_buf):
        flush(shard_idx)

    return stats


def print_dataset_info(ds, max_rows: int = 3) -> None:
    print(f"Loaded split with {len(ds)} rows")
    if len(ds) == 0:
        return
    print("Columns:", ds.column_names)
    emb_key = infer_embedding_key(ds[0])
    print("Inferred embedding key:", emb_key)

    for i in range(min(max_rows, len(ds))):
        ex = ds[i]
        preview = {k: ex[k] for k in ex.keys() if k != emb_key}
        if "text" in preview and isinstance(preview["text"], str):
            preview["text"] = preview["text"][:160] + ("..." if len(preview["text"]) > 160 else "")
        print(f"\nRow {i} meta preview:")
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        if emb_key and emb_key in ex:
            v = ex[emb_key]
            print(f"Embedding len: {len(v)}  (first 5: {v[:5]})")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download + export DPR wiki embeddings (parquet, no scripts).")
    p.add_argument("--config", type=str, default="psgs_w100.nq.exact")
    p.add_argument("--split", type=str, default="train")

    p.add_argument("--hf_home", type=str, default=None)
    p.add_argument("--hf_datasets_cache", type=str, default=None)

    p.add_argument("--max_rows", type=int, default=0)

    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--shard_size", type=int, default=200_000)

    p.add_argument("--write_embeddings", action="store_true")
    p.add_argument("--write_meta", action="store_true")
    p.add_argument("--embedding_key", type=str, default=None)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    ds = load_wiki_dpr_parquet(
        config=args.config,
        split=args.split,
        hf_home=args.hf_home,
        hf_datasets_cache=args.hf_datasets_cache,
    )

    preview_rows = 3 if args.max_rows <= 0 else min(3, args.max_rows)
    print_dataset_info(ds, max_rows=preview_rows)

    if args.out_dir:
        # default to both if neither flag is set
        write_embeddings = args.write_embeddings or (not args.write_embeddings and not args.write_meta)
        write_meta = args.write_meta or (not args.write_embeddings and not args.write_meta)

        max_rows = None if args.max_rows <= 0 else args.max_rows

        stats = export_shards(
            ds,
            out_dir=args.out_dir,
            shard_size=args.shard_size,
            max_rows=max_rows,
            write_embeddings=write_embeddings,
            write_meta=write_meta,
            embedding_key=args.embedding_key,
        )
        print("\nExport complete:")
        print(f"- rows_processed: {stats.rows_processed}")
        print(f"- shards_written: {stats.shards_written}")
        print(f"- last_shard_rows: {stats.last_shard_rows}")
        print(f"- out_dir: {args.out_dir}")
    else:
        print("\nNo --out_dir provided, so nothing was exported (download is in HF cache).")


if __name__ == "__main__":
    main()
