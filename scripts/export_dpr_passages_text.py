#!/usr/bin/env python3

import argparse
import json
from datasets import load_dataset


def parse_config_to_subset(config: str) -> str:
    parts = config.split(".")
    if len(parts) < 2 or parts[0] != "psgs_w100":
        raise ValueError(f"Invalid config: {config}")
    return parts[1]


def load_dataset_text_only(config: str, split: str):
    subset = parse_config_to_subset(config)

    data_files = f"hf://datasets/facebook/wiki_dpr@main/data/psgs_w100/{subset}/{split}-*.parquet"

    ds = load_dataset(
        "parquet",
        data_files=data_files,
        split="train"
    )

    return ds


def export_text(ds, out_file: str, max_rows: int = None):
    total = len(ds) if max_rows is None else min(len(ds), max_rows)

    with open(out_file, "w", encoding="utf-8") as f:
        for i in range(total):
            ex = ds[i]

            row = {
                "id": ex.get("id"),
                "title": ex.get("title"),
                "text": ex.get("text"),
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"✅ Wrote {total} rows to {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Download DPR text only")
    parser.add_argument("--config", default="psgs_w100.nq.exact")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out_file", required=True)
    parser.add_argument("--max_rows", type=int, default=0)

    args = parser.parse_args()

    ds = load_dataset_text_only(args.config, args.split)

    max_rows = None if args.max_rows <= 0 else args.max_rows

    export_text(ds, args.out_file, max_rows=max_rows)


if __name__ == "__main__":
    main()