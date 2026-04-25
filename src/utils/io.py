from pathlib import Path
import json
from typing import Any, Iterable, Iterator


def load_json(path: Path | str) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def iter_jsonl(path: Path | str) -> Iterator[Any]:
    """
    Stream JSON objects from a JSONL/NDJSON file.

    Each non-empty line must contain one valid JSON object/value.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e


def iter_jsonl_batches(path: Path | str, batch_size: int) -> Iterator[list[Any]]:
    """
    Stream a JSONL file in batches.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    batch: list[Any] = []
    for item in iter_jsonl(path):
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch

def save_json(obj: Any, path: Path | str, *, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)

def save_jsonl(items: Iterable[Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
