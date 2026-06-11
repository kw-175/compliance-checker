from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def write_jsonl(records: Iterable[BaseModel | dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if isinstance(record, BaseModel):
                handle.write(record.model_dump_json() + "\n")
            else:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_single_jsonl(record: BaseModel | dict, path: Path) -> None:
    write_jsonl([record], path)


def read_jsonl(path: Path, model: type[T] | None = None) -> list[T] | list[dict]:
    if not path.exists():
        return []

    records: list[T] | list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            data = json.loads(line)
            if model is None:
                records.append(data)
            else:
                records.append(model.model_validate(data))
    return records
