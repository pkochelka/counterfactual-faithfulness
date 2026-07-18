"""Shared iteration over judged JSONL trees (data/judged, data/judged_fluency).

Layout: <judged_dir>/<generator-model>/judge_<dataset>_<variant>_<language>_<judge-model>.jsonl
with *.failures.jsonl sidecars (error records, no parsed score) that are always
skipped. Consumers keep their own per-record field extraction; this module only
owns directory walking, stem parsing, and JSONL decoding.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, NamedTuple


def parse_stem(stem: str) -> tuple[str, str, str]:
    # judge_<dataset>_<variant>_<language>_<judge-model...>
    parts = stem.split("_")
    if len(parts) >= 4 and parts[0] == "judge":
        return parts[1], parts[2], parts[3]
    return "", "", ""


def iter_judged_files(judged_dir: Path) -> Iterator[tuple[str, Path]]:
    for model_dir in sorted(judged_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for jsonl_path in sorted(model_dir.glob("*.jsonl")):
            if jsonl_path.name.endswith(".failures.jsonl"):
                continue
            yield model_dir.name, jsonl_path


class JudgedRecord(NamedTuple):
    model: str
    dataset: str
    variant: str
    language: str
    path: Path
    line_no: int
    record: dict[str, Any]


def iter_judged_records(judged_dir: Path) -> Iterator[JudgedRecord]:
    for model, jsonl_path in iter_judged_files(judged_dir):
        dataset, variant, language = parse_stem(jsonl_path.stem)
        with jsonl_path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                yield JudgedRecord(model, dataset, variant, language, jsonl_path, line_no, record)


def normalize_classified_model_name(name: str) -> str:
    """Map a data/classified/ model directory name to its data/judged/ counterpart.

    The two trees name a couple of models differently: classified drops
    judged's underscore-for-dot ("qwen3.5-122b" vs "qwen3_5-122b") and keeps
    an "-openrouter" suffix judged does not ("llama4-scout-openrouter" vs
    "llama4-scout").
    """
    if name.endswith("-openrouter"):
        name = name[: -len("-openrouter")]
    if "." in name:
        name = name.replace(".", "_", 1)
    return name
