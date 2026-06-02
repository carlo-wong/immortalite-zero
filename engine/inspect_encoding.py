"""Inspect encoding versions in checkpoint/shard files.

Run:
  python -m engine.inspect_encoding --checkpoint-dir checkpoints
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .encoding import ENCODING_VERSION

_SAMPLE_SHARD_PREFIX = "samples_iter_"
_SAMPLE_SHARD_SUFFIX = ".npz"


@dataclass
class _Record:
    kind: str
    path: str
    version: int | None
    note: str = ""


def _extract_checkpoint_version(path: str) -> tuple[int | None, str]:
    try:
        state = torch.load(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - defensive CLI path
        return None, f"load_error: {exc}"

    if not isinstance(state, dict):
        return None, "unexpected_state_type"
    return int(state.get("encoding_version", 1)), ""


def _extract_shard_version(path: str) -> tuple[int | None, str]:
    try:
        with np.load(path) as data:
            if "encoding_version" not in data:
                return 1, ""
            raw = np.asarray(data["encoding_version"]).reshape(-1)
            if raw.size == 0:
                return 1, ""
            return int(raw[0]), ""
    except Exception as exc:  # pragma: no cover - defensive CLI path
        return None, f"load_error: {exc}"


def _scan_dir(checkpoint_dir: str) -> list[_Record]:
    records: list[_Record] = []
    if not os.path.isdir(checkpoint_dir):
        return records

    for name in sorted(os.listdir(checkpoint_dir)):
        path = os.path.join(checkpoint_dir, name)
        if not os.path.isfile(path):
            continue
        if name.endswith(".pt"):
            version, note = _extract_checkpoint_version(path)
            records.append(_Record(kind="checkpoint", path=path, version=version, note=note))
        elif name.startswith(_SAMPLE_SHARD_PREFIX) and name.endswith(_SAMPLE_SHARD_SUFFIX):
            version, note = _extract_shard_version(path)
            records.append(_Record(kind="sample_shard", path=path, version=version, note=note))
    return records


def _print_group(records: list[_Record], kind: str) -> None:
    summary = _group_summary(records, kind)

    print(
        f"{kind}s: total={summary['total']} | v{ENCODING_VERSION}={summary['current']} | "
        f"v1={summary['v1']} | other={summary['other']} | unknown={summary['unknown']}"
    )


def _group_summary(records: list[_Record], kind: str) -> dict[str, int]:
    subset = [r for r in records if r.kind == kind]
    return {
        "total": len(subset),
        "current": sum(1 for r in subset if r.version == ENCODING_VERSION),
        "v1": sum(1 for r in subset if r.version == 1),
        "other": sum(1 for r in subset if r.version not in (None, 1, ENCODING_VERSION)),
        "unknown": sum(1 for r in subset if r.version is None),
    }


def _records_payload(records: list[_Record], *, only_incompatible: bool) -> list[dict[str, object]]:
    if only_incompatible:
        filtered = [r for r in records if r.version != ENCODING_VERSION]
    else:
        filtered = records
    payload: list[dict[str, object]] = []
    for rec in filtered:
        item = asdict(rec)
        item["status"] = "ok" if rec.version == ENCODING_VERSION else "mismatch"
        payload.append(item)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect encoding versions in checkpoints/shards.")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="directory to scan")
    parser.add_argument(
        "--only-incompatible",
        action="store_true",
        help="print only files not matching current encoding version",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit machine-readable JSON output",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint_dir):
        if args.json:
            print(
                json.dumps(
                    {
                        "checkpoint_dir": args.checkpoint_dir,
                        "current_encoding_version": ENCODING_VERSION,
                        "error": "checkpoint_dir_not_found",
                    }
                )
            )
        else:
            print(f"checkpoint dir not found: {args.checkpoint_dir}")
        return 2

    records = _scan_dir(args.checkpoint_dir)
    shown = _records_payload(records, only_incompatible=args.only_incompatible)
    if args.json:
        payload = {
            "checkpoint_dir": args.checkpoint_dir,
            "current_encoding_version": ENCODING_VERSION,
            "only_incompatible": bool(args.only_incompatible),
            "summary": {
                "checkpoints": _group_summary(records, "checkpoint"),
                "sample_shards": _group_summary(records, "sample_shard"),
            },
            "files": shown,
        }
        print(json.dumps(payload))
        return 0

    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"current_encoding_version: {ENCODING_VERSION}")
    _print_group(records, "checkpoint")
    _print_group(records, "sample_shard")

    if args.only_incompatible:
        print("\nincompatible files:")
    else:
        print("\nfiles:")

    for item in shown:
        version = item["version"]
        version_text = "unknown" if version is None else f"v{version}"
        note = f" ({item['note']})" if item["note"] else ""
        print(f"- [{item['kind']}] {item['status']} {version_text} {item['path']}{note}")

    if args.only_incompatible and not shown:
        print("- none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
