#!/usr/bin/env python
"""Run the current picture engine over the golden set and diff against a snapshot.

Usage:
  .venv/bin/python scripts/run_picture_golden_bench.py --tag before
  # ...make changes...
  .venv/bin/python scripts/run_picture_golden_bench.py --tag after
  .venv/bin/python scripts/run_picture_golden_bench.py --compare before after

Snapshots land in temp/golden_bench_<tag>.json. Per category we report
found / not-found (vs manifest primary category) and, with --compare,
finding-level label/bbox(IoU) drift between the two snapshots.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))


def _iou(a: dict, b: dict) -> float:
    try:
        ax1, ay1 = float(a["x"]), float(a["y"])
        ax2, ay2 = ax1 + float(a["w"]), ay1 + float(a["h"])
        bx1, by1 = float(b["x"]), float(b["y"])
        bx2, by2 = bx1 + float(b["w"]), by1 + float(b["h"])
        ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0
    except (KeyError, TypeError, ValueError):
        return 0.0


def run(tag: str, limit_per_category: int) -> None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("COMPLIANCE_COMPLIANCE_PROVIDER_MODE", "local")
    os.environ.setdefault("COMPLIANCE_LOCAL_COMPLIANCE_BASE_URL", "http://127.0.0.1:8200/openai/v1")
    os.environ.setdefault("COMPLIANCE_LOCAL_COMPLIANCE_MODEL", "Qwen/Qwen3.5-9B")
    from picture.application.use_cases import process_image

    manifest = json.loads((ROOT / "test_data" / "golden_picture" / "manifest.json").read_text(encoding="utf-8"))
    per_cat: dict[str, int] = {}
    results = []
    for entry in manifest:
        cat = entry["primary_category"]
        if per_cat.get(cat, 0) >= limit_per_category:
            continue
        per_cat[cat] = per_cat.get(cat, 0) + 1
        image = str(ROOT / entry["golden_path"])
        t0 = time.time()
        try:
            job = process_image(image, tenant_id="golden")
            data = job.model_dump() if hasattr(job, "model_dump") else {}
            findings = [
                {
                    "category": f.get("category") or f.get("finding_type"),
                    "label": f.get("label"),
                    "bbox": (f.get("region") or {}).get("bbox") if isinstance(f.get("region"), dict) else None,
                }
                for f in (data.get("findings") or [])
                if isinstance(f, dict)
            ]
            row = {
                "golden_path": entry["golden_path"],
                "primary_category": cat,
                "elapsed_s": round(time.time() - t0, 1),
                "decision": str(data.get("status")),
                "primary_found": any(str(f["category"]) == cat for f in findings),
                "findings": findings,
            }
        except Exception as exc:
            row = {
                "golden_path": entry["golden_path"],
                "primary_category": cat,
                "elapsed_s": round(time.time() - t0, 1),
                "error": f"{type(exc).__name__}: {exc}",
                "primary_found": False,
                "findings": [],
            }
        results.append(row)
        print(json.dumps({k: row[k] for k in ("primary_category", "primary_found", "elapsed_s")}, ensure_ascii=False), flush=True)

    out = ROOT / "temp" / f"golden_bench_{tag}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    found = sum(1 for r in results if r["primary_found"])
    print(f"TOTAL tag={tag}: {found}/{len(results)} primary categories found -> {out}")


def compare(before_tag: str, after_tag: str) -> None:
    before = json.loads((ROOT / "temp" / f"golden_bench_{before_tag}.json").read_text(encoding="utf-8"))
    after = json.loads((ROOT / "temp" / f"golden_bench_{after_tag}.json").read_text(encoding="utf-8"))
    bmap = {r["golden_path"]: r for r in before}
    diffs = 0
    for a in after:
        b = bmap.get(a["golden_path"])
        if b is None:
            continue
        notes = []
        if a["primary_found"] != b["primary_found"]:
            notes.append(f"primary_found {b['primary_found']}->{a['primary_found']}")
        a_cats = sorted({str(f["category"]) for f in a["findings"]})
        b_cats = sorted({str(f["category"]) for f in b["findings"]})
        if a_cats != b_cats:
            notes.append(f"categories {b_cats}->{a_cats}")
        for f in a["findings"]:
            if f["bbox"]:
                match = max(
                    (_iou(f["bbox"], g["bbox"]) for g in b["findings"] if g["bbox"] and g["category"] == f["category"]),
                    default=0.0,
                )
                if match < 0.5:
                    notes.append(f"bbox drift {f['category']} IoU={match:.2f}")
        if notes:
            diffs += 1
            print(f"DIFF {a['golden_path']}: {'; '.join(notes[:4])}")
    print(f"COMPARE {before_tag}->{after_tag}: {diffs}/{len(after)} samples differ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--per-category", type=int, default=1)
    args = parser.parse_args()
    if args.compare:
        compare(args.compare[0], args.compare[1])
    else:
        if not args.tag:
            parser.error("--tag is required for a bench run")
        run(args.tag, args.per_category)


if __name__ == "__main__":
    main()
