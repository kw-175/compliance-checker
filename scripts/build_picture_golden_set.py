#!/usr/bin/env python
"""Build a golden picture set from historical compliance_output_picture jobs.

Scans historical job reports, groups by finding category, and copies the
preprocessed source image for up to --per-category jobs into
test_data/golden_picture/<category>/. The manifest keeps the historical
findings as REFERENCE expectations (not hard assertions): detection quality
work compares current runs against them and against previous bench snapshots.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "test_data" / "golden_picture"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="compliance_output_picture")
    parser.add_argument("--per-category", type=int, default=3)
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    by_category: dict[str, list[dict]] = defaultdict(list)
    for report_path in sorted(glob.glob(str(ROOT / args.source / "*" / "report.json"))):
        try:
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        except Exception:
            continue
        findings = [f for f in report.get("findings", []) if isinstance(f, dict)]
        if not findings:
            continue
        preprocess = sorted(Path(report_path).parent.glob("preprocess/*"))
        if not preprocess:
            continue
        categories = sorted({str(f.get("category") or f.get("finding_type") or "") for f in findings})
        entry = {
            "job_id": report.get("job_id") or Path(report_path).parent.name,
            "report": str(Path(report_path).relative_to(ROOT)),
            "image": str(preprocess[0].relative_to(ROOT)),
            "categories": categories,
            "decision": str(report.get("decision") or ""),
            "reference_findings": [
                {
                    "category": f.get("category") or f.get("finding_type"),
                    "label": f.get("label"),
                    "provider": f.get("provider"),
                    "bbox": (f.get("region") or {}).get("bbox") if isinstance(f.get("region"), dict) else None,
                    "score": f.get("score"),
                }
                for f in findings
            ],
        }
        for category in categories:
            by_category[category].append(entry)

    manifest: list[dict] = []
    for category in sorted(by_category):
        chosen = by_category[category][: args.per_category]
        category_dir = OUTPUT_ROOT / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for index, entry in enumerate(chosen):
            src = ROOT / entry["image"]
            if not src.exists():
                continue
            dst = category_dir / f"{category}_{index}{src.suffix}"
            if not dst.exists():
                shutil.copy2(src, dst)
            manifest.append({**entry, "golden_path": str(dst.relative_to(ROOT)), "primary_category": category})

    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"golden set: {len(manifest)} samples across {len(by_category)} categories -> {manifest_path}")


if __name__ == "__main__":
    main()
