# Picture Compliance Processing Engine — Walkthrough

## Summary

Implemented a complete **图像合规处理引擎** module at `compliance-checker/picture/` with full modular architecture, matching the existing `audio` module's patterns (FastAPI, Pydantic, pydantic-settings, pytest).

## New Files (40+ files)

### Domain Layer
| File | Purpose |
|------|---------|
| [enums.py](file:///d:/CodeVS/CodePython/compliance-checker/picture/domain/enums.py) | RouteType, JobStatus, DecisionType, FindingType, RedactionMode, SafetyCategory, PIIEntityType, VisionObjectType, FailurePolicy |
| [models.py](file:///d:/CodeVS/CodePython/compliance-checker/picture/domain/models.py) | PictureJob, PictureFinding, PictureReport, PictureAsset, BBox, RegionMask, OCRLayoutResult, etc. |
| [exceptions.py](file:///d:/CodeVS/CodePython/compliance-checker/picture/domain/exceptions.py) | Layered exception hierarchy: PictureError → ProviderError, JobError, StorageError, etc. |
| [policy.py](file:///d:/CodeVS/CodePython/compliance-checker/picture/domain/policy.py) | ConfigurablePolicyEngine — YAML-based profile evaluation |

### Provider Layer (11 providers)
| Provider | Mock ✅ | Real Skeleton |
|----------|---------|---------------|
| Router | HeuristicRouter (aspect ratio + color variance) | — |
| Preprocessor | DefaultPreprocessor (EXIF strip, rotate, resize, PDF) | — |
| OCR | MockOCRLayoutProvider | PaddleOCR-VL, MinerU, Surya |
| PII | MockPIIDetector (regex patterns) | Presidio |
| Safety | MockSafetyModerator (filename heuristic) | ShieldGemma 2 |
| Vision | MockVisionDetector | YOLO26, Grounding DINO |
| Segmentation | MockSegmentationProvider | SAM 2 |
| Redaction | OpenCVRedactor (Pillow fallback) — **fully functional** | — |
| Storage | LocalFileStorageBackend ✅ | S3StorageBackend (skeleton) |

### Application Layer
| File | Purpose |
|------|---------|
| [orchestrator.py](file:///d:/CodeVS/CodePython/compliance-checker/picture/application/orchestrator.py) | 3 processing chains: document, natural, mixed (parallel) |
| [services.py](file:///d:/CodeVS/CodePython/compliance-checker/picture/application/services.py) | Individual service functions with timing & IoU-based dedup |
| [use_cases.py](file:///d:/CodeVS/CodePython/compliance-checker/picture/application/use_cases.py) | Provider factory + `process_image()` convenience function |

### API Layer
| File | Purpose |
|------|---------|
| [app.py](file:///d:/CodeVS/CodePython/compliance-checker/picture/api/app.py) | FastAPI entry point with CORS & health check |
| [routes.py](file:///d:/CodeVS/CodePython/compliance-checker/picture/api/routes.py) | 5 endpoints: create, status, result, findings, rerun |
| [schemas.py](file:///d:/CodeVS/CodePython/compliance-checker/picture/api/schemas.py) | Request/response schemas with OpenAPI support |

### Infrastructure
| File | Purpose |
|------|---------|
| [config.py](file:///d:/CodeVS/CodePython/compliance-checker/picture/infra/config.py) | PictureSettings with `PICTURE_` env prefix |
| [storage.py](file:///d:/CodeVS/CodePython/compliance-checker/picture/infra/storage.py) | Local + S3 storage backends |
| [repository.py](file:///d:/CodeVS/CodePython/compliance-checker/picture/infra/repository.py) | Thread-safe in-memory job repository |

### Config & Tests
| File | Purpose |
|------|---------|
| [default_cn_enterprise.yaml](file:///d:/CodeVS/CodePython/compliance-checker/picture/configs/default_cn_enterprise.yaml) | Default policy profile |
| test_orchestrator.py | 9 tests: all 3 chains + auto-routing + errors |
| test_policy.py | 11 tests: all decision paths + edge cases |
| test_redaction.py | 8 tests: all 4 modes + overlay |
| test_api.py | 12 tests: all endpoints + error cases |

## Test Results

```
40 passed, 14 warnings in 2.0s
```

## Smoke Test Results

```
document: decision=pass_redacted   findings=7  redactions=7
 natural: decision=pass_redacted   findings=3  redactions=3
   mixed: decision=pass_redacted   findings=7+ redactions=7+
  unsafe: decision=drop            status=DROPPED
```

## What Was Tested
- All 3 processing chains (document, natural, mixed)
- Mixed chain parallel execution (OCR + safety, PII + vision)
- Policy decisions: pass_raw, pass_redacted, drop
- All 4 redaction modes (black_box, gaussian_blur, pixelate, solid_fill)
- Overlay rendering
- All HTTP API endpoints
- Error handling (nonexistent file, bad MIME type, missing profile)
- Findings-to-image coordinate mapping
- IoU-based finding dedup

## No Files Modified Outside `picture/`
All changes are contained within `compliance-checker/picture/`.

## Directly Runnable vs. Skeleton Providers

| Provider | Status |
|----------|--------|
| All Mock providers | ✅ Fully runnable |
| OpenCVRedactor (Pillow) | ✅ Fully runnable |
| HeuristicRouter | ✅ Fully runnable |
| DefaultPreprocessor | ✅ Fully runnable |
| LocalFileStorageBackend | ✅ Fully runnable |
| PaddleOCR-VL | 🔧 Adapter skeleton (needs `paddleocr` install) |
| Presidio | 🔧 Adapter skeleton (needs `presidio-analyzer`) |
| ShieldGemma 2 | 🔧 Adapter skeleton (needs `transformers + torch`) |
| YOLO26 | 🔧 Adapter skeleton (needs `ultralytics`) |
| Grounding DINO | 🔧 Adapter skeleton (needs `groundingdino`) |
| SAM 2 | 🔧 Adapter skeleton (needs `segment-anything-2`) |
| MinerU / Surya | 🔧 Minimal skeleton (TODO implementation) |
| S3StorageBackend | 🔧 Adapter skeleton (needs `boto3`) |
