"""Quick smoke test for all processing chains."""
from picture.application.use_cases import process_image
from picture.infra.config import PictureSettings
from pathlib import Path
import tempfile

with tempfile.TemporaryDirectory() as td:
    settings = PictureSettings(
        work_dir=Path(td) / "work",
        storage_base_path=Path(td) / "storage",
    )
    fixtures = Path("picture/tests/fixtures").resolve()

    # Test all 3 chains
    for hint, fixture in [
        ("document", "sample_document.png"),
        ("natural", "sample_natural.png"),
        ("mixed", "sample_mixed.png"),
    ]:
        img = str(fixtures / fixture)
        job = process_image(img, settings=settings, options={"route_hint": hint})
        print(
            f"{hint:>8}: decision={job.policy_result.decision.value:15} "
            f"findings={len(job.findings)} redactions={len(job.redaction_operations)} "
            f"reasons={job.policy_result.reason_codes[:3]}"
        )

    # Test DROP chain
    img = str(fixtures / "sample_unsafe_explicit.png")
    job = process_image(img, settings=settings, options={"route_hint": "natural"})
    print(
        f"{'unsafe':>8}: decision={job.policy_result.decision.value:15} "
        f"status={job.status.value} reasons={job.policy_result.reason_codes}"
    )

print("\nAll chains executed successfully!")
