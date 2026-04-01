"""Quick smoke test for all processing chains."""
# 中文说明：这是一个手动烟测脚本，用最少代码验证三条主链路能否端到端跑通。
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

    # 中文说明：依次验证 document / natural / mixed 三条链路。
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

    # 中文说明：额外验证一次会触发 DROP 的 unsafe 场景。
    img = str(fixtures / "sample_unsafe_explicit.png")
    job = process_image(img, settings=settings, options={"route_hint": "natural"})
    print(
        f"{'unsafe':>8}: decision={job.policy_result.decision.value:15} "
        f"status={job.status.value} reasons={job.policy_result.reason_codes}"
    )

print("\nAll chains executed successfully!")
