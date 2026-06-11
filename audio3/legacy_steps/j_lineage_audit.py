"""
Step J: lineage audit.
"""

# 复用 text 目录中的旧谱系跟踪实现，避免重复维护。
from text.legacy_steps.j_lineage_audit import LineageTracker

# 对外仅暴露 LineageTracker。
__all__ = ["LineageTracker"]
