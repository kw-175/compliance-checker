package compliance

# 默认输出：整体 review，且 unit 决策为空。
default decision := {
  "overall_decision": "review",
  "unit_decisions": []
}

# 构造单条 unit 的决策结构，实际决策由 local_decision 给出。
unit_decision(unit) := {
  "unit_id": unit.unit_id,
  "decision": local_decision(unit),
  "reasons": [],
  "scores": {}
}

# 规则优先级从高风险到低风险依次匹配。
local_decision(unit) := "reject" if {
  # 发现密钥类命中时直接拒绝。
  unit.secret_count > 0
}

local_decision(unit) := "quarantine" if {
  # 无密钥但安全等级不通过，进入隔离。
  unit.secret_count == 0
  unit.safety_level == "unsafe"
}

local_decision(unit) := "review" if {
  # 无密钥且非 unsafe，但仍存在 PII，进入人工复核。
  unit.secret_count == 0
  unit.safety_level != "unsafe"
  unit.pii_count > 0
}

local_decision(unit) := "allow" if {
  # 无密钥、内容安全、且无 PII 时允许发布。
  unit.secret_count == 0
  unit.safety_level == "safe"
  unit.pii_count == 0
}
