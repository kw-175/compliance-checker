package compliance

default decision := {
  "overall_decision": "review",
  "unit_decisions": []
}

unit_decision(unit) := {
  "unit_id": unit.unit_id,
  "decision": local_decision(unit),
  "reasons": [],
  "scores": {}
}

local_decision(unit) := "reject" if {
  unit.secret_count > 0
}

local_decision(unit) := "quarantine" if {
  unit.secret_count == 0
  unit.safety_level == "unsafe"
}

local_decision(unit) := "review" if {
  unit.secret_count == 0
  unit.safety_level != "unsafe"
  unit.pii_count > 0
}

local_decision(unit) := "allow" if {
  unit.secret_count == 0
  unit.safety_level == "safe"
  unit.pii_count == 0
}
