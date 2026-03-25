# ──────────────────────────────────────────────────────────────
# 流水线步骤子包初始化 (Steps Package)
# ──────────────────────────────────────────────────────────────
#
# 本包包含流水线的 10 个主要步骤（14 个模块）：
#
#   a_source_intake.py      - 步骤 A: 输入接入
#   b1_source_classify.py   - 步骤 B1: 来源分类
#   b2a_trufflehog_scan.py  - 步骤 B2a: TruffleHog 密钥扫描
#   b2b_scancode_scan.py    - 步骤 B2b: ScanCode 许可证扫描
#   c_text_extract.py       - 步骤 C: 文本提取与清洗
#   d_dedup.py              - 步骤 D: 去重（精确 + 近似）
#   e1a_keyword_scan.py     - 步骤 E1a: FlashText2 关键词扫描
#   e1b_regex_scan.py       - 步骤 E1b: Hyperscan 正则扫描
#   f_privacy_detection.py  - 步骤 F: Presidio 隐私检测
#   g_safety_moderation.py  - 步骤 G: Qwen3Guard 安全审核
#   h_evidence_aggregation.py - 步骤 H: 证据聚合
#   i_policy_decision.py    - 步骤 I: OPA 策略决策
#   j_lineage_audit.py      - 步骤 J: OpenLineage 血缘审计
#
# 每个步骤模块均导出 run() 函数作为公共 API。
# ──────────────────────────────────────────────────────────────
