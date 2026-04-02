# 四种模态合规检测处理工作流程完整分析

## 概览表

| 模态 | 输入类型 | 核心链路 | 主要工具/模型 | 输出形式 | 最终决策 |
|------|---------|--------|-------------|--------|--------|
| **TEXT** | 文本文件、代码、PDF、HTML | A→B1→B2(∥)→C→D→E1(∥)→F→G→H→I→J | TruffleHog, ScanCode, trafilatura, Presidio, Qwen3Guard, OPA | JSONL/JSON | Allow/Review/Reject |
| **AUDIO** | 音频文件(.wav, .mp3等) | A→B1→B2(∥)→C0→C1→C1b→C1c→C2→D→E1(∥)→F→G→H→I→K→L | FFmpeg, Qwen3-ASR, faster-whisper, pyannote, Presidio, Qwen3Guard | JSONL/JSON + 脱敏音频 | Allow/Review/Reject |
| **PICTURE** | 图像文件、PDF页、截图 | Route + 三条链(Document/Natural/Mixed) + 脱敏 | OCR, Presidio, ShieldGemma2, YOLO, SAM2, OpenCV | JSON + 脱敏图像 | pass_raw/pass_redacted/drop |
| **VIDEO** | 视频容器、GIF、帧序列 | 抽帧→逐帧×Picture→音轨×Audio→时序聚合→合并→渲染回写 | FFmpeg/ffprobe, Picture引擎, Audio引擎 | JSONL/JSON + 脱敏视频 | Allow/Review/Reject |

---

## 1. TEXT 模态详细流程

### 1.1 架构流程图

```
Input Files
    ↓
[A] Source Intake
    ├─ 遍历文件路径
    ├─ 计算 SHA-256 哈希
    ├─ 检测 MIME 类型
    └─ 生成 SourceRecord
    ↓
[B1] Source Classification
    ├─ 基于扩展名/MIME 分类
    └─ 结果: code / repo / package / binary / web_text / pdf_text / mixed
    ↓
    ├─→ [B2a] TruffleHog Scan (∥并行)
    │   ├─ 执行CLI: trufflehog filesystem {path}
    │   ├─ 检测 secrets (API key, token, password)
    │   └─ 输出: raw_secret_hits.jsonl
    │
    └─→ [B2b] ScanCode Scan (∥并行)
        ├─ 执行CLI: scancode {path}
        ├─ 检测 license 和 copyright
        └─ 输出: source_compliance.jsonl
    ↓
[C] Text Extract
    ├─ 路径选择（优先级）:
    │  ├─ HTML → trafilatura 提取
    │  ├─ PDF  → PyMuPDF 提取
    │  └─ 其他 → 直接读取
    ├─ Unicode 规范化
    ├─ 空白压缩
    └─ 输出: cleaned_documents.jsonl
    ↓
[D] Dedup
    ├─ 精确去重 (SHA-256)
    ├─ 近似去重 (MinHash LSH via datasketch)
    ├─ 消除重复文本
    └─ 输出: deduped_documents.jsonl, dedup_map.jsonl
    ↓
    ├─→ [E1a] Keyword Scan (∥并行)
    │   ├─ 加载 config/keywords.txt
    │   ├─ 优先 FlashText2 (高效多模式匹配)
    │   ├─ 回落 str.find (如无FlashText2)
    │   └─ 输出: keyword_hits.jsonl
    │
    └─→ [E1b] Regex Scan (∥并行)
        ├─ 加载 config/patterns.yaml
        ├─ 优先 Hyperscan (高效正则匹配)
        ├─ 回落 Python re (如无Hyperscan)
        └─ 输出: regex_hits.jsonl
    ↓
[F] Privacy Detection & Redaction
    ├─ 使用 Microsoft Presidio
    │  ├─ 识别 PII 实体 (PERSON, PHONE, EMAIL, CREDIT_CARD 等)
    │  ├─ 可选注册 Meddies/meddies-pii NER 增强识别器
    │  └─ 脱敏替换: <PHONE>, <EMAIL>, <PERSON> 等
    ├─ Fallback: 无Presidio则回退内置规则
    └─ 输出: privacy_checked.jsonl (含脱敏文本)
    ↓
[G] Safety Moderation
    ├─ 优先 Qwen3Guard (千问安全卫士)
    │  ├─ 三级分类: Safe / Controversial / Unsafe
    │  ├─ 模型: Qwen/Qwen3-Guard-0.6B
    │  ├─ 检测类别: violent, sexual, political, jailbreak 等
    │  └─ 设备: GPU优先，CPU降级
    ├─ Fallback: 基于关键词 mock 分类
    └─ 输出: safety_checked.jsonl
    ↓
[H] Evidence Aggregation
    ├─ 聚合各步骤证据
    ├─ 生成 summary: 文档数、命中数、风险统计
    └─ 输出: evidence_bundle.json
    ↓
[I] Policy Decision
    ├─ 优先路径: 调用 OPA REST API
    │  ├─ 加载 text/policies/compliance.rego
    │  └─ 根据证据做策略决策
    ├─ 回落路径: 本地规则引擎
    │  ├─ secrets 维度: 0-1 评分
    │  ├─ compliance 维度: 0-1 评分
    │  ├─ privacy 维度: 0-1 评分
    │  ├─ safety 维度: 0-1 评分
    │  └─ text_scan 维度: 0-1 评分
    ├─ 最终决策: Allow / Review / Reject
    └─ 输出: decision.json
    ↓
[J] Lineage Audit
    ├─ 发送 OpenLineage 事件
    ├─ 事件类型: START / COMPLETE / FAIL
    └─ 收集 provider_versions, latencies

```

### 1.2 各步骤详解

| 步骤 | 工具/库 | 输入 | 处理逻辑 | 输出文件 | 备注 |
|------|--------|------|--------|--------|------|
| A | 标准库 | 文件路径 | 递归遍历，计算SHA-256哈希 | source_registry.jsonl | SourceRecord[] |
| B1 | 内置 | SourceRecord[] | MIME检测 + 扩展名映射 | source_profile.jsonl | SourceProfile[] |
| B2a | TruffleHog CLI | SourceRecord[] | 执行 `trufflehog filesystem {path}`，解析JSON | raw_secret_hits.jsonl | SecretHit[]，并行执行，自动去重 |
| B2b | ScanCode CLI | SourceProfile[] | 执行 `scancode {path}`，解析license/copyright | source_compliance.jsonl | ComplianceHit[]，并行执行 |
| C | trafilatura / PyMuPDF | SourceProfile[] | HTML→trafilatura / PDF→PyMuPDF / 其他→直接读取+Unicode规范化 | cleaned_documents.jsonl | CleanedDocument[] |
| D | datasketch | CleanedDocument[] | SHA-256精确去重 + MinHash LSH近似去重(threshold=0.8) | deduped_documents.jsonl, dedup_map.jsonl | DedupDocument[] |
| E1a | FlashText2 / str.find | DedupDocument[] | Aho-Corasick自动机多模式匹配+上下文提取(±60字符) | keyword_hits.jsonl | KeywordHit[]，并行执行 |
| E1b | Hyperscan / re | DedupDocument[] | 正则引擎匹配+捕获组提取 | regex_hits.jsonl | RegexHit[]，并行执行 |
| F | Presidio | DedupDocument[] | AnalyzerEngine(en_core_web_sm+可选HF NER)检测PII，Anonymizer替换 | privacy_checked.jsonl | PrivacyResult[]（含脱敏文本） |
| G | Qwen3Guard | PrivacyResult[] | AutoModelForCausalLM推理，三级分类(SAFE/CONTROVERSIAL/UNSAFE) | safety_checked.jsonl | SafetyResult[] |
| H | 内置 | B2a/b + D + E1a/b + F + G | 按doc_id/source_id构建查找表，聚合+统计 | evidence_bundle.json | EvidenceBundle |
| I | OPA/本地规则 | EvidenceBundle | OPA(httpx POST) 或 五维评分+阈值决策 | decision.json | PolicyDecision |
| J | OpenLineage | 所有步骤 | 发送START/COMPLETE/FAIL事件 | — | 血缘追踪（不阻塞主流程） |

#### 1.2.1 E1a 关键词扫描详解

**实现原理：**
```python
# 加载 config/keywords.txt，逐行读取（跳过注释和空行）
keywords = _load_keywords(settings.keywords_file)  # list[str]

# 优先路径：FlashText2 KeywordProcessor（Aho-Corasick 自动机）
from flashtext import KeywordProcessor
kp = KeywordProcessor()
for kw in keywords:
    kp.add_keyword(kw)

# 对每个文档逐行扫描
for doc in documents:
    # 返回 [(keyword, start_offset, end_offset), ...]
    matches = kp.extract_keywords(doc.text, span_info=True)
    
    # 提取上下文（匹配位置前后各60字符）
    for kw, start, end in matches:
        context = text[max(0, start-60):min(len(text), end+60)]
        output(KeywordHit(doc_id, kw, start, end, context))

# Fallback（无FlashText2）：使用 str.find() 逐词搜索
for doc in documents:
    for kw in keywords:
        pos = 0
        while True:
            offset = doc.text.find(kw, pos)
            if offset == -1:
                break
            context = _extract_context(doc.text, offset, offset+len(kw))
            output(KeywordHit(doc_id, kw, offset, offset+len(kw), context))
            pos = offset + 1
```

**产物示例：**
```jsonl
{"doc_id": "doc_1", "keyword": "password", "start": 150, "end": 158, "context": "...the password in the config file..."}
{"doc_id": "doc_2", "keyword": "credit_card", "start": 500, "end": 511, "context": "...credit_card number 1234-5678..."}
```

#### 1.2.2 H 证据聚合详解

**聚合逻辑：**
```python
# 构建倒排索引
secrets_by_source: dict[source_id] = defaultdict(list)      # B2a 结果
compliance_by_source: dict[source_id] = defaultdict(list)   # B2b 结果
kw_by_doc: dict[doc_id] = defaultdict(list)                 # E1a 结果
regex_by_doc: dict[doc_id] = defaultdict(list)              # E1b 结果
privacy_by_doc: dict[doc_id] = PrivacyResult                # F 结果
safety_by_doc: dict[doc_id] = SafetyResult                  # G 结果

# 按文档汇总
for doc in dedup_docs:
    # 通过 doc.source_id 反向查询 B2 结果
    doc_evidence = DocumentEvidence(
        doc_id=doc.doc_id,
        source_id=doc.source_id,
        is_duplicate=doc.is_duplicate,
        secret_hits=secrets_by_source[doc.source_id],
        compliance_hits=compliance_by_source[doc.source_id],
        keyword_hits=kw_by_doc[doc.doc_id],
        regex_hits=regex_by_doc[doc.doc_id],
        privacy=privacy_by_doc[doc.doc_id],
        safety=safety_by_doc[doc.doc_id],
    )
    bundle.documents.append(doc_evidence)

# 生成全局统计
bundle.summary = {
    "total_documents": len(dedup_docs),
    "unique_documents": sum(1 for d in dedup_docs if not d.is_duplicate),
    "secrets_found": len(secret_hits),
    "compliance_issues": len(compliance_hits),
    "keyword_hits": len(keyword_hits),
    "regex_hits": len(regex_hits),
    "pii_count": sum(r.pii_count for r in privacy_results),
    "unsafe_count": sum(1 for r in safety_results if r.safety_level==UNSAFE),
}
```

**产物示例：**
```json
{
  "pipeline_run_id": "run_abc123",
  "summary": {
    "total_documents": 100,
    "unique_documents": 95,
    "secrets_found": 3,
    "pii_count": 15,
    "unsafe_count": 2
  },
  "documents": [
    {
      "doc_id": "doc_1",
      "source_id": "src_1",
      "is_duplicate": false,
      "secret_hits": [{...}],
      "keyword_hits": [{...}],
      "privacy": {"pii_count": 2, "entities": [...]},
      "safety": {"safety_level": "safe", "categories": []}
    }
  ]
}
```

#### 1.2.3 I 策略决策详解（五维评分引擎）

**决策流程：**
```python
# 优先路径：OPA REST API
try:
    # POST to OPA_URL/OPA_POLICY_PATH（通常是 /v1/data/compliance/decision）
    response = httpx.post(
        f"{settings.opa_url}/{settings.opa_policy_path}",
        json={"input": {"pipeline_run_id": ..., "documents": [...]}},
        timeout=30
    )
    decision = parse_opa_response(response)
    return decision
except Exception:
    logger.warning("OPA failed, falling back to local rules")

# Fallback：五维评分系统 + 阈值决策
for doc in evidence_bundle.documents:
    # 维度1：Secrets (有=0, 无=1)
    score_secrets = 0 if len(doc.secret_hits) > 0 else 1
    
    # 维度2：Safety (UNSAFE=0, CONTROVERSIAL=0.5, SAFE=1)
    safety_scores = {UNSAFE: 0, CONTROVERSIAL: 0.5, SAFE: 1}
    score_safety = safety_scores.get(doc.safety.safety_level, 1)
    
    # 维度3：Privacy (PII>5=0.3, PII>0=0.7, 无=1)
    if doc.privacy.pii_count > 5:
        score_privacy = 0.3
    elif doc.privacy.pii_count > 0:
        score_privacy = 0.7
    else:
        score_privacy = 1
    
    # 维度4：Compliance (copyleft许可证=0.2, 正常=1)
    score_compliance = 0.2 if any(h.is_copyleft for h in doc.compliance_hits) else 1
    
    # 维度5：Text Scan (命中>20=0.2, >5=0.6, 正常=1)
    hit_count = len(doc.keyword_hits) + len(doc.regex_hits)
    if hit_count > 20:
        score_text = 0.2
    elif hit_count > 5:
        score_text = 0.6
    else:
        score_text = 1
    
    # 最终决策：取最小分数
    min_score = min(score_secrets, score_safety, score_privacy, score_compliance, score_text)
    
    if min_score <= 0:
        decision = Decision.REJECT       # 严重风险
    elif min_score <= 0.3:
        decision = Decision.QUARANTINE   # 高风险
    elif min_score <= 0.6:
        decision = Decision.REVIEW       # 中等风险
    else:
        decision = Decision.ALLOW        # 低风险
```

**五维评分映射表：**

| 维度 | 0分 | 0.3分 | 0.5分 | 0.7分 | 1分 |
|------|-----|-------|-------|-------|-----|
| **Secrets** | 发现Secrets | — | — | — | 无Secrets |
| **Safety** | UNSAFE内容 | — | CONTROVERSIAL内容 | — | SAFE内容 |
| **Privacy** | PII>5个 | — | — | PII>0个 | 无PII |
| **Compliance** | Copyleft许可 | — | — | — | 正常许可 |
| **Text Scan** | 命中>20条 | — | — | 命中>5条 | 命中≤5条 |

**产物示例：**
```json
{
  "pipeline_run_id": "run_abc123",
  "overall_decision": "REVIEW",
  "reason_codes": ["ALERT_PII_DETECTED", "ALERT_KEYWORD_HIT"],
  "document_decisions": [
    {
      "doc_id": "doc_1",
      "decision": "REJECT",
      "reasons": ["Secret detected in source"],
      "scores": {
        "secrets": 0,
        "safety": 1,
        "privacy": 0.7,
        "compliance": 1,
        "text_scan": 0.6
      }
    }
  ]
}
```

### 1.3 关键输出产物说明

**JSONL文件格式（可流式处理）：**
```jsonl
{"source_id": "file_1", "file_path": "/data/file.txt", "file_hash": "abc123...", ...}
{"source_id": "file_2", "file_path": "/data/file.pdf", "file_hash": "def456...", ...}
```

**最终决策产物（decision.json）：**
```json
{
  "pipeline_run_id": "a1b2c3d4...",
  "overall_decision": "REJECT",
  "reason_codes": ["ALERT_UNSAFE", "ALERT_SECRET_DETECTED"],
  "evidence_summary": {
    "total_documents": 50,
    "secrets_found": 3,
    "pii_entities": 12,
    "policy_violations": 5
  }
}
```

---

## 2. AUDIO 模态详细流程

### 2.1 架构流程图

```
Input Audio Files
    ↓
[A] Source Intake
    ├─ 遍历文件路径
    ├─ 计算 SHA-256（可选，音频文件较大）
    ├─ 检测 MIME 类型
    └─ 生成 SourceRecord
    ↓
[B1] Source Classification
    ├─ 基于扩展名/MIME 分类
    └─ 结果: audio / archive / repo / mixed
    ↓
    ├─→ [B2a] TruffleHog Scan (∥并行)
    │   ├─ 执行CLI: trufflehog filesystem {path}
    │   └─ 输出: raw_secret_hits.jsonl
    │
    └─→ [B2b] ScanCode Scan (∥并行)
        ├─ 执行CLI: scancode {path}
        └─ 输出: source_compliance.jsonl
    ↓
[C0] Audio Normalization
    ├─ FFmpeg 处理
    │  ├─ 格式转换 → WAV PCM 16kHz
    │  ├─ 音量规范化
    │  └─ 非破坏性处理（原始文件不动）
    ├─ Fallback: 无FFmpeg则复制源文件
    └─ 输出: normalized_audio_manifest.jsonl (含normalized_path)
    ↓
[C1] ASR Transcription (自动语音识别)
    ├─ 优先级优先级:
    │  1️⃣ Qwen3-ASR (Qwen/Qwen3-ASR-0.6B)
    │      ├─ 使用 transformers pipeline
    │      ├─ 返回 chunks + timestamps
    │      └─ 支持语言检测
    │  2️⃣ faster-whisper (OpenAI Whisper 快速版)
    │      ├─ CPU 友好
    │      └─ 返回 segments
    │  3️⃣ Sidecar transcript
    │      ├─ 寻找同名 .jsonl 或 _transcript.jsonl
    │      └─ 用户预先提供的转写
    │  4️⃣ Fallback: 占位文本
    ├─ 处理结果中的 segment 对应音频时区间
    └─ 输出: asr_segments.jsonl (ASRSegment[])
    ↓
[C1b] Diarization (说话人分离)
    ├─ 优先 pyannote/speaker-diarization-3.1
    │  ├─ 识别 Speaker A / Speaker B / ... 
    │  └─ 返回 speakerN_segments (时间跨度 + 说话人ID)
    ├─ Fallback: 假设单说话人
    └─ 输出: speaker_segments.jsonl (SpeakerSegment[])
    ↓
[C1c] Alignment (时间对齐) — 当前 pass-through
    ├─ 预留接口，当前直接透传
    ├─ 将来可补强: 强制对齐 ASR 和 diarization 的时间
    └─ 输出: aligned_segments.jsonl (AlignedSegment[])
    ↓
[C2] Transcript Build
    ├─ 合并 ASR segments + Speaker segments + Aligned segments
    ├─ 按时间排序
    ├─ 生成统一的 TranscriptUnit (speaker_id, start_time, end_time, text)
    └─ 输出: transcript_units.jsonl (TranscriptUnit[])
    ↓
[D] Dedup
    ├─ 精确去重 (SHA-256 on text)
    ├─ 近似去重 (Jaccard similarity 基于 token，阈值可配)
    └─ 输出: deduped_transcript_units.jsonl, dedup_map.jsonl
    ↓
    ├─→ [E1a] Keyword Scan (∥并行)
    │   ├─ 加载 config/keywords.txt
    │   ├─ FlashText2 匹配，上下文提取
    │   └─ 输出: keyword_hits.jsonl
    │
    └─→ [E1b] Regex Scan (∥并行)
        ├─ 加载 config/patterns.yaml
        ├─ 模式匹配，记录匹配位置和上下文
        └─ 输出: regex_hits.jsonl
    ↓
[F] Privacy Detection
    ├─ Presidio 检测 PII 实体
    ├─ 生成 redaction_spans (start_ms, end_ms, entity_type)
    ├─ Fallback: 内置正则 PII 检测
    └─ 输出: privacy_checked.jsonl + redaction_spans.jsonl
    ↓
[G] Safety Moderation
    ├─ Qwen3Guard 分类 (Safe/Controversial/Unsafe)
    ├─ Fallback: 基于关键词
    └─ 输出: safety_checked.jsonl
    ↓
[H] Evidence Aggregation
    ├─ 聚合所有证据
    ├─ 生成 summary
    └─ 输出: evidence_bundle.json
    ↓
[I] Policy Decision
    ├─ OPA 决策 或 本地规则引擎
    ├─ 最终: Allow / Review / Reject
    └─ 输出: decision.json
    ↓
[K] Audio Redaction
    ├─ 根据 redaction_spans 生成脱敏音频
    ├─ FFmpeg 处理:
    │  ├─ silence (静音处理)
    │  ├─ beep (蜂鸣音覆盖)
    │  └─ copy (复制不处理)
    ├─ 输出格式: WAV / MP3 （根据配置）
    └─ 输出: redacted_audio_manifest.jsonl (redacted_path)
    ↓
[L] Release Package
    ├─ 汇总最终产物
    ├─ 包含: transcript, evidence, decision, redacted_audio_path
    └─ 输出: release_package.json

```

### 2.2 各步骤详解

| 步骤 | 工具/库 | 输入 | 处理逻辑 | 输出文件 | 备注 |
|------|--------|------|--------|--------|------|
| A | 标准库 | 文件路径 | 遍历，计算哈希 | source_registry.jsonl | SourceRecord[] |
| B1 | 内置 | SourceRecord[] | MIME检测 + 扩展名 | source_profile.jsonl | SourceProfile[] |
| B2a | TruffleHog CLI | SourceRecord[] | `trufflehog filesystem {path}` | raw_secret_hits.jsonl | 并行，可降级 |
| B2b | ScanCode CLI | SourceProfile[] | `scancode {path}` | source_compliance.jsonl | 并行，可降级 |
| C0 | FFmpeg | SourceProfile[] | 格式转换 16kHz WAV（`ffmpeg -i {input} -ar 16000 {output}`) | normalized_audio_manifest.jsonl | 可降级为复制 |
| C1 | Qwen3-ASR / faster-whisper | normalized_audio | AutoModelForSpeechSeq2Seq推理 + return_timestamps | asr_segments.jsonl | 三级降级链 |
| C1b | pyannote | normalized_audio | 说话人分离（pyannote.audio) | speaker_segments.jsonl | 可降级为单说话人 |
| C1c | 内置 | asr_segments + speaker_segments | 当前pass-through | aligned_segments.jsonl | 预留扩展 |
| C2 | 内置 | 三类segments | 合并+按时间排序 | transcript_units.jsonl | TranscriptUnit[] |
| D | datasketch | TranscriptUnit[] | SHA-256精确+Jaccard Token相似度(阈值可配) | deduped_transcript_units.jsonl | 可降级 |
| E1a | FlashText2 | TranscriptUnit[] | keyword匹配 | keyword_hits.jsonl | 并行 |
| E1b | re | TranscriptUnit[] | regex匹配 | regex_hits.jsonl | 并行 |
| F | Presidio | TranscriptUnit[] | PII检测，生成脱敏跨度 | privacy_checked.jsonl + redaction_spans.jsonl | 可降级 |
| G | Qwen3Guard | PrivacyResult[] | 三级安全分类 | safety_checked.jsonl | 可降级 |
| H | 内置 | 各步骤输出 | 证据聚合 | evidence_bundle.json | EvidenceBundle |
| I | OPA / 本地规则 | EvidenceBundle | 策略决策 | decision.json | PolicyDecision |
| K | FFmpeg | redaction_spans.jsonl | 音频切割+处理（消音/蜂鸣) | redacted_audio_manifest.jsonl | 可降级 |
| L | 内置 | 前序所有输出 | 打包整理 | release_package.json | 最终交付物 |

#### 2.2.1 C1 ASR（语音识别）优先级链

**三级优先级降级机制：**
```python
# 第1优先级：Qwen3-ASR（千问0.6B模型）
try:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        settings.qwen_asr_model,  # 默认 "Qwen/Qwen3-ASR-0.6B"
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(...)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps=True
    )
    output = asr(record.normalized_path)
    chunks = output.get("chunks") or []
    # 返回 ASRSegment[] (start_time, end_time, text, confidence)
except:
    # 第2优先级：faster-whisper（更轻量）
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel("base")
        segments, info = model.transcribe(record.normalized_path)
        # 返回 ASRSegment[]
    except:
        # 第3优先级：Sidecar 转写文件
        try:
            candidates = [
                Path(record.original_path).with_suffix(".transcript.jsonl"),
                Path(record.original_path).with_name("sample_transcript.jsonl"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    segments = load_jsonl(candidate)
                    return segments
        except:
            # 第4优先级：占位文本
            return [ASRSegment(
                text="[ASR unavailable]",
                start_time=0.0,
                end_time=record.duration_seconds,
                engine_name="fallback"
            )]
```

#### 2.2.2 K 音频脱敏详解

**三种渲染策略（FFmpeg命令）：**

```bash
# 策略1：Silence（消音）- 将指定时间段音量设为0
ffmpeg -y \
  -i input.wav \
  -af "volume=enable='between(t,1.5,3.0):between(t,5.2,6.8)':volume=0" \
  output_silence.wav

# 策略2：Beep（蜂鸣覆盖）- 生成同频蜂鸣音覆盖
ffmpeg -y \
  -i input.wav \
  -filter_complex "[0:a]volume=enable='between(t,1.5,3.0)':volume=0[main];
                   sine=f=1000:sample_rate=16000:d=12,
                   aselect='between(t,1.5,3.0)',
                   asetpts=N/SR/TB[beep];
                   [main][beep]amix=inputs=2:normalize=0[out]" \
  -map "[out]" \
  output_beep.wav

# 策略3：Copy（复制）- 不做任何処理
cp input.wav output_copy.wav
```

**实现逻辑：**
```python
redaction_dir = output_dir / "redacted_audio"
spans_by_source = defaultdict(list)
for span in redaction_spans:  # [RedactionSpan(start_time, end_time, entity_type), ...]
    spans_by_source[span.source_id].append(span)

for record in normalized_records:
    source_spans = sorted(spans_by_source[record.source_id], key=lambda x: x.start_time)
    
    if not source_spans:
        # 无脱敏发现，直接复制
        shutil.copy2(record.normalized_path, target_path)
        strategy = RenderStrategy.COPY
    else:
        if settings.redaction_strategy == "silence":
            # 构造 FFmpeg 音量过滤器
            filters = [f"volume=enable='between(t,{s.start_time},{s.end_time})':volume=0" 
                       for s in source_spans]
            result = run_command([
                settings.ffmpeg_bin, "-y", "-i", record.normalized_path,
                "-af", ",".join(filters),
                target_path
            ], timeout=600)
        elif settings.redaction_strategy == "beep":
            # 构造复杂的 filter_complex（蜂鸣覆盖）
            duration = record.duration_seconds
            expr = "+".join([f"between(t,{s.start_time},{s.end_time})" for s in source_spans])
            filter_complex = (
                f"[0:a]volume=enable='{expr}':volume=0[main];"
                f"sine=f={settings.beep_frequency}:sample_rate=16000:d={duration},"
                f"aselect='{expr}',asetpts=N/SR/TB[beep];"
                "[main][beep]amix=inputs=2:normalize=0[out]"
            )
            result = run_command([...filter_complex...], timeout=600)
```

**产物示例：**
```jsonl
{"source_id": "audio_1", "original_path": "input.wav", "redacted_path": "output_silence.wav", "strategy": "silence"}
{"source_id": "audio_2", "original_path": "input2.wav", "redacted_path": "output2_beep.wav", "strategy": "beep"}
```

### 2.3 关键输出产物

**redaction_spans.jsonl 示例：**
```jsonl
{"source_id": "audio_1", "start_ms": 1500, "end_ms": 3000, "entity_type": "PHONE_NUMBER", "confidence": 0.95}
{"source_id": "audio_1", "start_ms": 5200, "end_ms": 6800, "entity_type": "EMAIL_ADDRESS", "confidence": 0.88}
```

**release_package.json 示例：**
```json
{
  "run_id": "abc123...",
  "source_info": {...},
  "transcript_summary": "Speaker summary with PII counts",
  "decision": {"overall": "REVIEW", "reason_codes": [...]},
  "redacted_audio_path": "/outputs/run_id/redacted.wav",
  "evidence_bundle": {...}
}
```

---

## 3. PICTURE 模态详细流程

### 3.1 架构流程图（三条并行链路）

```
Input Image (PNG/JPG/PDF/GIF/WEBP...)
    ↓
Router (Heuristic Classification)
    ├─ 分析图像特征 (文本量、人脸检测、自然特征)
    └─ 决策 → Document / Natural / Mixed
    ↓
Preprocess
    ├─ EXIF 读取+旋转矫正
    ├─ PDF 多页拆分 (逐页处理)
    ├─ 尺寸规整 (resample if too large)
    └─ 输出: 规范化图像
    ↓
┌──────────────────────────────────────────────────────────────┐
│ 三条并行路由链                                                  │
└──────────────────────────────────────────────────────────────┘
    ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【Document Chain】文档图像链路
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
[OCR + Layout Analysis]
    ├─ 优先: PaddleOCR-VL / MinerU / Surya  (一级模型)
    ├─ Mock: 占位符 (开发测试)
    └─ 输出: 文本块 (text_blocks[]) + 版面区域 (layout_regions[])
    ↓
[Text PII Detection]
    ├─ OCR 提取文本
    ├─ Presidio 检测 PII
    ├─ 文本跨度 → 图像坐标映射
    └─ 输出: PII findings (含bbox)
    ↓
[Vision Detect]
    ├─ YOLO26 / Grounding DINO
    ├─ 检测: 人脸、二维码、印章、工牌等
    └─ 输出: Vision findings (bbox + class)
    ↓
[Segmentation Refine (可选)]
    ├─ SAM2 精细分割
    ├─ 从边界框 → 像素级 mask
    └─ 用于高精度脱敏
    ↓
[Redaction]
    ├─ 遮挡模式选择:
    │  ├─ 文字 → black_box (黑色方块)
    │  ├─ 人脸 → gaussian_blur (高斯模糊)
    │  ├─ 二维码 → black_box
    │  └─ 印章 → solid_fill
    ├─ OpenCV / Pillow 渲染
    └─ 输出: compliant_image.png (带脱敏)
    ↓
[Policy Evaluate]
    ├─ 检查 findings 严重程度
    ├─ 决策: pass_raw (无脱敏) / pass_redacted (脱敏后) / drop (丢弃)
    └─ 输出: 决策 + reason_codes


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【Natural Chain】自然图像链路
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
[Safety Moderation]
    ├─ ShieldGemma2 (当启用时)
    ├─ 检测: 色情、暴力、仇恨
    ├─ Mock: 占位符
    └─ 输出: SafetyCategory 结果
    ↓
[Vision Detect]
    ├─ YOLO26 / Grounding DINO
    ├─ 检测: 人脸、敏感物品
    └─ 输出: Vision findings (bbox)
    ↓
[Segmentation Refine]
    ├─ SAM2 精细分割
    └─ 生成 mask
    ↓
[Redaction]
    ├─ 应用脱敏
    └─ 输出: compliant_image.png
    ↓
[Policy Evaluate]
    └─ 输出: pass_raw / pass_redacted / drop


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【Mixed Chain】混合截图链路 (双链并行)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ├─ [OCR ∥ Safety] 并行执行
    ├─ [Text PII ∥ Vision] 并行执行
    ├─ Merge findings
    ├─ [Segmentation + Redaction]
    └─ [Policy Evaluate]


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
[Final Output]
    ├─ PictureReport: job_id, findings[], decision
    ├─ Findings JSONL: 每行一条发现
    ├─ Compliant Image: 脱敏后的图像
    └─ Audit Trail: 完整处理日志
```

### 3.2 各步骤详解

| 组件 | 实现方式 | 功能 | 输入 | 输出 | 备注 |
|------|---------|------|------|------|------|
| Router | HeuristicRouter | 分类路由 | 图像 | Document/Natural/Mixed | 启发式规则 |
| Preprocessor | DefaultPreprocessor | EXIF处理、PDF拆页 | 图像路径 | 规范化图像路径 | 原图不动 |
| OCRLayoutProvider | PaddleOCR-VL / Mock | 文本提取+版面 | 图像 | text_blocks[], layout_regions[] | 多模型支持 |
| PIIDetector | Presidio / Mock | PII检测 | OCR文本 | PII findings[] | 可映射到bbox |
| SafetyModerator | ShieldGemma2 / Mock | 安全分类 | 图像 | SafetyCategory | image-classification任务 |
| VisionDetector | YOLO26 / Grounding DINO / Mock | 目标检测 | 图像 | Vision findings[] (bbox) | 支持多类型 |
| SegmentationProvider | SAM2 / Mock | 像素级分割 | 图像 + bbox | 分割 mask | 精度优化 |
| Redactor | OpenCVRedactor | 脱敏渲染 | 图像 + findings | 脱敏图像 | 支持多种模式 |
| PolicyEngine | ConfigurablePolicyEngine | 策略决策 | findings[] + policy config | pass_raw / pass_redacted / drop | 基于rules |
| StorageBackend | LocalFileStorageBackend / S3 | 产物持久化 | 图像 + JSON | 存储位置 | 可切换存储 |

#### 3.2.1 Provider 工厂模式详解

**核心设计：根据环境变量动态选择 Provider**

```python
# picture/application/use_cases.py - 标准工厂模式

def create_ocr_provider(settings: PictureSettings | None = None) -> OCRLayoutProvider:
    """Create an OCR provider based on settings."""
    settings = settings or get_settings()
    provider_name = settings.ocr_provider.lower()
    
    # 环境变量 PICTURE_OCR_PROVIDER 可选值：
    # - "paddleocr" → PaddleOCRVLProvider(lang, use_gpu)
    # - "mineru" → MinerUProvider()
    # - "surya" → SuryaProvider()
    # - "mock" 或其他 → MockOCRLayoutProvider() [默认]

    if provider_name == "paddleocr":
        from picture.providers.ocr.paddleocr_vl import PaddleOCRVLProvider
        return PaddleOCRVLProvider(
            lang=settings.paddleocr_lang,
            use_gpu=settings.paddleocr_use_gpu,
        )
    elif provider_name == "mineru":
        from picture.providers.ocr.mineru import MinerUProvider
        return MinerUProvider()
    elif provider_name == "surya":
        from picture.providers.ocr.surya import SuryaProvider
        return SuryaProvider()
    else:
        from picture.providers.ocr.mock import MockOCRLayoutProvider
        return MockOCRLayoutProvider()

# 类似的工厂方法还有：
# - create_pii_detector: Presidio / Mock
# - create_safety_moderator: ShieldGemma2 / Mock
# - create_vision_detector: YOLO26 / Grounding DINO / Mock
# - create_segmentation_provider: SAM2 / Mock
# - create_storage: LocalFileStorageBackend / S3StorageBackend
```

**环境变量配置示例：**
```bash
export PICTURE_OCR_PROVIDER=paddleocr        # ocr provider
export PICTURE_PII_PROVIDER=presidio         # pii provider
export PICTURE_SAFETY_PROVIDER=mock          # safety provider
export PICTURE_VISION_PROVIDER=yolo26        # vision provider
export PICTURE_SEGMENTATION_PROVIDER=sam2    # segmentation provider
export PICTURE_STORAGE_BACKEND=local         # storage backend
export PICTURE_WORK_DIR=./compliance_output_picture
```

#### 3.2.2 三条并行链路的具体实现

**文档链 (Document Route)：**
```python
# picture/application/orchestrator.py - _process_document_chain()
def _process_document_chain(self, image_path: str) -> list[PictureFinding]:
    # 1. OCR + Layout Analysis
    ocr_result = run_ocr_layout(self._ocr, image_path)
    # 返回: OCRLayoutResult(full_text, text_blocks[bbox,text,...], layout_regions[...])
    
    # 2. Text PII Detection（文本→坐标映射）
    text_pii_findings = run_text_pii_detection(self._pii, ocr_result)
    # 对每个 PII 实体，尝试通过 OCR block 映射回 bbox
    
    # 3. Vision Detection（独立）
    vision_findings = run_vision_detection(self._vision, image_path)
    # YOLO/DINO 检测人脸、二维码等
    
    # 4. 合并 findings
    all_findings = text_pii_findings + vision_findings
    
    # 5. Segmentation Refine（可选）
    refined_findings = run_segmentation_refinement(
        self._segmentation,
        image_path,
        all_findings
    )
    
    return refined_findings
```

**自然链 (Natural Route)：**
```python
def _process_natural_chain(self, image_path: str) -> list[PictureFinding]:
    # 1. Safety Moderation
    safety_result = run_safety_moderation(self._safety, image_path)
    # ShieldGemma2 → [{"label": "sexually_explicit", "score": 0.95}, ...]
    
    # 2. Vision Detection
    vision_findings = run_vision_detection(self._vision, image_path)
    
    # 3. Segmentation + Redaction
    refined = run_segmentation_refinement(
        self._segmentation,
        image_path,
        vision_findings
    )
    
    return refined
```

**混合链 (Mixed Route)：**
```python
def _process_mixed_chain(self, image_path: str) -> list[PictureFinding]:
    # 并行执行两条子链
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Phase 1: OCR ∥ Safety
        ocr_future = executor.submit(run_ocr_layout, self._ocr, image_path)
        safety_future = executor.submit(run_safety_moderation, self._safety, image_path)
        
        ocr_result = ocr_future.result()
        safety_result = safety_future.result()
        
        # Phase 2: Text PII ∥ Vision
        pii_future = executor.submit(run_text_pii_detection, self._pii, ocr_result)
        vision_future = executor.submit(run_vision_detection, self._vision, image_path)
        
        text_findings = pii_future.result()
        vision_findings = vision_future.result()
    
    # 合并所有 findings
    all_findings = text_findings + vision_findings
    
    # Segmentation + Redaction
    refined = run_segmentation_refinement(self._segmentation, image_path, all_findings)
    
    return refined
```

#### 3.2.3 脱敏渲染的多种模式

**OpenCV/Pillow 脱敏实现：**
```python
# picture/providers/redaction/opencv_redactor.py

def redact_finding(self, image, finding: PictureFinding) -> Image:
    """Apply redaction to an image region based on redaction mode."""
    
    if not finding.region or not finding.region.bbox:
        return image
    
    x1, y1, x2, y2 = finding.region.bbox  # BBox坐标
    roi = image[y1:y2, x1:x2]              # 感兴趣区域
    
    mode = _get_redaction_mode(finding.category)
    
    if mode == "black_box":
        # 填充黑色方块
        roi[:, :] = [0, 0, 0]
    elif mode == "gaussian_blur":
        # 高斯模糊（常用于人脸）
        roi_blurred = cv2.GaussianBlur(roi, (51, 51), 0)
        image[y1:y2, x1:x2] = roi_blurred
    elif mode == "pixelate":
        # 像素化（马赛克）
        factor = 20
        small = cv2.resize(roi, (roi.shape[1]//factor, roi.shape[0]//factor))
        pixelated = cv2.resize(small, (roi.shape[1], roi.shape[0]), 
                               interpolation=cv2.INTER_NEAREST)
        image[y1:y2, x1:x2] = pixelated
    elif mode == "solid_fill":
        # 纯色填充
        roi[:, :] = [200, 200, 200]  # 灰色
    
    return image
```

**脱敏模式配置示例：**
```yaml
# picture/configs/default_cn_enterprise.yaml
redaction_modes:
  text:               black_box      # 文字→黑色方块
  face:               gaussian_blur  # 人脸→高斯模糊
  qr_code:            black_box      # 二维码→黑色方块
  barcode:            black_box      # 条形码→黑色方块
  signature:          solid_fill     # 签名→灰色填充
  stamp:              solid_fill     # 印章→灰色填充
  default:            black_box      # 其他→黑色方块
```

### 3.3 关键输出产物

**findings.jsonl 示例：**
```jsonl
{"finding_id": "f001", "finding_type": "pii_text", "category": "phone_number", "region": {"bbox": [100, 200, 300, 250]}, "confidence": 0.95}
{"finding_id": "f002", "finding_type": "vision", "category": "face", "region": {"bbox": [50, 50, 400, 500]}, "confidence": 0.98}
```

**report.json 示例：**
```json
{
  "job_id": "job_123",
  "status": "COMPLETED",
  "route": "document",
  "decision": "pass_redacted",
  "findings": [
    {"type": "pii", "count": 3, "categories": ["phone", "id"]},
    {"type": "vision", "count": 2, "categories": ["face", "qr_code"]}
  ],
  "reason_codes": ["REDACTION_REQUIRED"],
  "compliant_image_uri": "s3://bucket/compliant.png"
}
```

---

## 4. VIDEO 模态详细流程

### 4.1 架构流程图

```
Input: Video File / GIF / Frame Sequence
    ↓
[Media Detection]
    ├─ FFmpeg/ffprobe 探测媒体信息
    ├─ 或 Pillow 打开动画图像
    └─ 提取: fps, duration, codec, has_audio
    ↓
[Prepare Input Source]
    ├─ 复制或引用输入到工作目录
    ├─ 建立清晰的输入/产物边界
    └─ 输出: prepared_source_path
    ↓
[Load Sequence] - 抽帧步骤
    ├─ 三种来源处理:
    │  
    │  1️⃣ 动画图像 (GIF/WebP/APNG)
    │     ├─ Pillow 逐帧迭代
    │     ├─ 保留每帧时长 (duration_ms)
    │     └─ Frames[]
    │  
    │  2️⃣ 帧目录 (frame_0001.png, frame_0002.png, ...)
    │     ├─ 按序加载
    │     ├─ 附加默认帧时长 (default_frame_duration_ms)
    │     └─ Frames[]
    │  
    │  3️⃣ 正式视频容器 (MP4/MOV/MKV/AVI/M4V/WEBM)
    │     ├─ ffprobe 获取元数据
    │     ├─ ffmpeg select=not(mod(n, frame_stride)) 采样
    │     ├─ 为每帧计算时间戳 (pts_ms)
    │     ├─ 计算帧时长 (duration_ms)
    │     └─ Frames[] (含时间轴信息)
    │
    ├─ 支持抽帧策略
    │  ├─ frame_stride: 每隔 N 帧取 1 帧 (降低计算)
    │  ├─ max_frames: 限制总抽帧数
    │  └─ default_frame_duration_ms: 默认帧间隔
    │
    └─ 输出: Sequence (frames[], source_kind, fps, duration_ms, has_native_audio)
    ↓
[Analyze Frames] - 逐帧 Picture 处理
    ├─ 并行处理多个帧 (max_workers)
    ├─ 调用 PictureComplianceOrchestrator 对每一帧
    │  ├─ 路由 (document / natural / mixed)
    │  ├─ OCR / PII / Safety / Vision / Segmentation 等
    │  ├─ 脱敏
    │  └─ 策略决策
    ├─ 输出: FrameJob[] (每帧的处理结果)
    └─ 保留: provider_versions, findings, decision
    ↓
[Derive Segments] - 时序聚合
    ├─ 把逐帧结果聚合为视频级片段
    ├─ 相邻帧 decision 相同 → 合并为一个片段
    ├─ 片段跨度: start_ms, end_ms, decision, reason_codes
    └─ 输出: Segment[] (在时间轴上连续)
    ↓
[Build Video Findings] - 映射发现
    ├─ 把每帧的 findings 映射到时间轴
    ├─ 每个 finding 关联 start_ms, end_ms
    ├─ 这样UI可以在进度条上显示风险区间
    └─ 输出: VideoFinding[]
    ↓
[Audio Track Processing] - 音轨处理（可选）
    ├─ 检查是否需要处理音轨
    ├─ 路径 1: Sidecar 音轨
    │  ├─ 检查 audio.wav / audio.mp3
    │  └─ 或 options.sidecar_audio_path
    ├─ 路径 2: 原生音轨 (仅 video_container)
    │  ├─ ffmpeg 抽取音轨 → WAV
    │  └─ 生成 native_audio_extracted = true
    │
    ├─ [如果有音轨]
    │  ├─ 调用 AudioCompliancePipeline
    │  ├─ 获得 audio_decision, evidence_bundle, redacted_audio_path
    │  └─ 折叠到视频决策: REVIEW/REJECT 升级视频风险
    │
    └─ 输出: audio_decision (如适用)
    ↓
[Aggregate Policy] - 视频级策略决策
    ├─ 合并 video findings 和 audio findings （如有）
    ├─ 最严格原则:
    │  ├─ 视频 decision = DROP → 最终 DROP
    │  ├─ 音频 decision = REJECT → 视频升级 DROP
    │  ├─ 音频 decision = REVIEW → 视频升级 REVIEW
    │  └─ 其他 → Allow
    ├─ 生成最终决策 + reason_codes
    └─ 输出: VideoDecision
    ↓
[Render Sequence Outputs] - 脱敏视频渲染
    ├─ 对应三种输入处理:
    │  
    │  1️⃣ 动画图像 → 脱敏帧 + Pillow 重新编码
    │     ├─ 接受脱敏帧列表
    │     ├─ 重建 GIF / WebP
    │     └─ 输出: compliant.gif / compliant.webp
    │  
    │  2️⃣ 帧目录 → 输出脱敏帧目录 （copy）
    │     ├─ 复制脱敏帧到 output_dir/frames
    │     └─ 输出: 帧目录路径
    │  
    │  3️⃣ 视频容器 → FFmpeg 重新 mux
    │     ├─ FFmpeg 合成脱敏帧序列 + 音轨
    │     ├─ 支持脱敏音频 (如 audio_redaction 有输出)
    │     ├─ 输出编码: libx264 (H.264) 或配置编码器
    │     └─ 输出: compliant_video.mp4 (或其他格式)
    │
    └─ 输出: VideoAsset (compliant_uri, metadata)
    ↓
[Output Files]
    ├─ frame_manifest.jsonl
    │  ├─ 逐行: frame_id, frame_index, pts_ms, route, decision
    │  └─ 用于UI调试和细粒度审计
    │
    ├─ segment_manifest.jsonl
    │  ├─ 时间段级别的决策
    │  └─ start_ms, end_ms, decision, reason_codes
    │
    ├─ video_findings.jsonl
    │  ├─ 每行一条 finding，带时间跨度
    │  ├─ 用于进度条和悬停提示
    │  └─ type, category, bbox, start_ms, end_ms
    │
    ├─ evidence_bundle.json
    │  └─ 汇总所有证据 (帧级 + 音轨级)
    │
    ├─ decision.json
    │  ├─ 最终决策: Allow / Review / Reject
    │  ├─ reason_codes
    │  └─ 脱敏资源位置
    │
    ├─ [compliant video]
    │  ├─ compliant_video.mp4 (或 .gif / 帧目录)
    │  ├─ 完全可交付
    │  └─ 包含脱敏画面 + 脱敏音轨 (如有)
    │
    └─ [可选产物]
        ├─ redacted_audio.wav (如有音轨处理)
        └─ audio_evidence_bundle.json
```

### 4.2 步骤详解 - 核心处理表

| 步骤 | 工具 | 输入 | 处理逻辑 | 输出 | 备注 |
|------|------|------|--------|------|------|
| Media Detection | FFmpeg/Pillow | 视频路径 | 探测元数据 | 视频信息 (fps, duration, codec) | 决定后续处理方式 |
| Load Sequence | FFmpeg + Pillow | 视频/GIF/帧 | 1) Pillow逐帧 2) Pillow帧目录 3) FFmpeg抽帧 | Sequence (frames[]) | 支持frame_stride采样 |
| Analyze Frames | Picture引擎 | Frames[] | 并行调用PictureOrchestrator | FrameJob[] (findings, decision) | 逐帧处理，可并行 |
| Derive Segments | 内置 | FrameJob[] | 聚合相邻同决策帧 | Segment[] (时间跨度级) | 压缩产物 |
| Build Video Findings | 内置 | FrameJob[] | 映射 findings 到时间轴 | VideoFinding[] (含start_ms, end_ms) | 用于UI进度条 |
| Audio Sidecar | Audio引擎 | 音頻文件 | 调用 AudioCompliancePipeline | audio_decision, decisions_json | 可选，可降级 |
| Aggregate Policy | 内置规则 | Video + Audio 决策 | 最严格原则合并 | 最终 VideoDecision | 视频优先级更高 |
| Render Outputs | FFmpeg / Pillow | 脱敏帧 + 音轨 | 1) 重编码GIF 2) 复制帧 3) mux视频 | 脱敏视频文件 | 物理级输出 |

### 4.2.1 各步骤的详细处理逻辑

**[1] Media Detection - 媒体类型探测**
```python
# video/pipeline.py - detect_media_type()
def detect_media_type(input_path: str) -> str:
    """判断输入的媒体类型（video_container / animated_image / frame_directory）"""
    
    # 路径检查
    if os.path.isdir(input_path):
        # 目录 → 尝试读取帧文件
        frames = glob(os.path.join(input_path, "*.png")) + glob(os.path.join(input_path, "*.jpg"))
        if frames: 
            return "frame_directory"
    
    # 文件检查
    ext = input_path.lower().split('.')[-1]
    
    if ext in ['gif', 'webp', 'apng']:
        return "animated_image"
    
    if ext in ['mp4', 'mov', 'mkv', 'avi', 'flv', 'wmv']:
        return "video_container"
    
    raise ValueError(f"Unsupported media type: {ext}")
```

**[2] Load Sequence - 帧序列加载**

**情况 A: Video 容器文件（MP4/MOV等）**
```bash
# FFmpeg 抽帧命令（自适应采样）
ffmpeg -y \
  -i input.mp4 \
  -vf "select='not(mod(n\,{frame_stride}))',scale={width}:{height}" \
  -vsync 0 \
  frame_%05d.png

# frame_stride=1: 每帧提取 (1fps × 60sec = 60帧)
# frame_stride=5: 每5帧提取 (0.2fps × 60sec = 12帧)
# frame_stride=25: 每25帧提取 (1/5fps × 60sec = 2.4帧，接近截图)
```

**情况 B: 动态图像（GIF/WebP/APNG）**
```python
# 使用 Pillow 逐帧迭代
from PIL import Image
import io

image = Image.open("input.gif")
frames = []
for frame_index in range(image.n_frames):
    image.seek(frame_index)
    # 获取帧的时间戳（毫秒）
    duration_ms = image.info.get('duration', 100)  # 默认100ms
    pts_ms = sum([image_list[i].info.get('duration', 100) for i in range(frame_index)])
    
    frames.append({
        'frame_id': f'f_{frame_index:05d}',
        'frame_index': frame_index,
        'image_data': image.copy().convert('RGB'),
        'pts_ms': pts_ms,
        'duration_ms': duration_ms
    })

return Sequence(frames=frames, source_kind='animated_image', fps=1000/avg_duration_ms)
```

**情况 C: 帧目录（预抽帧的 PNG/JPG 序列）**
```python
# 按字典序读取帧文件
frame_files = sorted(glob(os.path.join(frame_dir, "*.png")) + 
                     glob(os.path.join(frame_dir, "*.jpg")))

frames = []
for idx, frame_file in enumerate(frame_files):
    image = Image.open(frame_file)
    # 使用索引推算时间轴（假设帧率已知，如25fps）
    pts_ms = idx * (1000 / settings.assumed_fps)
    
    frames.append({
        'frame_id': f'f_{idx:05d}',
        'frame_index': idx,
        'image_data': image.convert('RGB'),
        'pts_ms': pts_ms,
        'duration_ms': 1000 / settings.assumed_fps
    })

return Sequence(frames=frames, source_kind='frame_directory', fps=settings.assumed_fps)
```

**[3] Analyze Frames - 并行逐帧处理**
```python
# video/pipeline.py - 伪代码
def analyze_frames(sequence: Sequence, max_workers: int = 8):
    """并行处理每一帧，调用 PictureComplianceOrchestrator"""
    
    frame_jobs = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        for frame in sequence.frames:
            # 提交帧处理任务
            future = executor.submit(
                self.picture_orchestrator.process,
                image_uri=frame.image_data,  # 或临时保存为文件再传路径
                run_id=f"{self.run_id}_{frame.frame_id}"
            )
            futures[future] = frame
        
        # 收集结果
        for future in as_completed(futures):
            frame = futures[future]
            try:
                result = future.result(timeout=60)  # 单帧处理超时60秒
                
                frame_jobs.append(FrameJob(
                    frame_id=frame.frame_id,
                    frame_index=frame.frame_index,
                    pts_ms=frame.pts_ms,
                    duration_ms=frame.duration_ms,
                    route=result['route'],  # document / natural / mixed
                    decision=result['decision'],  # allow / review / reject
                    findings=result['findings'],  # find[]
                    provider_versions=result['provider_versions'],  # 用于审计
                    processing_time_ms=result['processing_time_ms']
                ))
            except Exception as e:
                # 单帧失败 → 记录为 REVIEW 决策
                frame_jobs.append(FrameJob(
                    frame_id=frame.frame_id,
                    frame_index=frame.frame_index,
                    pts_ms=frame.pts_ms,
                    decision='review',  # 降级到 REVIEW
                    reason='Frame processing error: ' + str(e)
                ))
    
    return frame_jobs
```

**[4] Derive Segments - 时间轴聚合**
```python
# video/application/services.py - derive_segments() 核心逻辑
def derive_segments(frame_jobs: List[FrameJob]) -> List[VideoSegment]:
    """把逐帧决策聚合为连续的时间段"""
    
    if not frame_jobs:
        return []
    
    segments = []
    current_segment = None
    
    for frame_job in sorted(frame_jobs, key=lambda x: x.frame_index):
        if current_segment is None:
            # 初始化第一个段
            current_segment = VideoSegment(
                segment_id=f"seg_{len(segments):03d}",
                decision=frame_job.decision,
                start_pts_ms=frame_job.pts_ms,
                start_frame_index=frame_job.frame_index,
                finding_ids=[],
                reason_codes=[]
            )
        elif frame_job.decision == current_segment.decision:
            # 决策相同 → 延伸当前段
            current_segment.end_frame_index = frame_job.frame_index
        else:
            # 决策改变 → 结束当前段并开始新段
            current_segment.end_pts_ms = frame_job.pts_ms - frame_job.duration_ms
            segments.append(current_segment)
            
            current_segment = VideoSegment(
                segment_id=f"seg_{len(segments):03d}",
                decision=frame_job.decision,
                start_pts_ms=frame_job.pts_ms,
                start_frame_index=frame_job.frame_index,
                finding_ids=[],
                reason_codes=[]
            )
    
    # 收尾
    if current_segment:
        current_segment.end_pts_ms = frame_jobs[-1].pts_ms + frame_jobs[-1].duration_ms
        segments.append(current_segment)
    
    return segments
```

**[5] Audio Track Processing - 音轨提取与处理**
```python
# video/pipeline.py - 伪代码
def process_audio_track(video_path: str, options: VideoOptions):
    """从视频中提取音轨，调用 AudioCompliancePipeline"""
    
    # 步骤1: 寻找音轨来源
    sidecar_audio_path = None
    
    # 尝试路径1: Sidecar 音频文件
    for ext in ['.wav', '.mp3', '.aac', '.flac']:
        candidate = video_path.replace(os.path.splitext(video_path)[1], ext)
        if os.path.exists(candidate):
            sidecar_audio_path = candidate
            break
    
    # 尝试路径2: 原生音轨（如果是视频容器）
    native_audio_path = None
    if not sidecar_audio_path and options.source_kind == 'video_container':
        # 使用 ffprobe 检查是否有音轨
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'csv=p=0',
            video_path
        ], capture_output=True, text=True)
        
        if result.stdout.strip():  # 有音轨
            # 提取音轨
            native_audio_path = f"{os.path.splitext(video_path)[0]}_extracted_audio.wav"
            subprocess.run([
                'ffmpeg', '-y', '-i', video_path,
                '-q:a', '9',  # 最高质量
                '-ac', '2',   # 立体声
                native_audio_path
            ])
    
    # 步骤2: 调用 AudioCompliancePipeline
    audio_input_path = sidecar_audio_path or native_audio_path
    if not audio_input_path:
        return None  # 无音轨
    
    audio_pipeline = AudioCompliancePipeline(settings=settings)
    audio_records = [SourceRecord(source_id=f"{options.run_id}_audio", file_path=audio_input_path)]
    
    audio_result = audio_pipeline.execute(audio_records, options)
    
    return {
        'audio_decision': audio_result.policy_decision.overall_decision,
        'evidence_bundle': audio_result.evidence_bundle,
        'redacted_audio_path': audio_result.redacted_audio_path,
        'provider_versions': audio_result.provider_versions
    }
```

**[6] Aggregate Policy - 合并视频+音频决策**
```python
# video/pipeline.py - aggregate_policy() 伪代码
def aggregate_policy(
    video_segments: List[VideoSegment],
    audio_result: Optional[dict],
    settings: VideoSettings
) -> VideoDecision:
    """合并视频帧决策 + 音频决策 → 最终视频决策"""
    
    # 计统计信息
    frame_stats = {
        'total_frames': len(frame_jobs),
        'allow_frames': len([f for f in frame_jobs if f.decision == 'allow']),
        'review_frames': len([f for f in frame_jobs if f.decision == 'review']),
        'reject_frames': len([f for f in frame_jobs if f.decision == 'reject'])
    }
    
    # 确定最严格的视频决策
    decisions = [seg.decision for seg in video_segments]
    if 'reject' in decisions:
        video_decision = 'reject'
    elif 'review' in decisions:
        video_decision = 'review'
    else:
        video_decision = 'allow'
    
    # 与音频决策合并（音频发现也升级视频决策）
    if audio_result:
        audio_decision = audio_result['audio_decision']
        if audio_decision == 'reject':
            video_decision = 'reject'
        elif audio_decision == 'review' and video_decision != 'reject':
            video_decision = 'review'
    
    reason_codes = []
    if 'reject' in decisions:
        reason_codes.append('VIDEO_REJECTED_FRAMES_FOUND')
    if 'review' in decisions:
        reason_codes.append('VIDEO_REVIEW_FRAMES_FOUND')
    if audio_result and audio_result['audio_decision'] != 'allow':
        reason_codes.append(f'AUDIO_{audio_result["audio_decision"].upper()}')
    
    return VideoDecision(
        overall_decision=video_decision,
        reason_codes=reason_codes,
        frame_stats=frame_stats,
        audio_stats={
            'decision': audio_result['audio_decision'] if audio_result else None,
            'pii_found': bool(audio_result and audio_result['evidence_bundle'].pii_findings) if audio_result else False
        }
    )
```

**[7] Render Outputs - 最终脱敏输出**
```bash
# 步骤1: 脱敏后的帧序列 + 音轨 → 重新编码为视频（或GIF）
# 输出格式根据原始输入类型决定

# 情况1: 原始是 MP4 → 输出脱敏后也是 MP4
ffmpeg -y \
  -framerate {fps} \
  -i redacted_frame_%05d.png \
  -i redacted_audio.wav \
  -c:v libx264 \
  -preset fast \
  -crf 23 \
  -c:a aac \
  -b:a 128k \
  compliant_video.mp4

# 情况2: 原始是 GIF → 输出脱敏后也是 GIF
ffmpeg -y \
  -framerate {fps} \
  -i redacted_frame_%05d.png \
  -vf "scale={width}:{height}:flags=lanczos" \
  compliant_video.gif

# 情况3: 原始是帧目录 → 输出脱敏后的帧目录
cp redacted_frames/*.png output_frame_dir/
```

### 4.3 关键输出产物

**frame_manifest.jsonl 示例：**
```jsonl
{"frame_id": "f_0001", "frame_index": 1, "pts_ms": 33, "duration_ms": 33, "route": "natural", "decision": "allow", "findings": 0}
{"frame_id": "f_0010", "frame_index": 10, "pts_ms": 330, "duration_ms": 33, "route": "natural", "decision": "review", "findings": 2}
```

**video_findings.jsonl 示例：**
```jsonl
{"finding_id": "vf001", "type": "vision", "category": "face", "bbox": [50, 50, 400, 500], "start_ms": 1000, "end_ms": 2000, "confidence": 0.97}
{"finding_id": "vf002", "type": "safety", "category": "violence", "start_ms": 3500, "end_ms": 4200}
```

**decision.json 示例：**
```json
{
  "run_id": "video_abc123...",
  "overall_decision": "REVIEW",
  "reason_codes": ["VIDEO_CONTAINS_PII", "AUDIO_CONTAINS_SPEECH"],
  "compliant_video_uri": "s3://bucket/compliant_video.mp4",
  "frame_stats": {
    "total_frames": 300,
    "allow_frames": 280,
    "review_frames": 15,
    "reject_frames": 5
  },
  "audio_stats": {
    "decision": "ALLOW",
    "redacted_audio_uri": "s3://bucket/redacted_audio.wav"
  }
}
```

### 4.3.1 VIDEO 模式的完整输出示例

**segment_manifest.jsonl 示例（完整视频分段摘要）：**
```jsonl
{"segment_id": "seg_001", "start_ms": 0, "end_ms": 2500, "start_frame": 0, "end_frame": 62, "decision": "review", "reason_codes": ["FACE_DETECTED"], "findings_count": 2}
{"segment_id": "seg_002", "start_ms": 2500, "end_ms": 8000, "start_frame": 63, "end_frame": 200, "decision": "allow", "reason_codes": [], "findings_count": 0}
{"segment_id": "seg_003", "start_ms": 8000, "end_ms": 12000, "start_frame": 201, "end_frame": 300, "decision": "reject", "reason_codes": ["PII_ID_CARD_DETECTED"], "findings_count": 3}
```

**video_decision.json 完整示例（包含所有决策信息）：**
```json
{
  "pipeline_run_id": "video_20250124_meeting_xyz",
  "source_uri": "s3://uploads/meeting_20250124.mp4",
  "source_size_bytes": 524288000,
  "source_duration_ms": 12000,
  "source_fps": 25,
  "source_resolution": "1920x1080",
  
  "processing_summary": {
    "start_time": "2025-01-24T10:15:00Z",
    "end_time": "2025-01-24T10:15:30Z",
    "total_processing_ms": 30000,
    "frames_processed": 300,
    "frame_processing_parallelism": 8,
    "audio_processed": true
  },
  
  "video_analysis": {
    "total_frames": 300,
    "frames_by_decision": {
      "allow": 280,
      "review": 15,
      "reject": 5
    },
    "frames_by_route": {
      "natural": 200,
      "document": 80,
      "mixed": 20
    },
    "segments": [
      {
        "segment_id": "seg_001",
        "decision": "review",
        "start_frame": 0,
        "end_frame": 62,
        "start_ms": 0,
        "end_ms": 2500,
        "reason_codes": ["FACE_DETECTED"],
        "findings": [
          {
            "finding_id": "vf_seg001_001",
            "type": "vision",
            "category": "face",
            "confidence": 0.97,
            "bbox": [100, 150, 400, 600]
          },
          {
            "finding_id": "vf_seg001_002",
            "type": "vision",
            "category": "face",
            "confidence": 0.94,
            "bbox": [800, 200, 1100, 650]
          }
        ]
      },
      {
        "segment_id": "seg_002",
        "decision": "allow",
        "start_frame": 63,
        "end_frame": 200,
        "start_ms": 2500,
        "end_ms": 8000
      },
      {
        "segment_id": "seg_003",
        "decision": "reject",
        "start_frame": 201,
        "end_frame": 300,
        "start_ms": 8000,
        "end_ms": 12000,
        "reason_codes": ["PII_ID_CARD_DETECTED", "DOCUMENT_ROUTE_CRITICAL_FINDING"],
        "findings": [
          {
            "finding_id": "vf_seg003_001",
            "type": "pii",
            "category": "id_card",
            "confidence": 0.992,
            "bbox": [500, 400, 900, 700],
            "detected_fields": ["id_number", "name", "address"]
          }
        ]
      }
    ]
  },
  
  "audio_analysis": {
    "processed": true,
    "audio_source": "native",
    "audio_duration_ms": 12000,
    "audio_decision": "review",
    "reason_codes": ["SPEECH_CONTAINS_PII"],
    "findings": {
      "pii_entities": [
        {
          "entity_type": "ID_NUMBER",
          "confidence": 0.89,
          "start_ms": 1500,
          "end_ms": 2000,
          "text": "[PII]",
          "redaction_strategy": "beep",
          "redacted_audio_uri": "s3://bucket/redacted_audio.wav"
        }
      ],
      "safety_level": "safe",
      "keywords_detected": 0
    }
  },
  
  "policy_decision": {
    "overall_decision": "REJECT",
    "decision_source": "video_frames",  // video segment 中 reject_frames>0
    "confidence": 0.99,
    "recommendation": "Content contains PII. Manual review recommended before release.",
    "reason_codes": [
      "frame_reject_threshold_exceeded",
      "pii_id_card_detected",
      "audio_pii_detected"
    ]
  },
  
  "redaction_outputs": {
    "compliant_video_uri": "s3://bucket/processing/{run_id}/compliant_video.mp4",
    "redacted_video_size_bytes": 262144000,
    "redaction_mode": "selective",  // 仅脱敏 REVIEW/REJECT 部分
    "redaction_quality_preset": "fast",  // fast / balanced / high_quality
    "redaction_details": {
      "total_redacted_frames": 20,
      "total_redacted_segments": 2,
      "redaction_methods": {
        "facial_blur": 8,
        "id_card_pixelate": 10,
        "audio_beep": 1
      }
    },
    "redacted_audio_uri": "s3://bucket/processing/{run_id}/redacted_audio.wav"
  },
  
  "provider_versions": {
    "video_framework": "ffmpeg/7.0",
    "frame_extraction": "pillow/10.1.0",
    "picture_orchestrator": "v2.3.1",
    "picture_providers": {
      "ocr": "paddleocr/2.7.0",
      "pii": "presidio/2.2.350",
      "safety": "shieldgemma2",
      "vision": "yolo26@ultralytics/8.0",
      "segmentation": "sam2@meta"
    },
    "audio_pipeline": "v1.8.2",
    "audio_providers": {
      "asr": "faster-whisper",
      "diarization": "pyannote/3.1",
      "pii": "presidio/2.2.350",
      "safety": "qwen3guard"
    }
  },
  
  "lineage": {
    "input_file_hash": "sha256:a1b2c3d4e5f6...",
    "provenance_events": [
      {
        "timestamp": "2025-01-24T10:15:00Z",
        "event": "media_detection",
        "details": {"detected_format": "video_container", "codec": "h264"}
      },
      {
        "timestamp": "2025-01-24T10:15:05Z",
        "event": "frame_extraction",
        "details": {"frames_extracted": 300, "stride": 1}
      },
      {
        "timestamp": "2025-01-24T10:15:20Z",
        "event": "frame_analysis_complete",
        "details": {"duration_ms": 15000}
      },
      {
        "timestamp": "2025-01-24T10:15:25Z",
        "event": "audio_extraction",
        "details": {"audio_source": "native_track"}
      },
      {
        "timestamp": "2025-01-24T10:15:28Z",
        "event": "policy_decision",
        "details": {"decision": "reject"}
      },
      {
        "timestamp": "2025-01-24T10:15:30Z",
        "event": "redaction_complete",
        "details": {"output_uri": "s3://..."}
      }
    ]
  },
  
  "audit_info": {
    "triggered_by": "compliance_system",
    "user_id": "user_123",
    "organization_id": "org_abc",
    "policy_version": "v2.1.0",
    "policy_rules_evaluated": 12,
    "rules_triggered": 2
  }
}
```

**frame_manifest.jsonl 详细示例（完整帧级别记录）：**
```jsonl
{"frame_id": "f_0001", "frame_index": 1, "pts_ms": 40, "duration_ms": 40, "route": "natural", "decision": "review", "findings_count": 1, "findings": [{"type": "face", "confidence": 0.97}]}
{"frame_id": "f_0002", "frame_index": 2, "pts_ms": 80, "duration_ms": 40, "route": "natural", "decision": "review", "findings_count": 1, "findings": [{"type": "face", "confidence": 0.96}]}
{"frame_id": "f_0003", "frame_index": 3, "pts_ms": 120, "duration_ms": 40, "route": "natural", "decision": "allow", "findings_count": 0}
...
{"frame_id": "f_0201", "frame_index": 201, "pts_ms": 8000, "duration_ms": 40, "route": "document", "decision": "reject", "findings_count": 1, "findings": [{"type": "id_card", "confidence": 0.992}]}
```

**redaction_manifest.jsonl 示例（脱敏操作记录）：**
```jsonl
{"redaction_id": "red_001", "frame_id": "f_0001", "finding_id": "vf001", "redaction_method": "blur", "bbox": [100, 150, 400, 600], "intensity": 0.8}
{"redaction_id": "red_002", "frame_id": "f_0201", "finding_id": "vf_seg003_001", "redaction_method": "pixelate", "bbox": [500, 400, 900, 700], "intensity": 1.0}
{"redaction_id": "red_audio_001", "segment_id": "audio", "start_ms": 1500, "end_ms": 2000, "redaction_method": "beep", "frequency_hz": 1000}
```

---

## 5. 四种模态对比总表

| 维度 | TEXT | AUDIO | PICTURE | VIDEO |
|------|------|-------|---------|-------|
| **输入维度** | 1D (标量文本) | 1D时序 (音波) | 2D空间 (单帧) | 2D + 1D时序 (画面+时间) |
| **核心复用** | — | 文本检测复用TEXT | OCR+PII | Picture逐帧 + Audio混音 |
| **并行优化** | B2(∥), E1(∥) | B2(∥), E1(∥) | Route内三链并行 | 帧级并行 |
| **脱敏手段** | 文本替换 (⟨TAG⟩) | 音频消音/蜂鸣 | 图像遮挡/模糊/像素化 | 视频帧脱敏 + 音轨消音 |
| **最小环境** | Python + text库 | Python + audio库 + FFmpeg | Python + picture库 + OpenCV | Python + video库 + FFmpeg + Picture + Audio |
| **典型吞吐** | 文档/秒 | 分钟/秒 | 图像/秒 | 帧/秒 |
| **降级机制** | 多级回落 (CLI → Library → 规则) | 多级回落 (ASR → Whisper → Sidecar → 占位) | Provider mock (所有provider可mock) | Frame降级 + Audio降级 |
| **最终产物** | decision.json + 脱敏文本摘要 | decision.json + release_package.json + 脱敏音频 | report.json + 脱敏图像 | decision.json + 脱敏视频 + 分段manifest |
| **审计追踪** | OpenLineage 事件 | OpenLineage 事件 | Timestamp+Job ID | Frame manifest + Segment manifest |

---

## 5.1 Provider 工厂函数详解

每个模态都采用**工厂模式**（Factory Pattern）来选择和实例化 Provider，实现"环境变量驱动"而不改代码的灵活性。

### 5.1.1 PICTURE 模态 Provider 工厂

**picture/application/use_cases.py 的核心工厂函数：**

```python
from picture.config import PictureSettings, get_settings
from picture.providers.ocr import *
from picture.providers.pii import *
from picture.providers.safety import *
from picture.providers.vision import *
from picture.providers.segmentation import *

class PictureProviderFactory:
    """工厂类, 根据配置生成各个provider实例"""
    
    @staticmethod
    def create_ocr_provider(settings: PictureSettings = None):
        """OCR Provider 工厂"""
        settings = settings or get_settings()
        provider_name = settings.ocr_provider.lower()
        
        if provider_name == "paddleocr":
            from picture.providers.ocr.paddleocr import PaddleOCRProvider
            return PaddleOCRProvider(
                model_name=settings.paddleocr_model,
                use_gpu=settings.use_gpu,
                use_angle_cls=True
            )
        elif provider_name == "mineru":
            from picture.providers.ocr.mineru import MineRUProvider  
            return MineRUProvider(model_path=settings.mineru_model_path)
        elif provider_name == "surya":
            from picture.providers.ocr.surya import SuryaProvider
            return SuryaProvider(model_name=settings.surya_model)
        else:
            # 默认Mock provider
            from picture.providers.ocr.mock import MockOCRProvider
            return MockOCRProvider()
    
    @staticmethod
    def create_safety_moderator(settings: PictureSettings = None):
        """Safety Moderator 工厂"""
        settings = settings or get_settings()
        provider_name = settings.safety_provider.lower()
        
        if provider_name == "shieldgemma2":
            from picture.providers.safety.shieldgemma2 import ShieldGemmaSafetyModerator
            return ShieldGemmaSafetyModerator(
                model_name=settings.shieldgemma_model,  # "google/shieldgemma-2b-img"
                device=settings.shieldgemma_device,      # "auto" / "cuda" / "cpu"
                cache_dir=settings.hf_cache_dir
            )
        else:
            from picture.providers.safety.mock import MockSafetyModerator
            return MockSafetyModerator()
    
    @staticmethod
    def create_vision_detector(settings: PictureSettings = None):
        """Vision Detector 工厂（目标检测）"""
        settings = settings or get_settings()
        provider_name = settings.vision_provider.lower()
        
        if provider_name == "yolo26":
            from picture.providers.vision.yolo26 import YOLO26VisionDetector
            return YOLO26VisionDetector(
                model_path=settings.yolo26_model_path,  # "yolov8m.pt"
                confidence_threshold=settings.vision_confidence_threshold
            )
        elif provider_name == "grounding_dino":
            from picture.providers.vision.grounding_dino import GroundingDINODetector
            return GroundingDINODetector(
                model_id=settings.grounding_dino_model_id,
                confidence_threshold=settings.vision_confidence_threshold
            )
        else:
            from picture.providers.vision.mock import MockVisionDetector
            return MockVisionDetector()
    
    @staticmethod
    def create_segmentation_provider(settings: PictureSettings = None):
        """Segmentation Provider 工厂"""
        settings = settings or get_settings()
        provider_name = settings.segmentation_provider.lower()
        
        if provider_name == "sam2":
            from picture.providers.segmentation.sam2 import SAM2SegmentationProvider
            return SAM2SegmentationProvider(
                model_id=settings.sam2_model_id,  # "facebook/sam2-hiera-large"
                device=settings.segmentation_device
            )
        else:
            from picture.providers.segmentation.mock import MockSegmentationProvider
            return MockSegmentationProvider()
    
    @staticmethod
    def create_pii_detector(settings: PictureSettings = None):
        """PII Detector 工厂（复用 Presidio）"""
        settings = settings or get_settings()
        
        if settings.use_presidio_pii:
            from picture.providers.pii.presidio import PresidioPIIDetector
            return PresidioPIIDetector(
                languages=settings.presidio_languages
            )
        else:
            from picture.providers.pii.mock import MockPIIDetector
            return MockPIIDetector()
    
    @staticmethod
    def create_redactor(settings: PictureSettings = None):
        """Redactor 工厂（脱敏渲染）"""
        settings = settings or get_settings()
        
        # 优先使用 OpenCV, 降级到 Pillow
        try:
            from picture.providers.redaction.opencv_redactor import OpenCVRedactor
            return OpenCVRedactor(
                blur_strength=settings.redaction_blur_strength,
                pixelate_block_size=settings.redaction_pixelate_block_size
            )
        except ImportError:
            from picture.providers.redaction.pillow_redactor import PillowRedactor
            return PillowRedactor(
                blur_strength=settings.redaction_blur_strength,
                pixelate_block_size=settings.redaction_pixelate_block_size
            )
```

**使用示例：**
```python
from picture.application.use_cases import PictureProviderFactory
from picture.config import get_settings

# 获取当前配置
settings = get_settings()

# 创建不同的 provider 实例
ocr = PictureProviderFactory.create_ocr_provider(settings)
safety = PictureProviderFactory.create_safety_moderator(settings)
vision = PictureProviderFactory.create_vision_detector(settings)
segmentation = PictureProviderFactory.create_segmentation_provider(settings)
pii = PictureProviderFactory.create_pii_detector(settings)
redactor = PictureProviderFactory.create_redactor(settings)

# 所有 provider 现在都已根据配置自动选择最优实现
# 如果配置不可用，自动降级到 Mock
ocr_result = ocr.detect(image_path)  # 用的是 PaddleOCR 或 Mock
```

**环境变量配置示例：**
```bash
# 选择 OCR 提供者
export PICTURE_OCR_PROVIDER=paddleocr      # 或 mineru, surya
export PADDLEOCR_MODEL=paddleocr_v3

# 选择 Safety 提供者
export PICTURE_SAFETY_PROVIDER=shieldgemma2  # 或 mock
export SHIELDGEMMA_DEVICE=cuda

# 选择 Vision 提供者
export PICTURE_VISION_PROVIDER=yolo26        # 或 grounding_dino
export VISION_CONFIDENCE_THRESHOLD=0.45

# 选择 Segmentation 提供者
export PICTURE_SEGMENTATION_PROVIDER=sam2
export SAM2_MODEL_ID=facebook/sam2-hiera-large

# PII 检测
export PICTURE_USE_PRESIDIO_PII=true

# 脱敏配置
export PICTURE_REDACTION_BLUR_STRENGTH=25
export PICTURE_REDACTION_PIXELATE_BLOCK_SIZE=16
```

### 5.1.2 TEXT 模态 Provider 工厂

**text/application/use_cases.py 的关键工厂函数：**

```python
class TextProviderFactory:
    """TEXT 模态的 Provider 工厂"""
    
    @staticmethod
    def create_secrets_scanner():
        """Secrets Scanner 工厂"""
        # TruffleHog 总有 Mock 实现，不会完全失败
        try:
            return TruffleHogScanner(cli_path="/usr/bin/trufflehog")
        except FileNotFoundError:
            return MockSecretsScanner()
    
    @staticmethod
    def create_license_scanner():
        """License Scanner 工厂"""
        try:
            return ScanCodeLicenseScanner(cli_path="/usr/bin/scancode")
        except FileNotFoundError:
            return MockLicenseScanner()
    
    @staticmethod
    def create_pii_detector(use_huggingface: bool = False):
        """PII Detector 工厂"""
        if use_huggingface:
            return PresidioPIIDetectorWithHF()
        else:
            return PresidioPIIDetector(use_hf_models=False)
    
    @staticmethod
    def create_safety_classifier():
        """Safety Classifier 工厂"""
        try:
            return Qwen3GuardSafetyClassifier(model_name="qwen/qwen3guard")
        except Exception:
            return MockSafetyClassifier()
    
    @staticmethod
    def create_dedup_provider():
        """Dedup 工厂"""
        try:
            return DataSketchDeduplicator(minhash_num_perm=128)
        except Exception:
            return SimpleHashDeduplicator()
```

### 5.1.3 AUDIO 模态 Provider 工厂

**audio/application/use_cases.py 的关键工厂函数：**

```python
class AudioProviderFactory:
    """AUDIO 模态的 Provider 工厂"""
    
    @staticmethod
    def create_asr_provider():
        """ASR 工厂（三层降级）"""
        # Tier 1: Qwen3-ASR
        try:
            return Qwen3ASRProvider(model_name="qwen/qwen3-asr")
        except Exception:
            pass
        
        # Tier 2: faster-whisper
        try:
            return FasterWhisperASRProvider(model_size="base")
        except Exception:
            pass
        
        # Tier 3: Sidecar 或 Mock
        return MockASRProvider()
    
    @staticmethod
    def create_diarization_provider():
        """Speaker Diarization 工厂"""
        try:
            return PyannoteAudioDiarizer(
                model_name="pyannote/speaker-diarization-3.1",
                use_auth_token=os.environ.get("HF_TOKEN")
            )
        except Exception:
            return MockDiarizer()
    
    @staticmethod
    def create_audio_redactor():
        """Audio Redaction 工厂"""
        # 总是使用 FFmpeg，通过配置选择策略
        return FFmpegAudioRedactor(
            redaction_strategy=os.environ.get("AUDIO_REDACTION_STRATEGY", "beep"),
            beep_frequency=int(os.environ.get("AUDIO_BEEP_FREQUENCY", "1000"))
        )
```

### 5.1.4 VIDEO 模态 Provider 工厂

**video/application/use_cases.py 的关键工厂函数：**

```python
class VideoProviderFactory:
    """VIDEO 模态的 Provider 工厂（组编排）"""
    
    @staticmethod
    def create_picture_orchestrator():
        """Picture 引擎工厂"""
        from picture.application.orchestrator import PictureComplianceOrchestrator
        settings = get_settings()
        return PictureComplianceOrchestrator(settings=settings)
    
    @staticmethod
    def create_audio_pipeline():
        """Audio Pipeline 工厂"""
        from audio.pipeline import AudioCompliancePipeline
        settings = get_settings()
        return AudioCompliancePipeline(settings=settings)
    
    @staticmethod
    def create_frame_loader():
        """Frame Loader 工厂（处理三种输入）"""
        return FrameSequenceLoader(
            frame_stride=int(os.environ.get("VIDEO_FRAME_STRIDE", "1")),
            max_frames=int(os.environ.get("VIDEO_MAX_FRAMES", "600")),
            scale_width=int(os.environ.get("VIDEO_SCALE_WIDTH", "1280")),
            scale_height=int(os.environ.get("VIDEO_SCALE_HEIGHT", "720"))
        )
```

---

## 6. 共性设计模式

### 6.1 三层降级策略

所有四个模态都遵循**"最优→备选→规则→占位符"** 的递进降级模式：

```

### 6.1 三层降级策略

所有四个模态都遵循**"最优→备选→规则→占位符"** 的递进降级模式：

```
最优级 (生产模型)
    ↓ (不可用 or 失败)
备选级 (轻量模型 or 另一工具)
    ↓ (都不可用 or 失败)
规则级 (正则 / 关键词 / 启发式)
    ↓ (都失败 or 明确关闭)
占位符 (Mock / 透传原值 / 默认安全)
```

### 6.1.1 各模态的具体降级链实现

#### TEXT 模态 - Secrets 扫描降级链

```python
# text/steps/b2a_trufflehog_scan.py - 5层降级链
def run(records: List[SourceRecord], settings: TextSettings) -> List[SecretRecord]:
    """TruffleHog Secrets 扫描 - 最复杂的降级链"""
    
    results = []
    
    for record in records:
        secret_hits = []
        
        # 降级方案 1: 原生 TruffleHog CLI
        try:
            output = subprocess.run(
                ["trufflehog", "filesystem", "--json", record.file_path],
                capture_output=True,
                text=True,
                timeout=300
            )
            if output.returncode == 0:
                for line in output.stdout.split('\n'):
                    if line:
                        hit_obj = json.loads(line)
                        secret_hits.append(SecretHit(
                            source_id=record.source_id,
                            secret_type=hit_obj.get('type'),
                            confidence=hit_obj.get('confidence', 0.8),
                            line_number=hit_obj.get('line'),
                            engine_name="trufflehog_native"
                        ))
                results.append(SecretRecord(
                    source_id=record.source_id,
                    hits=secret_hits,
                    degradation_level=0  # 无降级
                ))
                continue
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("TruffleHog CLI not found or timeout")
        
        # 降级方案 2: TruffleHog Python 库
        try:
            from truffleHog3 import regex_scanner
            detections = regex_scanner.scan_directory(record.file_path)
            for detection in detections:
                secret_hits.append(SecretHit(
                    source_id=record.source_id,
                    secret_type=detection.get('type'),
                    confidence=0.7,  # 库的置信度稍低
                    line_number=detection.get('line'),
                    engine_name="trufflehog_lib"
                ))
            results.append(SecretRecord(
                source_id=record.source_id,
                hits=secret_hits,
                degradation_level=1  # 降级一级
            ))
            continue
        except ImportError:
            logger.warning("TruffleHog library not installed")
        
        # 降级方案 3: 正则启发式规则
        try:
            patterns = {
                'aws_key': r'AKIA[0-9A-Z]{16}',
                'github_pat': r'ghp_[A-Za-z0-9_]{36,}',
                'private_key': r'-----BEGIN (RSA|EC|OPENSSH) PRIVATE KEY-----',
                'api_key': r'api[_-]?key["\']?\s*[:=]\s*["\']([A-Za-z0-9\-_]{20,})',
            }
            
            with open(record.file_path, 'r', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    for pattern_name, pattern in patterns.items():
                        if re.search(pattern, line):
                            secret_hits.append(SecretHit(
                                source_id=record.source_id,
                                secret_type=pattern_name,
                                confidence=0.5,  # 启发式置信度更低
                                line_number=line_num,
                                engine_name="regex_patterns"
                            ))
            
            results.append(SecretRecord(
                source_id=record.source_id,
                hits=secret_hits,
                degradation_level=2  # 降级两级
            ))
            continue
        except Exception as e:
            logger.error(f"Regex scanning failed: {e}")
        
        # 降级方案 4: 基于文件名启发式
        try:
            suspicious_filenames = [
                '.env', '.env.local', 'secrets.json', 'credentials.json',
                'config.yaml', 'private.key', 'id_rsa'
            ]
            
            basename = os.path.basename(record.file_path)
            if basename in suspicious_filenames:
                secret_hits.append(SecretHit(
                    source_id=record.source_id,
                    secret_type="suspected_secret_file",
                    confidence=0.6,
                    line_number=-1,
                    engine_name="filename_heuristic"
                ))
            
            results.append(SecretRecord(
                source_id=record.source_id,
                hits=secret_hits,
                degradation_level=3  # 降级三级
            ))
            continue
        except Exception:
            pass
        
        # 降级方案 5: 占位符（无法检测）
        results.append(SecretRecord(
            source_id=record.source_id,
            hits=[],
            degradation_level=4,  # 完全降级
            warning="All secrets scanning methods failed; returning empty results"
        ))
    
    return results
```

#### AUDIO 模态 - ASR 转录降级链

```python
# audio/steps/c1_asr_transcribe.py - 4层降级链
def run(records: List[SourceRecord], settings: AudioSettings) -> List[ASRResult]:
    """ASR Transcription - 严格的降级链"""
    
    results = []
    
    for record in records:
        # 降级方案 1: Qwen3-ASR（最高质量）
        try:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            
            model_id = "qwen/qwen3-asr"
            device = "cuda" if settings.use_gpu else "cpu"
            dtype = torch.float16 if settings.use_gpu else torch.float32
            
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=dtype, use_safetensors=True, cache_dir=settings.hf_cache_dir
            ).to(device)
            processor = AutoProcessor.from_pretrained(model_id, cache_dir=settings.hf_cache_dir)
            
            audio, sr = librosa.load(record.file_path, sr=16000)
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            
            with torch.no_grad():
                predicted_ids = model.generate(**inputs.to(device))
            
            transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            results.append(ASRResult(
                source_id=record.source_id,
                transcript=transcript,
                segments=[ASRSegment(text=transcript, start_ms=0, end_ms=int(len(audio)/sr*1000), confidence=0.95)],
                engine_name="qwen3_asr",
                degradation_level=0
            ))
            continue
        except Exception as e:
            logger.warning(f"Qwen3-ASR failed: {e}")
        
        # 降级方案 2: faster-whisper（次优，速度快）
        try:
            from faster_whisper import WhisperModel
            
            model = WhisperModel(
                settings.whisper_model_size,  # "base"
                device="cuda" if settings.use_gpu else "cpu",
                compute_type="float16" if settings.use_gpu else "float32"
            )
            
            segments, info = model.transcribe(
                record.file_path,
                language=settings.audio_language,
                beam_size=5
            )
            
            transcript = " ".join([seg.text for seg in segments])
            
            results.append(ASRResult(
                source_id=record.source_id,
                transcript=transcript,
                segments=[ASRSegment(
                    text=seg.text,
                    start_ms=int(seg.start*1000),
                    end_ms=int(seg.end*1000),
                    confidence=seg.confidence if hasattr(seg, 'confidence') else 0.8
                ) for seg in segments],
                engine_name="faster_whisper",
                degradation_level=1
            ))
            continue
        except Exception as e:
            logger.warning(f"faster-whisper failed: {e}")
        
        # 降级方案 3: Sidecar 转录文件
        try:
            # 寻找 sidecar .transcript.jsonl 或 .transcript.json 文件
            sidecar_transcript = record.file_path.replace(os.path.splitext(record.file_path)[1], '.transcript.jsonl')
            
            if os.path.exists(sidecar_transcript):
                segments = []
                with open(sidecar_transcript, 'r') as f:
                    for line in f:
                        seg = json.loads(line)
                        segments.append(ASRSegment(
                            text=seg['text'],
                            start_ms=int(seg['start_ms']),
                            end_ms=int(seg['end_ms']),
                            confidence=seg.get('confidence', 0.85)
                        ))
                
                transcript = " ".join([seg.text for seg in segments])
                
                results.append(ASRResult(
                    source_id=record.source_id,
                    transcript=transcript,
                    segments=segments,
                    engine_name="sidecar_transcript",
                    degradation_level=2
                ))
                continue
        except Exception as e:
            logger.warning(f"Sidecar transcript loading failed: {e}")
        
        # 降级方案 4: 占位符（完全失败）
        results.append(ASRResult(
            source_id=record.source_id,
            transcript="[无法转录]",
            segments=[ASRSegment(
                text="[Audio content not transcribed]",
                start_ms=0,
                end_ms=-1,
                confidence=0
            )],
            engine_name="placeholder",
            degradation_level=3,
            warning="All ASR methods failed; using placeholder transcript"
        ))
    
    return results
```

#### PICTURE 模态 - OCR Provider 降级链

```python
# picture/application/use_cases.py - OCR 工厂的降级实现
def create_ocr_provider(settings: PictureSettings = None) -> OCRProvider:
    """OCR Provider 工厂 - 自动降级链"""
    settings = settings or get_settings()
    
    # 降级方案 1: PaddleOCR（最好的准确度）
    try:
        from picture.providers.ocr.paddleocr import PaddleOCRProvider
        return PaddleOCRProvider(
            model_name=settings.paddleocr_model,
            use_gpu=settings.use_gpu,
            use_angle_cls=True,
            degradation_level=0
        )
    except Exception as e:
        logger.warning(f"PaddleOCR init failed: {e}")
    
    # 降级方案 2: MiniRU（轻量级）
    try:
        from picture.providers.ocr.mineru import MineRUProvider
        return MineRUProvider(
            model_path=settings.mineru_model_path,
            degradation_level=1
        )
    except Exception as e:
        logger.warning(f"MineRU init failed: {e}")
    
    # 降级方案 3: Surya（开源）
    try:
        from picture.providers.ocr.surya import SuryaProvider
        return SuryaProvider(
            model_name=settings.surya_model,
            degradation_level=2
        )
    except Exception as e:
        logger.warning(f"Surya init failed: {e}")
    
    # 降级方案 4: Tesseract（经典）
    try:
        from picture.providers.ocr.tesseract import TesseractOCRProvider
        return TesseractOCRProvider(
            lang=settings.ocr_language,
            degradation_level=3
        )
    except Exception as e:
        logger.warning(f"Tesseract not available: {e}")
    
    # 降级方案 5: Mock（占位符）
    logger.error("All OCR providers failed; using Mock")
    from picture.providers.ocr.mock import MockOCRProvider
    return MockOCRProvider(degradation_level=4)
```

#### VIDEO 模态 - 帧处理降级链

```python
# video/pipeline.py - 帧处理的容错实现
def analyze_frames(sequence: Sequence, max_workers: int = 8) -> List[FrameJob]:
    """并行帧处理 - 单帧失败时的降级策略"""
    
    frame_jobs = []
    failed_frames = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        for frame in sequence.frames:
            future = executor.submit(
                process_single_frame,
                frame=frame,
                settings=settings
            )
            futures[future] = frame
        
        for future in as_completed(futures):
            frame = futures[future]
            
            try:
                # 降级方案 1：正常处理
                result = future.result(timeout=60)
                frame_jobs.append(FrameJob(
                    frame_id=frame.frame_id,
                    decision=result['decision'],
                    findings=result['findings'],
                    degradation_level=0
                ))
                
            except TimeoutError:
                # 降级方案 2：单帧超时 → REVIEW（保守决策）
                logger.warning(f"Frame {frame.frame_id} processing timeout")
                frame_jobs.append(FrameJob(
                    frame_id=frame.frame_id,
                    decision='review',
                    findings=[],
                    degradation_level=1,
                    reason='Processing timeout'
                ))
                failed_frames.append(frame.frame_id)
                
            except Exception as e:
                # 降级方案 3：单帧异常 → ALLOW（最优假设）
                logger.error(f"Frame {frame.frame_id} processing failed: {e}")
                frame_jobs.append(FrameJob(
                    frame_id=frame.frame_id,
                    decision='allow',
                    findings=[],
                    degradation_level=2,
                    reason=f'Processing error: {str(e)}'
                ))
                failed_frames.append(frame.frame_id)
    
    # 记录降级统计
    stats = {
        'total_frames': len(sequence.frames),
        'success_frames': len([j for j in frame_jobs if j.degradation_level == 0]),
        'timeout_frames': len([j for j in frame_jobs if j.degradation_level == 1]),
        'error_frames': len([j for j in frame_jobs if j.degradation_level == 2])
    }
    
    logger.info(f"Frame processing stats: {stats}")
    
    return frame_jobs
```

### 6.2 JSONL 作为通用产物格式

- **可流式处理**：每行一条记录，不需要全部加载到内存
- **可审计**：每条记录自成体系，查询时间范围时可二分搜索
- **可追踪**：每条记录带 source_id/frame_id/segment_id 可追溯
- **可并行聚合**：多条管道独立产生 JSONL，后续可并行合并

### 6.3 决策的三级结构

```
Finding Level: 单个发现 (违反什么规则)
    ↓
Evidence Level: 证据聚合 (有多少违反、哪些类别)
    ↓
Decision Level: 最终决策 (Allow / Review / Reject / pass_raw / pass_redacted / drop)
```

### 6.4 Provider 的抽象化设计

每个模态都定义了清晰的 Provider 抽象接口，允许：
- Mock 实现用于开发测试
- 多个真实实现可共存（YOLO vs DINO）
- 运行时切换无需改代码（环境变量配置）
- 总是有 fallback 实现（不会完全崩溃）

---

## 7. 端到端数据流示例

### 7.1 完整场景：用户上传一段视频

```
User Upload Video "meeting.mp4"
    ↓
[VIDEO Pipeline]
    ├─ FFmpeg probe → 分辨率 1920x1080, 25fps, 60sec, stereo audio
    │
    ├─ Load Sequence
    │  ├─ frame_stride=5 (每5帧取1帧)
    │  ├─ 抽取 300 帧
    │  └─ Frames[] with pts_ms (0, 200, 400, ..., 11800)
    │
    ├─ Analyze Frames (并行，max_workers=8)
    │  ├─ 帧1 (0ms) → Picture.natural → 发现: 人脸×1
    │  ├─ 帧2 (200ms) → Picture.mixed → 发现: 人脸×2, 文字(身份证)×1
    │  ├─ 帧3 (400ms) → Picture.natural → 发现: 无
    │  └─ ... (300帧)
    │
    ├─ Derive Segments
    │  ├─ s1: 0-1000ms, decision=REVIEW (人脸)
    │  ├─ s2: 1000-5000ms, decision=ALLOW (无敏感内容)
    │  ├─ s3: 5000-8000ms, decision=REJECT (身份证)
    │  └─ s4: 8000-12000ms, decision=ALLOW
    │
    ├─ Extract Native Audio
    │  └─ ffmpeg → audio.wav (mono, 16kHz)
    │
    ├─ Run Audio Pipeline
    │  ├─ ASR (Qwen3-ASR) → "我的身份证号是 xxxxx"
    │  ├─ PII Detection → 发现身份证号 @1500ms
    │  ├─ Audio Redaction → beep covering 1500~2000ms
    │  └─ audio_decision = REVIEW
    │
    ├─ Aggregate Policy
    │  ├─ video 最严 = REJECT (来自帧3的身份证)
    │  ├─ audio 决策 = REVIEW (来自语音PII)
    │  ├─ 最终 = REJECT (拒绝通过)
    │  └─ reason_codes = ["PII_VISUAL_ID", "PII_SPEECH_ID"]
    │
    ├─ Render Sequence
    │  ├─ 脱敏帧序列 + 脱敏音轨
    │  ├─ FFmpeg mux → compliant_video.mp4
    │  └─ 输出质量 = 720p (降采样)
    │
    └─ Outputs
        ├─ frame_manifest.jsonl (300行, 每帧记录)
        ├─ segment_manifest.jsonl (4行, 每段记录)
        ├─ video_findings.jsonl (N行, 每个发现)
        ├─ decision.json (最终决策)
        ├─ compliant_video.mp4 (脱敏视频)
        └─ redacted_audio.wav (脱敏音轨)

[总耗时: ~60秒 (取决于并行度和模型速度)]
```

---

## 8. 关键结论

### 8.1 设计哲学

1. **降维处理**: Audio → Transcript (复用Text), Video → Frames (复用Picture) + Audio (复用Audio)
2. **复用能力**: 不重复建轮子，上层编排下层能力
3. **多源兼容**: 支持多个supply来源（多个ASR模型、多个OCR引擎等）
4. **优雅降级**: 没有某项能力时不中断，而是用规则或mock
5. **审计至上**: 所有产物都是JSONL/JSON，可完整重现

### 8.2 工程质量

- **清晰的关系图**: 每个步骤的输入/输出都是强类型Pydantic模型
- **血缘追踪**: OpenLineage集成，知道每条数据从何而来
- **并行优化**: 关键路径 (B2∥, E1∥, Picture混合链∥) 减少端到端延迟
- **可观测性**: 每步记录latency, provider_version, 异常情况

### 8.3 生产适配

- **环境变量驱动**: COMPLIANCE_*, PICTURE_*, AUDIO_*, VIDEO_* 前缀
- **存储可切换**: LocalFile vs S3 无缝切换
- **模型版本固定**: 支持HF模型的特定版本指定，而不总是拉最新
- **监控友好**: 每条决策都有reason_codes和evidence_summary便于告警和检查

---

## 附录：技术栈速查

| 模态 | 工具/库 | 版本/信息 | 用途 |
|------|--------|---------|------|
| **TEXT** | trafilatura | HTML提取 | 网页文本提取 |
| | PyMuPDF | PDF处理 | PDF文本提取 |
| | Presidio | PII检测 | 统一PII识别框架 |
| | Qwen3Guard | 千问0.6B | 文本安全三级分类 |
| | OPA | Rego引擎 | 策略编织 |
| | FlashText2 | 关键词 | 高效关键词匹配 |
| | datasketch | MinHash | 近似去重 |
| | TruffleHog | CLI | Secrets扫描 |
| | ScanCode | CLI | License/Copyright扫描 |
| **AUDIO** | FFmpeg | 音频处理 | 归一化、消音、mux |
| | Qwen3-ASR | 千问0.6B | 语音转文本 |
| | faster-whisper | Whisper快速版 | ASR backup |
| | pyannote | v3.1 | 说话人分离 |
| | 复用TEXT | 前述所有 | Transcript文本检测 |
| **PICTURE** | PaddleOCR-VL | Paddle深度学习 | OCR主模型 |
| | Presidio | | PII检测 |
| | ShieldGemma2 | Google | 图像安全审核 |
| | YOLO26 | Ultralytics | 目标检测 |
| | SAM2 | Meta | 像素级分割 |
| | OpenCV / Pillow | | 图像处理和脱敏 |
| **VIDEO** | FFmpeg/ffprobe | 视频处理 | 抽帧、容器处理 |
| | Pillow | GIF/WebP | 动画处理 |
| | 复用PICTURE | | 逐帧视觉检测 |
| | 复用AUDIO | | 音轨检测 |

