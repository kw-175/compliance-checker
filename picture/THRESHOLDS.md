# 检测阈值清单（L4 盘点）

集中登记散落各处的检测阈值，便于统一调优与回归对照。机械收敛到单一配置表需在黄金集基线就绪后进行（scripts/run_picture_golden_bench.py）。

| 阈值 | 默认值 | 位置 | 作用 |
|------|--------|------|------|
| safety_drop_threshold | 0.7 | configs/default_cn_enterprise.yaml | safety 类别触发 DROP 的最低分 |
| min_score_threshold | 0.5 | configs/default_cn_enterprise.yaml | finding 进入策略评估的最低分 |
| SAM3 confidence_threshold | 0.35 | providers/vision/sam3_api.py | SAM3 文本提示分割基础阈值 |
| SAM3 融合降阈值 | face 0.20 / 其他 0.25 | providers/vision/qwen_sam3_fusion.py detect() | 语义确认后的类别级放宽 |
| sam3_keep_without_qwen_threshold | 0.75 | 同上 | 无 Qwen 确认时保留 SAM3 结果的门 |
| semantic_threshold | 0.55 | 同上 | Qwen 语义检出 requires_redaction 的门 |
| _violation_candidate_quality | quality<0.45 拒绝 | providers/safety/qwen_sam3_fusion.py | 安全侧 SAM3 候选综合质量闸 |
| point_box_size_ratio | dangerous 0.05 / explicit 0.16 / hate,self_harm 0.06 / 其余 0.10 | 同上 | 点提示框尺寸比例闸 |
| SCRFD confidence_threshold | 0.40 | providers/vision/privacy_specialists.py | 人脸检测置信度 |
| _face_identifiability | 短边<24px 或 分<0.45 判不可识别 | 同上 | 可识别人脸闸（极小远景脸不脱敏） |
| MediaPipe min_detection_confidence | 0.35（当前环境禁用） | 同上 | 备用人脸检测 |
| clip_moderation_confidence_threshold | 0.55 | video/config/settings.py | 视频动作识别置信度 |
| frame_difference_threshold | 0.02 | video/config/settings.py | 重复帧判定 |
| scene_change_threshold | 0.18 | video/config/settings.py | 场景切换判定 |
| analysis_change_threshold | 0.12 | video/config/settings.py | 分析帧变化判定 |
