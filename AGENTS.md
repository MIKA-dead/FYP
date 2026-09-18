# 多模态动画生成代理系统规则

## 核心工作流
1. **意图编译**：将用户提示词 + 参考图 ID 编译为 Intent Schema v1（`.agents/skills/intent-routing/`），产出可检查的 Decision Trace
2. **计划模式**：motion-planner **只基于 Intent Schema** 选模块、排依赖（`scripts/intent/route_modules.py`）
3. **构建模式**：按计划调用 motion-coder、motion-tester、motion-documenter（模块可 mock）
4. **修订模式**：识别 texture / motion / key_pose / speed，仅重跑 `modules_to_rerun`，冻结 bg/camera/appearance
5. **汇总输出**：收集结果与 Decision Trace，生成最终报告

## Intent 路由测试
- 用例：`test-samples/intent/prompts_with_gold.json`（P001–P018）
- **交互测试台**：`python scripts/intent/test_console.py` → http://127.0.0.1:8765
- 一键跑：`python scripts/intent/run_pipeline.py`
- 输出：`outputs/intent/*_decision_trace.json` + `summary.json`
- 给 UI 组员：测试台点「导出给组员」→ `outputs/intent/handoff_for_ui.json`

## 评估指标（每次任务必须收集）
- **任务完成率**：成功子任务数 / 总子任务数
- **响应时间**：从输入到最终输出的总耗时（秒）
- **Token 消耗**：本次任务消耗的总 Token 数
- **子代理效率**：每个子代理的处理时间（秒）
- **质量评分**：由 motion-tester 给出（0-100分）

## 优化策略（已启用）
- [x] 缓存：相同文本描述在 24 小时内直接返回缓存结果
- [x] 并行：无依赖的子任务（如风格化和时间线编辑）并行执行
- [x] 知识预热：生成前先加载 `motion-patterns` 和 `style-library` 技能

## 知识整合
- 每次生成完成后，提取成功模式，更新 `skills/motion-patterns/SKILL.md`
- 评分 >= 85 的风格组合，记录到 `skills/style-library/SKILL.md`
- Intent 路由成功案例（verdict=correct）回写 `skills/intent-routing/` 的 taxonomy / 模块理由模板
- 每周分析 `metrics/` 目录数据，生成性能趋势报告


## 实验控制开关（仅用于消融实验）
- `CACHE_ENABLED`: true/false
- `PARALLEL_ENABLED`: true/false
- `KNOWLEDGE_WARMUP_ENABLED`: true/false