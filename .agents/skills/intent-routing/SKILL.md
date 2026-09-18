# Intent Routing（提示词解析 → 模块路由）

## 何时使用
- 用户输入**长提示词 + 参考图 ID**，需要编译成结构化意图再交给 `motion-planner`
- 视频生成后的**修改意见**（texture / motion / key_pose / speed）需要局部重跑模块
- 需要产出可检查的 **Decision Trace**（决策对错表）

## 产物位置
| 文件 | 用途 |
|------|------|
| `intent_schema.json` | Intent Schema v1（Planner 只消费此结构） |
| `module_registry.json` | 可路由模块清单（两周内均为 mock） |
| `action_taxonomy.json` | 细粒度动作 ↔ HumanML3D 粗类 + 修订关键词 |
| `test-samples/intent/prompts_with_gold.json` | 10 条带 gold label 的测试提示词 |
| `scripts/intent/run_pipeline.py` | 端到端演示与自动评分 |

## 流程
1. UI / 调用方提交 `{ prompt, refs, mode: generate|revise }`
2. `parse_intent.py` → Intent Schema
3. `route_modules.py` → 模块调用计划（依赖 / 并行组 / owner_agent）
4. `mock_modules.py` → stub 结果（可替换为真实后端）
5. `evaluate_decision.py` → 对照 gold：`correct` / `partial` / `wrong`

## 原则
- **HumanML3D 分类只挂在 timeline 段上**，不直接解析整段剧本
- **修订默认冻结** `bg_lock` / `camera_lock` / `appearance_ref`，只重跑相关模块
- Planner **禁止**在未产出 Intent 时直接调度 coder
