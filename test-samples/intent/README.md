# Intent Routing 测试集

用于验证：**提示词 → Intent Schema → 模块路由 → Decision Trace**。

## 交互测试台（推荐你先用这个）

```bash
python scripts/intent/test_console.py
```

浏览器打开 `http://127.0.0.1:8765`：

1. 左侧点选 gold 用例，或自己粘贴提示词
2. 右侧实时看：调用模块、timeline、plan、自动评分
3. 可人工覆盖 verdict（correct/partial/wrong）
4. 点「导出给组员」→ `outputs/intent/handoff_for_ui.json` + `HANDOFF_README.md`

> 这是**开发自测台**，不是组员的产品 UI。组员只接 `ui_contract` 字段。

## 命令行批量跑

```bash
python scripts/intent/run_pipeline.py
python scripts/intent/run_pipeline.py --case P001
```

## 人工检查要点

1. `intent.timeline` 段数与动作是否合理
2. `intent.modules_needed` 是否包含 gold、是否误召禁止模块
3. `plan.steps` 依赖是否说得通
4. `review.verdict`：自动分 + 你的 manual_override
