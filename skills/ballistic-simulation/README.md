# ballistic-simulation

面向**内/外弹道学**的论文复现与代码编写智能体 skill。

## 功能

- **论文复现**：给一篇 PDF 论文 → 渲染 → 公式 OCR → 模型重建 → 确认门 → 求解器生成 → 数值仿真 → 与论文结果对齐 → 归因报告。
- **代码编写**：给需求 → 基于已验证模板生成弹道计算代码 + 合理性校验。

## 特性

- **锚点优先**：基于 `templates/` 下已验证求解器（经典内弹道、压缩空气弹射、无后坐炮）改编，不凭空编造。
- **先问后做**：参数/方程缺失时先问用户，不擅自假设。
- **泛化**：模板库自动匹配 + 生成求解器，适配多样化弹道模型；新验证一个模型即可扩充模板库。

## 依赖

- 千问 Qwen-VL API（视觉公式 OCR，唯一外部 API）
- Python 3 + numpy（运行生成的求解器）

## 目录结构

```
ballistic-skill/
├── SKILL.md            # skill 指令（工作流 + 原则）
├── code/               # 流水线代码（llm/pipeline/template_lib）
├── templates/          # 已验证求解器库（含元数据 + 已知误差）
└── .env.example        # API key 配置模板
```

## 配置

复制 `.env.example` 为 `.env`，填入：
- `DASHSCOPE_API_KEY`（千问视觉公式 OCR）
- `DASHSCOPE_MODEL`（如 qwen3-vl-plus）