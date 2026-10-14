# dsh-skill-ballistic-simulation

内/外弹道学的论文复现与代码编写 skill（DeepSeek Harness 插件）。

## 功能
- **论文复现**：给一篇 PDF 论文 → 公式 OCR → 模型重建 → 确认门 → 求解器生成 → 数值仿真 → 对齐 → 归因报告。
- **代码编写**：给需求 → 基于已验证模板生成弹道计算代码 + 校验。

## 安装
```bash
dsh plugin --profile <name> add dsh-skill-ballistic-simulation
```
安装后 skill 即可用；卸载即停用，灵活可控。

## 配置
复制 `skills/ballistic-simulation/.env.example` 为 `.env`，填入：
- `DEEPSEEK_API_KEY`（文本推理/代码生成）
- `DASHSCOPE_API_KEY`（千问视觉公式 OCR）
- `DASHSCOPE_MODEL`（如 qwen3-vl-plus）

## 目录
```
skills/ballistic-simulation/    # skill 本体（SKILL.md + 流水线代码 + 已验证模板库）
cordis.patch.yml                # bundle patch：挂载隔离 skill-filesystem + bundledSkillDir
```