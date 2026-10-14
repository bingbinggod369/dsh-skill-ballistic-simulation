---
name: ballistic-simulation
description: 内/外弹道学的论文复现与代码编写。当用户要求"复现某篇弹道论文"或"编写某弹道模型代码"时使用本 skill。
whenToUse: 用户要求复现弹道论文、编写弹道模型代码、或进行内/外弹道数值仿真时使用。
---

# 弹道仿真复现

## 用途
面向内/外弹道学的论文仿真智能体，两大能力：
- **论文复现**：给一篇 PDF 论文 → 提取方程/参数 → 数值复现 → 与论文结果对齐 → 归因报告。
- **代码编写**：给需求 → 基于已验证模板生成弹道计算代码 + 校验。

## 核心原则（必须遵守）
1. **锚点优先**：一切生成/推断必须基于 `templates/` 下已验证的求解器（事实锚点），不凭空编造公式或参数。
2. **先问后做**：遇参数缺失、方程缺失、不确定假设，必须先问用户，不擅自决定；用户授权"由你决定"后，才可用参考值补齐并标注。
3. **方程闭环**：提取方程后列出未知量、数方程，验证封闭可解。
4. **可复现**：交付完整方程集 + 参数表（逐条溯源）+ 求解配置 + 报告。

## 资源
| 路径 | 说明 |
|---|---|
| `templates/` | 已验证求解器库（经典内弹道、压缩空气弹射、无后坐炮），每个带 `.meta.json`（关键词 + 验证来源 + 已知误差）|
| `code/llm.py` | LLM 接入：`make_backend()`(文本=DeepSeek)、`make_vision_backend()`(视觉=千问)、`vision_ocr()`(公式OCR) |
| `code/pipeline.py` | 复现流水线：`render_pdf`/`ocr_pages`/`reconstruct_model`/`infer_uncertainties`/`generate_solver_with_repair`/`run_from_model` |
| `code/template_lib.py` | 模板加载 + 关键词自动匹配 `match_template()` |

## 任务 A：代码编写
1. 解析需求：模型类型、参数、约束；标记缺失项。
2. 用 `template_lib.match_template(需求文本)` 匹配最接近的已验证模板。
3. 基于匹配模板改编生成代码（锚点优先）；每个公式注释标注出处；缺失参数先问用户。
4. 生成后运行验证 + 物理合理性检查；给出依据。

## 任务 B：论文复现（完整流程，每步记录）
1. **渲染**：`pipeline.render_pdf(pdf, outdir)` → PDF 转图片（LibreOffice Kit，路径可用环境变量 LO_NODE/LO_CLI 配置）。
2. **OCR**：`pipeline.ocr_pages(images)` 逐页识别文字+公式（走千问视觉 API；注意本环境 agent 模型无图像输入，不能自己看图）。
3. **重建**：`pipeline.reconstruct_model(ocr)` 用 DeepSeek 提取结构化模型（假设/方程/参数/结果，results 用标准键 t_start/t_muzzle/v_muzzle/max_overload 等）。
4. **推断**：`pipeline.infer_uncertainties(model)` 找出不确定点。
5. **确认门**：把提取的方程/参数/不确定点给用户确认；用户确认或授权后才继续，不擅自往下。
6. **生成求解器**：`template_lib.match_template(ocr)` 匹配模板 → `pipeline.generate_solver_with_repair(model, outdir, reference_code=模板源码)` 生成求解器（含语法错误+结果偏差双层自修复）。
7. **运行+对齐+报告**：`pipeline.run_from_model(model, outdir)` 运行求解器、与论文结果对齐、DeepSeek 归因、出报告。
8. **接受/拒绝**：给用户看对齐表 + 归因；不接受则调整推断重试（每次推断+结果记录在案）。

## API 配置（skill 根目录 .env）
- `DEEPSEEK_API_KEY`：文本推理/代码生成（DeepSeek）
- `DASHSCOPE_API_KEY`：千问视觉 OCR
- `DASHSCOPE_MODEL`：视觉模型名（如 qwen3-vl-plus）

## 关键细节
- **求解器生成必须给参考模板**：实测从零生成逻辑错误率高（4 次修复仍错），给已验证模板后 1 次即正确。
- **公式 OCR 必须走视觉 API**：本环境 agent 模型不支持图像输入。
- **模板库的已知误差**（`.meta.json` 的 validation.errors）：LLM 改编时注意，避免把模板固有偏差误判为自己改错。
- **模板库可扩充**：新验证一个模型 → 求解器 + `.meta.json`（含关键词）放入 `templates/` 即可，无需改流程。