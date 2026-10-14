"""全自动论文复现流水线（独立于助手）。

流程：PDF → 视觉OCR(千问) → 模型重建(DeepSeek) → 确认门 → 求解器生成(DeepSeek)
      → 仿真 → 对齐 → 归因(DeepSeek) → 报告。
泛化关键：模型重建与求解器生成都由 LLM 完成，不硬编码具体弹道模型。
"""
from __future__ import annotations

import json
import re
import os
import subprocess
import importlib.util
import time

from . import llm


def _extract_json(text: str):
    """从 LLM 输出中稳健提取 JSON（优先代码块，支持对象与数组）。"""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidate = (m.group(1) if m else text).strip()
    try:
        return json.loads(candidate)
    except Exception:
        pass
    # 兜底：按首个出现的括号类型取首尾
    fb = candidate.find("{"); fbr = candidate.find("[")
    op, cl = ("[", "]") if (fbr >= 0 and (fb < 0 or fbr < fb)) else ("{", "}")
    start = candidate.find(op); end = candidate.rfind(cl)
    if start >= 0 and end > start:
        return json.loads(candidate[start:end + 1])
    raise ValueError("无法从输出提取 JSON")


# ============ 提示词 ============

_SYS_RECON = "你是弹道学论文复现专家。从论文 OCR 文本中提取结构化数学模型，输出严格 JSON。"

_RECON_PROMPT = """请从下面的论文 OCR 文本中提取数学模型，输出 JSON，字段：

- assumptions: 基本假设（字符串数组）
- equations: 所有方程，每条 {num: 公式编号如"(4)", latex: LaTeX, meaning: 物理含义, type: "ode" 或 "algebraic"}（含 d/dt 的归 ode）
- parameters: 所有参数，每条 {symbol: 符号, meaning: 含义, value: 数值(数字或字符串), unit: 单位}
- results: 论文给出的数值结果，每条 {key: 标准英文键, name: 中文名, value: 数值, unit: 单位}。key 只能从这些选：t_start(起运动时刻)、t_muzzle(出筒时刻/弹射总时间)、v_muzzle(出筒速度/初速)、max_overload(最大过载)、avg_overload(平均过载)、max_pressure(最大压强/膛压)、muzzle_pressure(炮口压强)。没有对应项就跳过。

要求：只输出 JSON，不要其它文字；缺失项用 null 或空数组。

OCR 文本：
"""

_SYS_SOLVER = "你是数值计算专家。根据给定的方程组编写完整可运行的 Python 求解器。"

_SOLVER_PROMPT = """请根据下面的弹道模型 JSON，编写一个**完整可独立运行**的 Python 求解脚本。要求：

1. 用 numpy + 手写 RK4（不要 scipy）。
2. 确定状态变量（ODE 中的待积分变量）与代数中间量；给出微分方程右端函数。
3. 参数在脚本顶部用字典给出（取模型 JSON 里 parameters 的 value）。
4. 初值由参数的"初始"值确定。
5. 正确处理分段函数、临界/亚临界等条件。
6. **某些参数是时间的函数**（如发射阀流通面积 S(t)=a·t² 且封顶到最大值），必须在代码里实现这个时变关系，不要当成常数。
7. 脚本末尾用 `if __name__ == "__main__":` 运行，并**必须**用一行输出结构化结果：
   `print("RESULT_JSON=" + json.dumps({"t_start":..,"t_muzzle":..,"v_muzzle":..,"max_overload":..}))`（字段英文，单位 SI）。
8. 只输出 Python 代码（用 ```python 包裹），不要解释。

模型 JSON：
"""

_SYS_INFER = "你是弹道学专家，负责审查模型中的不确定点。"

_INFER_PROMPT = """以下是提取的弹道模型 JSON。请找出其中"不确定/有歧义/需用户确认"的点（如定义不明确、符号歧义、分段条件不清、参数缺失）。对每个点输出 {point: 描述, inference: 建议的推断, basis: 依据}。输出 JSON 数组。

模型 JSON：
"""

_REPAIR_PROMPT = """你生成的求解器代码运行时报错。请修复错误，重新输出完整代码（用 ```python 包裹）。只输出代码。

原代码：
{code}

错误信息：
{error}
"""

_SYS_ATTR = "你是弹道学专家，负责分析仿真与论文结果的偏差。"

_ATTR_PROMPT = """以下是仿真结果与论文数值的对比。请分析偏差规律，给出最可能的原因（参数取值、方程细节、数值方法、单位约定等），并给出调整建议。直接输出分析文字。

对比：
"""


# ============ 各步骤 ============

def reconstruct_model(ocr_text: str, backend=None) -> dict:
    backend = backend or llm.make_backend()
    resp = backend.complete([{"role": "system", "content": _SYS_RECON},
                             {"role": "user", "content": _RECON_PROMPT + ocr_text}])
    return _extract_json(resp)


def infer_uncertainties(model: dict, backend=None) -> list:
    backend = backend or llm.make_backend()
    resp = backend.complete([{"role": "system", "content": _SYS_INFER},
                             {"role": "user", "content": _INFER_PROMPT + json.dumps(model, ensure_ascii=False)}])
    try:
        return _extract_json(resp)
    except Exception:
        return [{"point": "解析失败", "inference": None, "basis": resp}]


def generate_solver(model: dict, backend=None, reference_code: str = None) -> str:
    backend = backend or llm.make_backend()
    prompt = _SOLVER_PROMPT + json.dumps(model, ensure_ascii=False)
    if reference_code:
        prompt += "\n\n## 参考实现（已正确验证的同类求解器，请基于它改编方程与结构，不要另起炉灶）\n```python\n" + reference_code + "\n```"
    resp = backend.complete([{"role": "system", "content": _SYS_SOLVER},
                             {"role": "user", "content": prompt}])
    m = re.search(r"```(?:python)?\s*(.*?)```", resp, re.S)
    return (m.group(1) if m else resp).strip()


def attribute(comparison: str, backend=None) -> str:
    backend = backend or llm.make_backend()
    return backend.complete([{"role": "system", "content": _SYS_ATTR},
                             {"role": "user", "content": _ATTR_PROMPT + comparison}])


def _parse_result_json(stdout: str):
    m = re.search(r"RESULT_JSON\s*=\s*(\{.*?\})", stdout, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def _reference_values(model: dict):
    refs = {}
    for r in model.get("results", []):
        v = r.get("value")
        if not isinstance(v, (int, float)):
            continue
        key = r.get("key")
        if key:
            refs[key] = v
            continue
        # 兜底：按中文名关键词（兼容旧格式）
        name = str(r.get("name", ""))
        if ("运动" in name or "起" in name) and "时刻" in name:
            refs.setdefault("t_start", v)
        elif "时刻" in name or "总时间" in name:
            refs.setdefault("t_muzzle", v)
        elif "速度" in name or "初速" in name:
            refs.setdefault("v_muzzle", v)
    return refs


def _rel_error(res: dict, refs: dict):
    """返回最大相对误差（仅比较两者共有的键）。"""
    keys = [k for k in res if k in refs and refs[k]]
    if not keys:
        return None
    return max(abs(res[k] - refs[k]) / abs(refs[k]) for k in keys)


def _repair(code: str, error: str, backend):
    resp = backend.complete([
        {"role": "system", "content": _SYS_SOLVER},
        {"role": "user", "content": _REPAIR_PROMPT.format(code=code, error=error[-3000:])},
    ])
    m = re.search(r"```(?:python)?\s*(.*?)```", resp, re.S)
    return (m.group(1) if m else resp).strip()


def generate_solver_with_repair(model: dict, workdir: str, backend=None, max_attempts=4, reference_code=None):
    """生成求解器 + 运行 + 双层自修复（语法错误 + 结果偏差），返回 (代码, stdout, 尝试次数)。"""
    backend = backend or llm.make_backend()
    os.makedirs(workdir, exist_ok=True)
    code = generate_solver(model, backend, reference_code=reference_code)
    refs = _reference_values(model)
    stdout = ""
    attempts = 1
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        stdout = run_solver_code(code, workdir)
        if "SyntaxError" in stdout or "Traceback" in stdout:
            code = _repair(code, stdout, backend)
            continue
        res = _parse_result_json(stdout)
        if res is None:
            code = _repair(code, "缺少 RESULT_JSON 输出，请按要求补充", backend)
            continue
        err = _rel_error(res, refs) if refs else None
        if err is None or err <= 0.30:
            return code, stdout, attempts
        code = _repair(code, "结果与论文参考值偏差 %.0f%%（你的=%s，论文=%s），请检查逻辑错误（时变参数、启动条件、方程实现等）" % (err * 100, res, refs), backend)
    return code, stdout, attempts


# ============ 执行生成的求解器 ============

def run_solver_code(code: str, workdir: str) -> str:
    """把生成的求解器写到临时文件并运行，返回 stdout。"""
    path = os.path.join(workdir, "_generated_solver.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    r = subprocess.run([os.sys.executable, path], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=300, cwd=workdir)
    return (r.stdout or "") + ("\n[STDERR]\n" + r.stderr if r.stderr else "")


# ============ 从 OCR 文本全流程 ============

def run_from_ocr(ocr_text: str, output_dir: str, backend=None, reference_code: str = None) -> dict:
    """从 OCR 文本出发的全流程：重建 → 推断 → 生成求解器(自修复) → 运行 → 对齐 → 归因 → 报告。"""
    os.makedirs(output_dir, exist_ok=True)
    backend = backend or llm.make_backend()
    out = {"output_dir": output_dir}

    # 0. 自动匹配参考模板（未显式指定时）
    if reference_code is None:
        from . import template_lib
        tpl, score = template_lib.match_template(ocr_text)
        out["template_matched"] = tpl["id"] if tpl else None
        out["template_score"] = score
        if tpl and tpl.get("code"):
            reference_code = tpl["code"]

    # 1. 模型重建
    model = reconstruct_model(ocr_text, backend)
    out["model"] = model
    with open(os.path.join(output_dir, "model.json"), "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False, indent=2)

    # 2. 不确定点推断
    unc = infer_uncertainties(model, backend)
    out["uncertainties"] = unc

    # 3. 生成求解器（含双层自修复）
    code, stdout, attempts = generate_solver_with_repair(model, output_dir, backend, reference_code=reference_code)
    out["solver_code"] = code
    out["solver_stdout"] = stdout
    out["attempts"] = attempts
    with open(os.path.join(output_dir, "generated_solver.py"), "w", encoding="utf-8") as f:
        f.write(code)

    # 4. 对齐：解析结果 vs 论文参考值
    res = _parse_result_json(stdout)
    refs = _reference_values(model)
    alignment = []
    if res and refs:
        for k in sorted(set(res) & set(refs)):
            rv, pv = res[k], refs[k]
            rel = abs(rv - pv) / abs(pv)
            alignment.append({"key": k, "sim": rv, "paper": pv, "rel_err": round(rel, 4)})
    out["alignment"] = alignment
    out["result_json"] = res
    out["reference_values"] = refs

    # 5. 归因（若有偏差）
    if alignment:
        cmp_text = "\n".join("| %s | %.4g | %.4g | %.1f%% |" % (a["key"], a["sim"], a["paper"], a["rel_err"] * 100) for a in alignment)
        out["attribution"] = attribute(cmp_text, backend)
    else:
        out["attribution"] = "（无对齐数据）"

    # 6. 报告
    report = _build_report(model, unc, attempts, alignment, out["attribution"])
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    out["report"] = report
    return out


def _build_report(model, uncertainties, attempts, alignment, attribution) -> str:
    L = ["# 论文复现报告（自动生成）", ""]
    L.append("- 假设 %d 条，方程 %d 条，参数 %d 个" % (
        len(model.get("assumptions", [])), len(model.get("equations", [])), len(model.get("parameters", []))))
    L.append("- 求解器生成尝试次数：%d" % attempts)
    L.append("")
    L.append("## 不确定点（LLM 推断，需人工复核）")
    L.append("")
    if uncertainties:
        for u in uncertainties:
            if isinstance(u, dict):
                L.append("- %s → 推断：%s（依据：%s）" % (u.get("point", ""), u.get("inference", ""), u.get("basis", "")))
    else:
        L.append("- （无）")
    L.append("")
    L.append("## 结果对齐（仿真 vs 论文）")
    L.append("")
    if alignment:
        L.append("| 指标 | 仿真 | 论文 | 相对误差 |")
        L.append("|---|---|---|---|")
        for a in alignment:
            L.append("| %s | %.4g | %.4g | %.1f%% |" % (a["key"], a["sim"], a["paper"], a["rel_err"] * 100))
    else:
        L.append("（无对齐数据）")
    L.append("")
    L.append("## 偏差归因")
    L.append("")
    L.append(attribution)
    return "\n".join(L)

# ============ PDF 渲染 + OCR ============

def render_pdf(pdf_path: str, output_dir: str):
    """用 LibreOffice Kit 渲染 PDF 为图片，返回图片路径列表。"""
    import shutil
    node = os.environ.get("LO_NODE", r"D:\deepseek\resources\runtime\primary-runtime\dependencies\node\bin\node.exe")
    cli = os.environ.get("LO_CLI", r"D:\deepseek\resources\app.asar.unpacked\dsh\node_modules\@deepseek-ai\libreoffice-kit\lib\cli.js")
    os.makedirs(output_dir, exist_ok=True)  # 先建父目录，否则渲染时 mkdir pages 会 ENOENT
    pages_dir = os.path.join(output_dir, "pages")
    if os.path.exists(pages_dir):
        shutil.rmtree(pages_dir, ignore_errors=True)
    r = subprocess.run([node, cli, "render", "--input", pdf_path, "--output-dir", pages_dir, "--dpi", "200"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
    try:
        manifest = json.loads(r.stdout)
        return [im["path"] for im in manifest.get("images", [])]
    except Exception:
        raise RuntimeError("PDF 渲染失败: " + (r.stdout or "") + (r.stderr or ""))


def ocr_pages(image_paths, progress_cb=None) -> str:
    """用视觉模型逐页 OCR，返回拼接文本。"""
    texts = []
    for i, p in enumerate(image_paths, 1):
        if progress_cb:
            progress_cb("OCR 第 %d/%d 页…" % (i, len(image_paths)))
        texts.append(llm.vision_ocr([p]))
    return "\n\n".join(texts)


def run_from_model(model: dict, output_dir: str, backend=None, reference_code=None) -> dict:
    """给定已重建模型，执行：生成求解器(自修复) → 运行 → 对齐 → 归因 → 报告。"""
    os.makedirs(output_dir, exist_ok=True)
    backend = backend or llm.make_backend()
    out = {"output_dir": output_dir, "model": model}

    code, stdout, attempts = generate_solver_with_repair(model, output_dir, backend, reference_code=reference_code)
    out["solver_code"] = code
    out["solver_stdout"] = stdout
    out["attempts"] = attempts
    with open(os.path.join(output_dir, "generated_solver.py"), "w", encoding="utf-8") as f:
        f.write(code)

    res = _parse_result_json(stdout)
    refs = _reference_values(model)
    alignment = []
    if res and refs:
        for k in sorted(set(res) & set(refs)):
            rv, pv = res[k], refs[k]
            rel = abs(rv - pv) / abs(pv)
            alignment.append({"key": k, "sim": rv, "paper": pv, "rel_err": round(rel, 4)})
    out["alignment"] = alignment
    out["result_json"] = res
    out["reference_values"] = refs

    if alignment:
        cmp_text = "\n".join("| %s | %.4g | %.4g | %.1f%% |" % (a["key"], a["sim"], a["paper"], a["rel_err"] * 100) for a in alignment)
        out["attribution"] = attribute(cmp_text, backend)
    else:
        out["attribution"] = "（无对齐数据）"

    report = _build_report(model, [], attempts, alignment, out["attribution"])
    with open(os.path.join(output_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    out["report"] = report
    return out