"""LLM 接入层：可插拔后端 + 锚点优先提示词构建。

对应架构文档 3.2「锚点优先」与 3.3「先问后做」：
- 提示词把检索到的 golden code、方程、参数作为唯一事实锚点注入；
- 明确要求模型：引用依据、缺失即提问、不自作主张、输出可复现。

后端可插拔：环境变量配置真实 API（DeepSeek/OpenAI 兼容），否则用离线 stub。
"""
from __future__ import annotations

import base64
import importlib
import json
import os
import urllib.request


# ============ 后端接口 ============

class LLMBackend:
    """LLM 后端接口：complete(messages) -> 文本。"""
    def complete(self, messages) -> str:
        raise NotImplementedError


class StubBackend(LLMBackend):
    """离线桩：不调用任何 API，返回提示词本身，用于无 key 时演示与测试。"""
    def complete(self, messages) -> str:
        out = ["[STUB 后端] 未配置 LLM API，以下为将发送给模型的提示词：", ""]
        for m in messages:
            out.append("【%s】" % m["role"])
            out.append(m["content"])
            out.append("")
        return "\n".join(out)


class OpenAICompatibleBackend(LLMBackend):
    """OpenAI/DeepSeek 兼容 chat/completions 后端（仅用标准库 urllib，零依赖）。"""
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def _post(self, messages) -> str:
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps({"model": self.model, "messages": messages}).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.api_key},
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    def complete(self, messages) -> str:
        return self._post(messages)

    def complete_vision(self, text: str, image_paths: list) -> str:
        """发送文本 + 图像（base64 data URI）给视觉模型。"""
        content = [{"type": "text", "text": text}]
        for p in image_paths:
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + b64}})
        return self._post([{"role": "user", "content": content}])


def _load_dotenv():
    """从项目根目录 .env 加载 KEY=VALUE（不覆盖已存在的环境变量）。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, ".env")
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def make_backend() -> LLMBackend:
    """按环境变量构建后端（无 key 时回退 stub）。"""
    _load_dotenv()
    if os.environ.get("DEEPSEEK_API_KEY"):
        return OpenAICompatibleBackend(
            os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            os.environ["DEEPSEEK_API_KEY"],
            os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAICompatibleBackend(
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            os.environ["OPENAI_API_KEY"],
            os.environ.get("OPENAI_MODEL", "gpt-4o"))
    if os.environ.get("LLM_API_KEY"):
        return OpenAICompatibleBackend(
            os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            os.environ["LLM_API_KEY"],
            os.environ.get("LLM_MODEL", "deepseek-chat"))
    return StubBackend()


def make_vision_backend() -> LLMBackend:
    """视觉模型后端（公式/版面 OCR）：优先 Qwen-VL，其次 GLM-4V / GPT-4o。"""
    _load_dotenv()
    if os.environ.get("DASHSCOPE_API_KEY"):
        return OpenAICompatibleBackend(
            os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            os.environ["DASHSCOPE_API_KEY"],
            os.environ.get("DASHSCOPE_MODEL", "qwen-vl-max"))
    if os.environ.get("GLM_API_KEY"):
        return OpenAICompatibleBackend(
            os.environ.get("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
            os.environ["GLM_API_KEY"],
            os.environ.get("GLM_MODEL", "glm-4v"))
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAICompatibleBackend(
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            os.environ["OPENAI_API_KEY"],
            os.environ.get("OPENAI_MODEL", "gpt-4o"))
    return StubBackend()


def vision_ocr(image_paths, instruction="识别图中的全部文字与公式，公式用 LaTeX 表示，保留公式编号。"):
    """用视觉模型识别页面图像，返回识别的文本（含 LaTeX 公式）。"""
    backend = make_vision_backend()
    if isinstance(backend, StubBackend):
        return "[STUB] 未配置视觉模型 API（DASHSCOPE_API_KEY / GLM_API_KEY / OPENAI_API_KEY）"
    return backend.complete_vision(instruction, image_paths)


# ============ 领域分类 ============

_INTERIOR_KW = ["内弹道", "膛压", "燃速", "火药", "装药", "炮", "弹丸", "膛内", "身管", "无后座", "无后坐", "高射炮", "膛线"]
_EXTERIOR_KW = ["外弹道", "弹道", "射程", "阻力", "空气", "弹道系数", "质点弹道", "落点", "射角", "弹形", "6dof", "4dof", "马赫", "风"]

def classify_domain(requirement: str) -> str:
    """简易领域分类：内/外弹道关键词打分。"""
    text = requirement.lower()
    ei = sum(1 for k in _INTERIOR_KW if k in text)
    ee = sum(1 for k in _EXTERIOR_KW if k in text)
    if ei >= ee and ei > 0:
        return "interior_ballistics"
    if ee > ei and ee > 0:
        return "exterior_ballistics"
    return "interior_ballistics"  # 默认内弹道（当前唯一有 golden 的领域）


# ============ 提示词构建 ============

SYSTEM_RULES = """你是弹道学（内弹道/外弹道）专用代码生成智能体。

必须遵守：
1. 锚点优先：下方给出的参考代码（已验证 golden code）与方程库是你唯一的事实依据；
   生成的代码必须基于参考代码改造，不得凭空编写公式或参数。
2. 引用依据：每个关键公式/参数在代码注释中标注方程 id 与出处（书名+页码）。
3. 先问后做：遇到缺失参数、缺失方程或不确定假设，必须先向用户提问，不得擅自取值；
   仅在用户明确授权"由你决定"时，才可用知识库参考值补齐，并在报告中标注。
4. 可复现：输出完整方程集（闭环）+ 参数表（逐条溯源）+ 求解配置。
5. 单位用 SI，输出结果时换算到领域习惯单位（MPa、mm、m/s）。"""


def _load_template_source(entry_point: str) -> str:
    """按模块名直接读源码文件，不 import 模块（避免触发 numpy 等依赖、避免副作用）。"""
    mod_name = entry_point.split(":")[0]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, *mod_name.split(".")) + ".py"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_messages(requirement: str, domain: str, hits: dict, template) -> list:
    """构建 messages（system + user），把事实锚点注入 user 上下文。"""
    user_parts = []
    user_parts.append("## 用户需求")
    user_parts.append(requirement)
    user_parts.append("")
    user_parts.append("## 领域")
    user_parts.append(domain)
    user_parts.append("")
    user_parts.append("## 参考代码（已验证 golden code，请基于它改造，不要另起炉灶）")
    user_parts.append("- id: %s" % template.id)
    user_parts.append("- 标题: %s" % template.title)
    user_parts.append("- 出处: %s" % json.dumps(template.source, ensure_ascii=False))
    user_parts.append("- 假设: %s" % "；".join(template.assumptions))
    user_parts.append("")
    user_parts.append("## 方程库（事实锚点，逐条可用）")
    for e in hits["equations"]:
        user_parts.append("- [%s] %s : $%s$ （%s）" % (e.id, e.name, e.latex, e.source.get("book", "")))
    user_parts.append("")
    user_parts.append("## 参数库（量级参考）")
    for p in hits["parameters"]:
        user_parts.append("- [%s] %s : 典型 %s，范围 %s（%s）" % (
            p.id, p.name, json.dumps(p.typical, ensure_ascii=False),
            json.dumps(p.value_range, ensure_ascii=False), p.source.get("book", "")))
    user_parts.append("")
    user_parts.append("## 参考代码源码")
    user_parts.append("```python")
    user_parts.append(_load_template_source(template.entry_point))
    user_parts.append("```")
    user_parts.append("")
    user_parts.append("## 任务")
    user_parts.append("按用户需求，基于参考代码生成可运行的新代码。")
    user_parts.append("- 若需求中的参数/方程已齐全：直接生成代码，每个公式标注方程 id + 出处。")
    user_parts.append("- 若缺失参数/方程/假设：先输出【需要用户确认的问题】清单，不要擅自编造。")
    user_parts.append("只输出代码（或问题清单），附简短说明。")
    return [
        {"role": "system", "content": SYSTEM_RULES},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


# ============ 编排：需求 -> 分类 -> 检索 -> 提示词 -> LLM ============

def generate_code(requirement: str, backend: LLMBackend = None, gc=None, eqs=None, params=None):
    """LLM 生成编排入口。返回 {domain, hits, template, messages, response, backend_name}。"""
    from . import kb
    backend = backend or make_backend()
    domain = classify_domain(requirement)
    hits = kb.retrieve(domain, requirement, gc=gc, eqs=eqs, params=params)
    if not hits["golden_codes"]:
        return {"ok": False, "domain": domain, "error": "未检索到匹配模板（domain=%s）" % domain,
                "hits": hits, "messages": [], "response": "", "backend_name": type(backend).__name__}
    template = hits["golden_codes"][0]
    messages = build_messages(requirement, domain, hits, template)
    response = backend.complete(messages)
    return {"ok": True, "domain": domain, "template": template, "hits": hits,
            "messages": messages, "response": response, "backend_name": type(backend).__name__}