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


