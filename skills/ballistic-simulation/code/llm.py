# -*- coding: utf-8 -*-
"""视觉 OCR 后端（千问 Qwen-VL / GLM-4V / GPT-4o）。

注意：文本推理（模型重建、求解器生成、结果对齐、归因）不再在这里调用 API，
由加载本 skill 的 agent 用 DSH 推理直接完成。这里只保留「看图」能力。
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request


class LLMBackend:
    def complete_vision(self, text: str, image_paths: list) -> str:
        raise NotImplementedError


class StubBackend(LLMBackend):
    def complete_vision(self, text, image_paths):
        return "[STUB] 未配置视觉模型 API（DASHSCOPE_API_KEY / GLM_API_KEY / OPENAI_API_KEY）"


class OpenAICompatibleBackend(LLMBackend):
    """OpenAI 兼容 chat/completions 视觉后端（仅用标准库 urllib，零依赖）。"""
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
    """从 skill 根目录 .env 加载 KEY=VALUE（不覆盖已存在的环境变量）。"""
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


def make_vision_backend() -> LLMBackend:
    """视觉模型后端（公式/版面 OCR）：优先 Qwen-VL，其次 GLM-4V / GPT-4o。"""
    _load_dotenv()
    if os.environ.get("DASHSCOPE_API_KEY"):
        return OpenAICompatibleBackend(
            os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            os.environ["DASHSCOPE_API_KEY"],
            os.environ.get("DASHSCOPE_MODEL", "qwen3-vl-plus"))
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