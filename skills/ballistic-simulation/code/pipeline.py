# -*- coding: utf-8 -*-
"""论文复现的机械部分：PDF 渲染 + 视觉 OCR。

文本推理（模型重建、求解器生成、结果对齐、归因、报告）由加载本 skill 的
agent 用 DSH 推理直接完成，不在这里调用文本 API。
"""
import json
import os
import subprocess

from . import llm


def render_pdf(pdf_path: str, output_dir: str):
    """用 LibreOffice Kit 渲染 PDF 为图片，返回图片路径列表。"""
    import shutil
    node = os.environ.get("LO_NODE", r"D:\deepseek\resources\runtime\primary-runtime\dependencies\node\bin\node.exe")
    cli = os.environ.get("LO_CLI", r"D:\deepseek\resources\app.asar.unpacked\dsh\node_modules\@deepseek-ai\libreoffice-kit\lib\cli.js")
    os.makedirs(output_dir, exist_ok=True)
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