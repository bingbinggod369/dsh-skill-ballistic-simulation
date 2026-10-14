"""参考模板库：加载已验证求解器 + 按关键词自动匹配。

泛化机制：新论文进来后，用关键词匹配最接近的已验证模板，供 LLM 改编（锚点优先）。
"""
from __future__ import annotations

import glob
import json
import os

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")


def load_templates():
    """加载所有模板（元数据 + 求解器源码）。"""
    templates = []
    for meta_path in sorted(glob.glob(os.path.join(TEMPLATES_DIR, "*.meta.json"))):
        meta = json.load(open(meta_path, encoding="utf-8"))
        solver_path = os.path.join(TEMPLATES_DIR, meta.get("solver_file", ""))
        if os.path.exists(solver_path):
            meta["code"] = open(solver_path, encoding="utf-8").read()
        else:
            meta["code"] = None
        templates.append(meta)
    return templates


def _score(text: str, template: dict) -> int:
    text = text.lower()
    return sum(1 for kw in template.get("keywords", []) if kw.lower() in text)


def match_template(text: str, templates=None, min_score: int = 2):
    """返回 (最佳模板, 得分)。得分低于 min_score 时返回 (None, 得分)。"""
    templates = templates or load_templates()
    best, best_score = None, 0
    for t in templates:
        s = _score(text, t)
        if s > best_score:
            best, best_score = t, s
    if best is None or best_score < min_score:
        return None, best_score
    return best, best_score


def list_templates():
    """返回模板摘要（不含源码）。"""
    out = []
    for t in load_templates():
        out.append({"id": t["id"], "name": t["name"], "domain": t["domain"],
                    "keywords": t.get("keywords", []), "validation": t.get("validation", {})})
    return out