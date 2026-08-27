from __future__ import annotations

import re
import uuid
from typing import Literal

from pydantic import BaseModel, Field


class FilterRule(BaseModel):
    """Правило фильтрации.

    pipeline: input   — фильтрация узлов из store при pull (по метаданным)
              output  — фильтрация текста генерации модели (язык, regex, длина)
              feedback — фильтрация tail перед подачей на следующий шаг

    mode:     text   — фильтрация на текстовом уровне (post-hoc, regex, detect_lang)
              logit  — фильтрация на уровне logits (TokenGroup, logit mask)

    type:     language — ru/en/zh/...
              topic   — тема узла (input) или ключевые слова (output/feedback)
              pattern — regex-паттерн
              tag     — произвольный тег в meta (input) или ключевое слово (output/feedback)
              length  — макс. длина текста в символах (output/feedback)

    action:   exclude — совпавшее ИСКЛЮЧАЕТСЯ
              allow   — ТОЛЬКО совпавшее допускается

    enabled:  включено ли правило
    depends_on: список ID пайплайнов, от которых зависит (пусто = независимо)
    group_dim: размерность TokenGroup (по умолчанию 64)
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    pipeline: Literal["input", "output", "feedback"] = "input"
    mode: Literal["text", "logit"] = "text"
    type: Literal["language", "topic", "pattern", "tag", "length"]
    action: Literal["exclude", "allow"]
    value: str
    description: str = ""
    enabled: bool = True
    depends_on: list[str] = Field(default_factory=list)
    group_dim: int = 64

    # --- matching for INPUT (node metadata) ---

    def matches_node(self, node) -> bool:
        meta = getattr(node, "meta", None) or {}
        text = getattr(node, "text", "")

        if self.type == "language":
            return meta.get("language") == self.value
        if self.type == "topic":
            topics = meta.get("topics", [])
            if isinstance(topics, str):
                topics = [topics]
            return self.value in topics
        if self.type == "pattern":
            try:
                return bool(re.search(self.value, text))
            except re.error:
                return False
        if self.type == "tag":
            tags = meta.get("tags", [])
            if isinstance(tags, str):
                tags = [tags]
            return self.value in tags
        return False

    # --- matching for OUTPUT / FEEDBACK (text) ---

    def matches_text(self, text: str) -> bool:
        if self.type == "language":
            return _detect_lang(text) == self.value
        if self.type == "pattern":
            try:
                return bool(re.search(self.value, text))
            except re.error:
                return False
        if self.type == "topic":
            return self.value.lower() in text.lower()
        if self.type == "tag":
            return self.value.lower() in text.lower()
        if self.type == "length":
            try:
                limit = int(self.value)
            except ValueError:
                return False
            return len(text) > limit
        return False

    def is_node_allowed(self, node) -> bool:
        if self.action == "exclude":
            return not self.matches_node(node)
        return self.matches_node(node)

    def is_text_allowed(self, text: str) -> bool:
        if self.action == "exclude":
            return not self.matches_text(text)
        return self.matches_text(text)


# --- language detection (simple Unicode range) ---

_ZH_RANGES = [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF)]
_RU_MIN, _RU_MAX = 0x0400, 0x04FF


def _detect_lang(text: str) -> str:
    zh = 0
    ru = 0
    en = 0
    for ch in text:
        cp = ord(ch)
        if any(lo <= cp <= hi for lo, hi in _ZH_RANGES):
            zh += 1
        elif _RU_MIN <= cp <= _RU_MAX:
            ru += 1
        elif ch.isalpha() and ch.isascii():
            en += 1
    if zh > 0:
        return "zh"
    total = ru + en
    if total == 0:
        return ""
    if ru / total > 0.5:
        return "ru"
    if en / total > 0.5:
        return "en"
    return ""


def passes_filters(node, rules: list[FilterRule]) -> bool:
    """Проверяет, проходит ли узел ВСЕ активные input-правила."""
    active = [r for r in rules if r.enabled and r.pipeline == "input"]
    if not active:
        return True
    allow_rules = [r for r in active if r.action == "allow"]
    exclude_rules = [r for r in active if r.action == "exclude"]
    if allow_rules:
        if not any(r.is_node_allowed(node) for r in allow_rules):
            return False
    for r in exclude_rules:
        if not r.is_node_allowed(node):
            return False
    return True


def passes_output_filters(text: str, rules: list[FilterRule], all_rules: list[FilterRule] | None = None) -> bool:
    """Проверяет, проходит ли текст генерации все активные output-правила.

    Учитывает зависимости: если depends_on содержит pipeline ID,
    зависимый пайплайн активен только когда все указанные пайплайны имеют хотя бы одно активное правило.
    """
    active = [r for r in rules if r.enabled and r.pipeline == "output"]
    if not active:
        return True
    if all_rules is None:
        all_rules = rules
    for r in active:
        if r.depends_on:
            for dep_id in r.depends_on:
                dep_rules = [x for x in all_rules if x.enabled and x.pipeline == dep_id]
                if not dep_rules:
                    return True
        if not r.is_text_allowed(text):
            return False
    return True


def passes_feedback_filters(text: str, rules: list[FilterRule], all_rules: list[FilterRule] | None = None) -> bool:
    """Проверяет, проходит ли tail все активные feedback-правила."""
    active = [r for r in rules if r.enabled and r.pipeline == "feedback"]
    if not active:
        return True
    if all_rules is None:
        all_rules = rules
    for r in active:
        if r.depends_on:
            for dep_id in r.depends_on:
                dep_rules = [x for x in all_rules if x.enabled and x.pipeline == dep_id]
                if not dep_rules:
                    return True
        if not r.is_text_allowed(text):
            return False
    return True
