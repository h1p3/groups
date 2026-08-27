from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .grammar.gbnf import element_grammar, schema_to_gbnf
from .grammar.schema import FinalAnswer

if TYPE_CHECKING:
    from .context.aggregate import ContextState
    from .groups.base import Group
    from .groups.filter import FilterRule

LANG_NAMES = {"ru": "русском", "en": "английском", "zh": "китайском", "de": "немецком",
              "fr": "французском", "es": "испанском", "ja": "японском"}


def _lang_directive_from_filters(filters: list["FilterRule"], pipeline: str = "input") -> str | None:
    exclude_langs = set()
    allow_langs = set()
    for f in filters:
        if f.type == "language" and f.pipeline == pipeline:
            if f.action == "exclude":
                exclude_langs.add(f.value)
            else:
                allow_langs.add(f.value)
    parts = []
    if exclude_langs:
        names = ", ".join(LANG_NAMES.get(l, l) for l in sorted(exclude_langs))
        parts.append(f"Не используй языки: {names}.")
    if allow_langs:
        names = ", ".join(LANG_NAMES.get(l, l) for l in sorted(allow_langs))
        parts.append(f"Отвечай ТОЛЬКО на: {names}.")
    return " ".join(parts) if parts else None

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "prompts"


class PromptSet:
    def __init__(self, prompts_dir: str | Path = DEFAULT_DIR, schema=FinalAnswer):
        self.dir = Path(prompts_dir)
        self.schema = schema
        self.system = (self.dir / "system.md").read_text(encoding="utf-8")
        self._env = Environment(
            loader=FileSystemLoader(str(self.dir)),
            autoescape=select_autoescape(enabled_extensions=("html", "htm", "xml")),
        )

    def _fragments(self, store, ids):
        out = []
        for node_id in ids:
            node = store.get(node_id)
            if node is not None:
                out.append({"id": node.node_id, "text": node.text})
        return out

    def step(self, group: "Group", task: str, state: "ContextState", tail: str, store, fragments_ids, filters=None) -> str:
        template = self._env.get_template("cot.jinja")
        system = self.system
        if filters:
            lang_directive = _lang_directive_from_filters(filters, "output")
            if not lang_directive:
                lang_directive = _lang_directive_from_filters(filters, "input")
            if lang_directive:
                system = system + "\n\n" + lang_directive
        return template.render(
            system=system,
            task=task,
            h=group.compact(state.h),
            active_count=len(state),
            fragments=self._fragments(store, fragments_ids),
            tail=tail[-1200:],
        )

    def final(self, group: "Group", task: str, state: "ContextState", filters=None) -> str:
        template = self._env.get_template("final.jinja")
        system = self.system
        if filters:
            lang_directive = _lang_directive_from_filters(filters, "output")
            if not lang_directive:
                lang_directive = _lang_directive_from_filters(filters, "input")
            if lang_directive:
                system = system + "\n\n" + lang_directive
        return template.render(
            system=system,
            task=task,
            h=group.compact(state.h),
            active_count=len(state),
        )

    def step_grammar(self, group: "Group") -> str | None:
        return element_grammar(group)

    def final_grammar(self) -> str | None:
        return schema_to_gbnf(self.schema.model_json_schema())
