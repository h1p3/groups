from ..groups.base import Group
from ..groups.cyclic import Cyclic
from ..groups.vector import VectorAdd


def _cyclic_grammar(n: int, max_enumerate: int = 128) -> str:
    if n <= max_enumerate:
        values = " | ".join(f'"{i}"' for i in range(n))
        return f'root ::= "element: " ({values}) "\\n"'
    return 'root ::= "element: " digit+ "\\n"\ndigit ::= [0-9]'


def _vector_grammar(dim: int) -> str:
    return (
        'root ::= "element: " "[" number ("," ws number)* "]" "\\n"\n'
        'number ::= "-"? [0-9]+ ("." [0-9]+)?\n'
        'ws ::= " "?'
    )


def element_grammar(group: Group) -> str:
    if isinstance(group, Cyclic):
        return _cyclic_grammar(group.n)
    if isinstance(group, VectorAdd):
        return _vector_grammar(group.dim)
    raise TypeError(f"element_grammar is not defined for {type(group).__name__}")


def _resolve(schema: dict, node: dict) -> dict:
    if "$ref" in node:
        name = node["$ref"].split("/")[-1]
        return _resolve(schema, schema["$defs"][name])
    return node


def _rule_name(parent: str, key: str) -> str:
    return parent + "".join(p.capitalize() for p in key.split("_"))


def schema_to_gbnf(schema: dict, root: str = "root") -> str:
    rules: dict[str, str] = {}

    def emit(name: str, node: dict) -> None:
        node = _resolve(schema, node)
        typ = node.get("type")
        if typ == "string":
            if "enum" in node:
                rules[name] = " | ".join(f'"{v}"' for v in node["enum"])
            else:
                text_name = _rule_name(name, "text")
                rules[name] = f'["] {text_name} ["]'
                rules[text_name] = '[^"\\n]*'
            return
        if typ in ("integer", "number"):
            rules[name] = '"-"? [0-9]+ ("." [0-9]+)?'
            return
        if typ == "boolean":
            rules[name] = '"true" | "false"'
            return
        if typ == "array":
            item = _resolve(schema, node.get("items", {"type": "string"}))
            item_name = _rule_name(name, "item")
            emit(item_name, item)
            rules[name] = f'"[" ws? ({item_name} ("," ws? {item_name})*)? ws? "]"'
            return
        if typ == "object":
            props = node.get("properties", {})
            pairs = []
            for key, sub in props.items():
                child = _rule_name(name, key)
                emit(child, sub)
                pairs.append(f'"{key}" ws? ":" ws? {child}')
            body = f'"," ws? '.join(pairs)
            rules[name] = f'"{{" ws? {body} ws? "}}"'
            return
        if typ is None:
            rules[name] = '"null"'
            return
        raise ValueError(f"unsupported JSON schema type {typ!r}")

    emit(root, schema)
    rules.setdefault("ws", '" "?')
    lines = [f"{name} ::= {expr}" for name, expr in rules.items()]
    return "\n".join(lines) + "\n"
