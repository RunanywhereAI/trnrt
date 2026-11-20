"""A request's output constraint (engine/grammar.py GrammarSpec) from the fields the OpenAI API, vLLM and SGLang use:

| Field | From | Constraint |
|---|---|---|
| `response_format` `{"type": "json_object"}` | OpenAI | a JSON object (SGLang's schema `{"type": "object"}`) |
| `response_format` `{"type": "json_schema", "json_schema": {"schema": ...}}` | OpenAI | the schema |
| `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` | vLLM (older) | schema, regex, one of the strings, EBNF |
| `structured_outputs` `{json, regex, choice, grammar, json_object}` | vLLM | the same |
| `json_schema` (a string), `regex`, `ebnf` | SGLang | schema, regex, EBNF |

At most one constraint per request, as SGLang requires ("Only one of json_schema, regex, ebnf, or structural_tag can be
set"). A malformed field is a ValueError, which the server answers with 400; a constraint xgrammar cannot compile is
refused by the engine when the request is added (also a 400).
"""

from __future__ import annotations

import json

from ..engine.grammar import GrammarSpec

OBJECT = '{"type": "object"}'
_VLLM_KEYS = ("json", "regex", "choice", "grammar", "json_object")
# vLLM's other StructuredOutputsParams fields (vllm/sampling_params.py), accepted only at their defaults: a client that
# sends the whole dataclass sends them. Any other key is refused whatever its value.
_VLLM_NEUTRAL: dict[str, tuple[object, ...]] = {
    "disable_any_whitespace": (None, False),
    "disable_additional_properties": (None, False),
    "whitespace_pattern": (None,),
    "structural_tag": (None,),
}


def _schema_text(v, where: str) -> str:
    if isinstance(v, str):
        try:
            json.loads(v)
        except ValueError:
            raise ValueError(f"{where} must be a JSON schema (an object, or its JSON text)") from None
        return v
    if isinstance(v, (dict, bool)):
        return json.dumps(v)
    raise ValueError(f"{where} must be a JSON schema (an object, or its JSON text)")


def _text(v, where: str) -> str:
    if not isinstance(v, str) or not v:
        raise ValueError(f"{where} must be a non-empty string")
    return v


def _choice(v, where: str) -> GrammarSpec:
    if not isinstance(v, list) or not v or not all(isinstance(x, str) and x for x in v):
        raise ValueError(f"{where} must be a non-empty list of non-empty strings")
    return GrammarSpec.choice(v)


def _response_format(rf) -> GrammarSpec | None:
    if rf is None:
        return None
    if not isinstance(rf, dict):
        raise ValueError("response_format must be an object")
    kind = rf.get("type", "text")
    if kind == "text":
        return None
    if kind == "json_object":
        return GrammarSpec("json_schema", OBJECT)
    if kind == "json_schema":
        js = rf.get("json_schema")
        if js is None and "schema" in rf:  # SGLang also takes the schema at the top of response_format
            js = {"schema": rf["schema"]}
        if not isinstance(js, dict) or js.get("schema") is None:
            raise ValueError('response_format json_schema needs "json_schema": {"name": ..., "schema": {...}}')
        return GrammarSpec("json_schema", _schema_text(js["schema"], "response_format.json_schema.schema"))
    raise ValueError(f"response_format type {kind!r} is not supported: text, json_object or json_schema")


def _vllm(so) -> list[GrammarSpec]:
    if so is None:
        return []
    if not isinstance(so, dict):
        raise ValueError("structured_outputs must be an object")
    unknown = sorted(k for k, v in so.items() if k not in _VLLM_KEYS and v not in _VLLM_NEUTRAL.get(k, ()))
    if unknown:
        raise ValueError(f"structured_outputs.{unknown[0]} is not supported: {', '.join(_VLLM_KEYS)}")
    out = []
    if so.get("json") is not None:
        out.append(GrammarSpec("json_schema", _schema_text(so["json"], "structured_outputs.json")))
    if so.get("regex") is not None:
        out.append(GrammarSpec("regex", _text(so["regex"], "structured_outputs.regex")))
    if so.get("choice") is not None:
        out.append(_choice(so["choice"], "structured_outputs.choice"))
    if so.get("grammar") is not None:
        out.append(GrammarSpec("ebnf", _text(so["grammar"], "structured_outputs.grammar")))
    json_object = so.get("json_object")
    if json_object is not None and not isinstance(json_object, bool):  # the string "false" must not turn it on
        raise ValueError("structured_outputs.json_object must be true or false")
    if json_object:
        out.append(GrammarSpec("json_schema", OBJECT))
    return out


def grammar_spec(body: dict) -> GrammarSpec | None:
    """The request's constraint, or None. ValueError for a malformed field or more than one constraint."""
    specs: list[tuple[str, GrammarSpec]] = []
    rf = _response_format(body.get("response_format"))
    if rf is not None:
        specs.append(("response_format", rf))
    specs += [("structured_outputs", s) for s in _vllm(body.get("structured_outputs"))]
    fields = (
        ("guided_json", lambda v, k: GrammarSpec("json_schema", _schema_text(v, k))),
        ("guided_regex", lambda v, k: GrammarSpec("regex", _text(v, k))),
        ("guided_choice", _choice),
        ("guided_grammar", lambda v, k: GrammarSpec("ebnf", _text(v, k))),
        ("json_schema", lambda v, k: GrammarSpec("json_schema", _schema_text(v, k))),
        ("regex", lambda v, k: GrammarSpec("regex", _text(v, k))),
        ("ebnf", lambda v, k: GrammarSpec("ebnf", _text(v, k))),
    )
    for key, make in fields:
        v = body.get(key)
        if v is not None:
            specs.append((key, make(v, key)))
    if len(specs) > 1:
        names = ", ".join(dict.fromkeys(k for k, _ in specs))
        raise ValueError(f"at most one output constraint per request, got {names}")
    return specs[0][1] if specs else None
