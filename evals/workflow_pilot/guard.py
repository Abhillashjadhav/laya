"""Lossless Choice preflight for one pinned Laya SDK; no model imports."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

SDK_COMMIT = "d113dca2512fb3eaca313534bc54c7162d87c1d4"
MODEL_REPO = "convaiinnovations/laya"
MODEL_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"
SOURCE_BLOBS = {
    "laya/common.py": "326b76a45e488f4143a0351b9531566272c759e4",
    "laya/agent.py": "ae03574dca69cb568a33c9c0dcc96ac70b94ecc6",
    "laya/__init__.py": "1f1acb4fb27d2818c1f4ea2e5f6848e725eef634",
    "laya/email.py": "e2e41faff0600f0f73ad4d0af5af63644161cc3d",
    "laya/lang.py": "cf26b3e94283c55f72861f8887e6d3b7daf204ae",
    "laya/presets.py": "ea2efd385e8135a190708787c84d7beae33f9f87",
    "laya/router.py": "545063ab1e82fe01bd6e4bc44b4259f18a3ed4af",
}


class GuardError(ValueError):
    def __init__(self, code, details=None):
        super().__init__(code)
        self.code = code
        self.details = details or {}


def canonical_hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise GuardError("DUPLICATE_JSON_KEY", {"key": key})
        result[key] = value
    return result


def load_json(path):
    def reject_constant(value):
        raise GuardError("NONFINITE_JSON_NUMBER", {"value": value})
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=_unique_object,
                      parse_constant=reject_constant)


def load_formatters(source_root):
    """Bind all imported Laya files, then load only four pure formatting functions.

    No upstream module import, model construction, or setup.py execution occurs.
    Fake-tokenizer tests exercise these actual functions, not a second serializer.
    """
    contents = {}
    hashes = {}
    for rel, expected in SOURCE_BLOBS.items():
        path = Path(source_root) / rel
        if not path.is_file():
            raise GuardError("SOURCE_UNAVAILABLE", {"path": str(path)})
        raw = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if blob != expected:
            raise GuardError("SOURCE_MISMATCH", {"path": rel, "expected": expected,
                                                 "actual": blob})
        hashes[rel] = {"git_blob_sha1": blob, "sha256": hashlib.sha256(raw).hexdigest()}
        contents[rel] = raw.decode("utf-8")
    wanted = {"serialize_state", "render_criterion", "render_options", "build_sequence"}
    tree = ast.parse(contents["laya/common.py"])
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
            and node.name in wanted]
    if {node.name for node in body} != wanted:
        raise GuardError("FORMATTER_FUNCTIONS_MISSING")
    namespace = {"json": json, "Dict": Dict, "List": List,
                 "Optional": Optional, "Union": Union}
    exec(compile(ast.Module(body=body, type_ignores=[]), "pinned_laya_common.py", "exec"),
         namespace)
    return namespace, hashes


def validate_case(case):
    if not isinstance(case, dict) or not isinstance(case.get("case_id"), str) or not case["case_id"].strip():
        raise GuardError("INVALID_CASE_ID")
    if "state" not in case or not isinstance(case["state"], (str, dict, list)):
        raise GuardError("INVALID_STATE")
    questions = case.get("questions")
    if not isinstance(questions, dict) or list(questions) != ["check"]:
        raise GuardError("EXPECTED_ONE_QUESTION_NAMED_CHECK")
    q = questions["check"]
    if not isinstance(q, dict) or q.get("type") != "choice":
        raise GuardError("CHOICE_ONLY")
    if set(q) != {"type", "instructions", "criteria"}:
        raise GuardError("UNSUPPORTED_QUESTION_FIELDS")
    if not isinstance(q["instructions"], str) or not q["instructions"].strip():
        raise GuardError("EMPTY_INSTRUCTIONS")
    criteria = q["criteria"]
    if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 8:
        raise GuardError("OPTION_COUNT_OUT_OF_RANGE", {"allowed": [2, 8]})
    for key, desc in criteria.items():
        if not isinstance(key, str) or not key.strip() or key != key.strip():
            raise GuardError("INVALID_OPTION_KEY")
        if not isinstance(desc, str) or not desc.strip():
            raise GuardError("INVALID_OPTION_DESCRIPTION", {"key": key})
    try:
        json.dumps(case["state"], allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise GuardError("STATE_NOT_FINITE_JSON") from exc
    return q


def preflight(case, tokenizer, config, formatters):
    """Return segment counts only if the upstream request preserves every token.

    This does not establish semantic evidence completeness or model correctness.
    The caller must supply the real pinned model tokenizer for actual inference.
    """
    q = validate_case(case)
    max_len, head_max_len = config.get("max_len"), config.get("head_max_len")
    if (type(max_len) is not int or type(head_max_len) is not int
            or not 16 <= head_max_len < max_len or max_len > 1024):
        raise GuardError("UNSUPPORTED_TOKEN_BUDGET")
    for attr in ("mask_token", "mask_token_id", "cls_token_id", "sep_token_id"):
        if getattr(tokenizer, attr, None) is None:
            raise GuardError("TOKENIZER_SPECIAL_TOKEN_MISSING", {"attribute": attr})
    internal = {"t": "choice", "ins": q["instructions"], "crit": q["criteria"]}
    state_text = formatters["serialize_state"](case["state"])
    options = formatters["render_options"](internal)
    mask = tokenizer.mask_token
    if not isinstance(mask, str) or not mask:
        raise GuardError("INVALID_MASK_TOKEN")
    segments = {"instructions": q["instructions"], "state": state_text}
    segments.update({"option:" + key: text for key, text in zip(q["criteria"], options)})
    replacements = [name for name, text in segments.items() if mask in text]
    if replacements:
        raise GuardError("RESERVED_MASK_WOULD_BE_REPLACED", {"segments": replacements})
    encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]
    head = encode("choice question: " + q["instructions"])
    option_tokens = [encode(" " + text) for text in options]
    state_tokens = encode(state_text)
    counts = {
        "max_len": max_len, "head_max_len": head_max_len,
        "instruction_tokens_original": len(head),
        "option_text_tokens_original": dict(zip(q["criteria"], map(len, option_tokens))),
        "state_tokens_original": len(state_tokens),
        "input_tokens_original": len(head) + sum(map(len, option_tokens))
                                 + len(options) + len(state_tokens) + 4,
        "special_tokens": len(options) + 4,
    }
    too_long = [key for key, ids in zip(q["criteria"], option_tokens) if len(ids) > 48]
    if too_long:
        raise GuardError("OPTION_TEXT_WOULD_BE_TRUNCATED", {**counts, "options": too_long})
    option_budget = head_max_len - sum(1 + len(ids) for ids in option_tokens)
    counts["instruction_budget"] = max(8, option_budget)
    # Conservative policy: any upstream option rebudgeting is outside this pilot.
    if option_budget < 16:
        raise GuardError("OPTION_REBUDGET_REQUIRED", counts)
    if len(head) > max(8, option_budget):
        raise GuardError("INSTRUCTIONS_WOULD_BE_TRUNCATED", counts)
    prefix = [tokenizer.cls_token_id] + head + [tokenizer.sep_token_id]
    markers = []
    for ids in option_tokens:
        markers.append(len(prefix))
        prefix.extend([tokenizer.mask_token_id] + ids)
    prefix.append(tokenizer.sep_token_id)
    room = max(0, max_len - len(prefix) - 1)
    counts["state_budget"] = room
    if len(state_tokens) > room:
        raise GuardError("STATE_WOULD_BE_TRUNCATED", counts)
    expected = prefix + state_tokens + [tokenizer.sep_token_id]
    actual, actual_markers = formatters["build_sequence"](
        tokenizer, case["state"], internal, max_len, head_max_len)
    if actual != expected or actual_markers != markers:
        raise GuardError("UPSTREAM_SERIALIZATION_MISMATCH", counts)
    counts.update({"input_tokens_retained": len(actual),
                   "instruction_tokens_retained": len(head),
                   "option_text_tokens_retained": counts["option_text_tokens_original"],
                   "state_tokens_retained": len(state_tokens),
                   "truncation": False, "replacement": False,
                   "option_order": list(q["criteria"]),
                   "serialized_state_sha256": hashlib.sha256(state_text.encode()).hexdigest(),
                   "sequence_sha256": canonical_hash(actual)})
    return counts
