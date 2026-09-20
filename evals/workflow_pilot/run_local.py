"""Evaluation only: one local Laya checkpoint, no routing or action execution."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import tempfile
import time

from guard import (GuardError, MODEL_REPO, MODEL_REVISION, SDK_COMMIT, canonical_hash,
                   load_formatters, load_json, preflight, validate_case)

DEPENDENCIES = ("torch", "transformers", "huggingface_hub", "safetensors", "numpy")


def offline_environment():
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                       "USE_TF": "0", "TOKENIZERS_PARALLELISM": "false"})
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        os.environ.pop(name, None)


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_model_dir(model_dir):
    if not model_dir.is_dir():
        raise GuardError("MODEL_DIRECTORY_UNAVAILABLE")
    manifest_path = model_dir / "pilot_model_manifest.json"
    if not manifest_path.is_file():
        raise GuardError("PINNED_MODEL_MANIFEST_UNAVAILABLE")
    manifest = load_json(manifest_path)
    expected = {"model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
                "sdk_commit": SDK_COMMIT}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise GuardError("MODEL_MANIFEST_REVISION_MISMATCH", {"field": key})
    if manifest.get("checkpoint") not in ("english", "multilingual", "typed-decisions"):
        raise GuardError("UNKNOWN_CHECKPOINT")
    files = manifest.get("files")
    required = {"model.safetensors", "rl_agent_config.json", "encoder/config.json",
                "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"}
    if not isinstance(files, dict) or not required.issubset(files):
        raise GuardError("MODEL_MANIFEST_INCOMPLETE")
    for rel, expected_hash in files.items():
        if not isinstance(rel, str) or not (
                rel in {"model.safetensors", "rl_agent_config.json", "encoder/config.json"}
                or rel.startswith("tokenizer/")):
            raise GuardError("UNSUPPORTED_MODEL_MANIFEST_FILE", {"file": rel})
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise GuardError("MODEL_MANIFEST_PATH_ESCAPE")
        path = (model_dir / rel).resolve()
        if not path.is_relative_to(model_dir.resolve()):
            raise GuardError("MODEL_MANIFEST_PATH_ESCAPE")
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise GuardError("MODEL_FILE_HASH_MISMATCH", {"file": rel})
    return manifest


def stage_model_copy(model_dir, manifest, runtime_root):
    """Copy only verified manifest-listed assets; ignore all unlisted local files."""
    for rel in manifest["files"]:
        target = runtime_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel == "model.safetensors":
            target.symlink_to((model_dir / rel).resolve())
        else:
            shutil.copy2(model_dir / rel, target)


def preserve_json(value):
    """Keep invalid response values as explicit tags so error evidence is writable."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"__invalid_float__": repr(value)}
    if isinstance(value, list):
        return [preserve_json(item) for item in value]
    if isinstance(value, dict):
        return {str(key): preserve_json(item) for key, item in value.items()}
    return {"__unsupported_type__": type(value).__name__, "repr": repr(value)[:1000]}


def final_status(rows, errors):
    if rows and not errors and all(row.get("review_status") == "OBSERVATION_ONLY" for row in rows):
        return "EXECUTED"
    if any(row.get("execution_status") in ("EXECUTED", "FAILED") for row in rows):
        return "PARTIAL"
    return "NOT_RUN"


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", delete=False) as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        temp = stream.name
    os.replace(temp, path)


def response_errors(raw, criteria):
    """Validate transport shape, not whether a judgment is correct."""
    try:
        answer = raw["answers"]["check"]
        probs = answer["probabilities"]
        if answer["type"] != "choice" or set(probs) != set(criteria):
            return ["CHOICE_RESPONSE_SCHEMA_MISMATCH"]
        if any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
               for p in probs.values()):
            return ["INVALID_PROBABILITY"]
        if abs(sum(probs.values()) - 1) > 0.001:
            return ["PROBABILITIES_DO_NOT_SUM_TO_ONE"]
        if answer["choice"] not in probs:
            return ["UNKNOWN_CHOSEN_LABEL"]
        if probs[answer["choice"]] + 0.0001 < max(probs.values()):
            return ["CHOICE_NOT_HIGHEST_PROBABILITY"]
        conf = answer["confidence"]
        if type(conf) not in (int, float) or not math.isfinite(conf) or not 0 <= conf <= 1:
            return ["INVALID_ENTROPY_CONFIDENCE"]
    except (KeyError, TypeError, AttributeError):
        return ["MALFORMED_RESPONSE"]
    return []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.output.resolve() == args.cases.resolve():
        parser.error("output must differ from the source cases file")
    offline_environment()
    runtime_copy = None
    report = {
        "status": "NOT_RUN", "scope": "EVALUATION_ONLY_NO_ACTION_AUTHORITY",
        "sdk_commit": SDK_COMMIT, "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
        "platform": {"python": platform.python_version(), "system": platform.system(),
                     "machine": platform.machine(), "cpu_count": os.cpu_count()},
        "dependencies": {}, "model_token_count_status": "NOT_MEASURED",
        "rows": [], "errors": [],
        "correctness": "NOT_ADJUDICATED_REFERENCES_NOT_READ",
        "confidence_definition": "Choice: 1 - Shannon_entropy(probabilities) / log(option_count)",
        "offline_scope": "HF/Transformers offline flags; no OS-level egress sandbox installed",
    }
    try:
        payload = load_json(args.cases)
        if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list) or not payload["cases"]:
            raise GuardError("EXPECTED_NONEMPTY_CASES_LIST")
        cases = payload["cases"]
        if len(cases) > 100:
            raise GuardError("PILOT_CASE_LIMIT", {"maximum": 100, "actual": len(cases)})
        ids = [case.get("case_id") if isinstance(case, dict) else None for case in cases]
        for case in cases:
            row = {
                "case_id": case.get("case_id") if isinstance(case, dict) else None,
                "input_index": len(report["rows"]), "execution_status": "NOT_RUN",
                "model_input_tokens": None,
                "model_input_tokens_status": "NOT_MEASURED", "raw_response": None,
                "local_inference_ms": None, "correctness": "NOT_ADJUDICATED",
            }
            report["rows"].append(row)
            try:
                q = validate_case(case)
                if ids.count(case["case_id"]) != 1:
                    raise GuardError("DUPLICATE_CASE_ID")
                row.update({"candidate_sha256": canonical_hash(case["state"]),
                            "request_sha256": hashlib.sha256(json.dumps(
                                {"state": case["state"], "questions": case["questions"]},
                                ensure_ascii=False, allow_nan=False).encode()).hexdigest(),
                            "option_order": list(q["criteria"])})
            except GuardError as exc:
                row.update({"review_status": "INPUT_REJECTED", "reason": exc.code,
                            "preflight": exc.details})
        if all(row.get("review_status") == "INPUT_REJECTED" for row in report["rows"]):
            raise GuardError("NO_VALID_INPUT_CASES")
        report["cases_file_sha256"] = sha256_file(args.cases)
        missing = [name for name in DEPENDENCIES if importlib.util.find_spec(name) is None]
        if missing:
            raise GuardError("MISSING_DEPENDENCIES", {"modules": missing})
        for name in DEPENDENCIES:
            report["dependencies"][name] = importlib.metadata.version(name)
        formatters, source_hashes = load_formatters(args.source_root)
        report["source_hashes"] = source_hashes
        manifest = verify_model_dir(args.model_dir.resolve())
        report["model_manifest"] = manifest
        # Upstream rewrites some tokenizer configs. Keep the pinned download immutable.
        runtime_copy = tempfile.TemporaryDirectory(prefix="laya_pilot_runtime_")
        runtime_root = Path(runtime_copy.name)
        stage_model_copy(args.model_dir, manifest, runtime_root)
        # Source is pinned before importing the upstream package. No pip/setup execution.
        sys.path.insert(0, str(args.source_root.resolve()))
        from laya import Agent
        import laya.agent as upstream_agent
        if Agent is not upstream_agent.Agent:
            raise GuardError("UNEXPECTED_EXPORTED_AGENT")
        if Path(upstream_agent.__file__).resolve() != (args.source_root / "laya/agent.py").resolve():
            raise GuardError("UNEXPECTED_IMPORTED_AGENT")
        start = time.perf_counter()
        agent = Agent(str(runtime_root), device=args.device, token=False)
        report["model_load_ms"] = round((time.perf_counter() - start) * 1000, 3)
        report["actual_device"] = str(agent.device)
        report["effective_model_config"] = agent.cfg
        report["runtime_tokenizer_config_sha256"] = sha256_file(
            runtime_root / "tokenizer/tokenizer_config.json")
        report["tokenizer_config_changed_by_upstream_loader"] = (
            report["runtime_tokenizer_config_sha256"] != manifest["files"]["tokenizer/tokenizer_config.json"])
        for row, case in zip(report["rows"], cases):
            if row.get("review_status") == "INPUT_REJECTED":
                continue
            try:
                row["preflight"] = preflight(case, agent.tok, agent.cfg, formatters)
                row["model_input_tokens"] = row["preflight"]["input_tokens_retained"]
                row["model_input_tokens_status"] = "MEASURED_WITH_REAL_TOKENIZER"
            except GuardError as exc:
                row.update({"review_status": "INPUT_REJECTED", "reason": exc.code,
                            "preflight": exc.details})
                continue
            start = time.perf_counter()
            row["execution_status"] = "FAILED"
            try:
                # Reference labels and other case metadata are not passed to the model.
                raw = agent.predict(case["state"], case["questions"])
                row["raw_response"] = preserve_json(raw)
                row["execution_status"] = "EXECUTED"
                errors = response_errors(raw, case["questions"]["check"]["criteria"])
                try:
                    json.dumps(raw, allow_nan=False)
                except (ValueError, TypeError):
                    errors.append("NON_JSON_RESPONSE_VALUES_PRESERVED_AS_TAGS")
                usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
                if not isinstance(usage, dict) or usage.get("input_tokens") != row["model_input_tokens"]:
                    errors.append("ACTUAL_TOKEN_USAGE_MISMATCH")
                row["review_status"] = "INVALID_RESPONSE" if errors else "OBSERVATION_ONLY"
                row["response_errors"] = errors
                row["actual_device"] = str(agent.device)
            except Exception as exc:
                row.update({"review_status": "INFERENCE_ERROR", "error_type": type(exc).__name__,
                            "error": str(exc)[:1000]})
            finally:
                row["local_inference_ms"] = round((time.perf_counter() - start) * 1000, 3)
        if any(row["model_input_tokens"] is not None for row in report["rows"]):
            report["model_token_count_status"] = "MEASURED_WITH_REAL_TOKENIZER"
    except Exception as exc:
        reason = exc.code if isinstance(exc, GuardError) else type(exc).__name__
        details = exc.details if isinstance(exc, GuardError) else {"message": str(exc)[:1000]}
        report["errors"].append({"reason": reason, "details": details})
        for row in report["rows"]:
            if row["execution_status"] == "NOT_RUN" and "review_status" not in row:
                row.update({"review_status": "UNAVAILABLE", "reason": reason})
    report["executed_cases"] = sum(row["execution_status"] == "EXECUTED" for row in report["rows"])
    report["status"] = final_status(report["rows"], report["errors"])
    if runtime_copy is not None:
        runtime_copy.cleanup()
    report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if platform.system() == "Darwin" else 1024)
    write_report(args.output, report)
    print(json.dumps({"status": report["status"], "executed_cases": report["executed_cases"],
                      "report": str(args.output), "errors": report["errors"]}))
    return 0 if report["status"] == "EXECUTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
