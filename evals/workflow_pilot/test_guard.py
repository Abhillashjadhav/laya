"""Guard/unit tests only. Fake tokens are NOT Laya token measurements or inference."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

from guard import GuardError, MODEL_REPO, MODEL_REVISION, SDK_COMMIT, load_formatters, load_json, preflight
import run_local


class FakeTokenizer:
    """Whitespace tokenizer solely for testing loss detection control flow."""
    mask_token = "[MASK]"
    mask_token_id, cls_token_id, sep_token_id = 1, 2, 3

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) + 10
            ids.append(self.vocab[word])
        return {"input_ids": ids}


def case():
    return {"case_id": "unit", "state": "The ledger says pending.", "questions": {
        "check": {"type": "choice", "instructions": "Does the claim match the ledger?",
                  "criteria": {"COMPLIES": "matches evidence", "VIOLATES": "contradicts evidence",
                               "UNDECIDABLE": "missing evidence"}}}}


class GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ.get("LAYA_SOURCE_ROOT", Path(__file__).resolve().parents[2]))
        cls.formatters, cls.hashes = load_formatters(cls.root)

    def check(self, sample=None, cfg=None):
        return preflight(sample or case(), FakeTokenizer(),
                         cfg or {"max_len": 512, "head_max_len": 192}, self.formatters)

    def rejects(self, code, sample, cfg=None):
        with self.assertRaises(GuardError) as caught:
            self.check(sample, cfg)
        self.assertEqual(caught.exception.code, code)
        return caught.exception.details

    def test_normal_retains_all_segments(self):
        result = self.check()
        self.assertFalse(result["truncation"])
        self.assertEqual(result["input_tokens_original"], result["input_tokens_retained"])
        self.assertEqual(result["option_order"], ["COMPLIES", "VIOLATES", "UNDECIDABLE"])

    def test_option_48_token_limit(self):
        c = case()
        c["questions"]["check"]["criteria"]["COMPLIES"] = "word " * 49
        d = self.rejects("OPTION_TEXT_WOULD_BE_TRUNCATED", c)
        self.assertGreater(d["option_text_tokens_original"]["COMPLIES"], 48)

    def test_head_rebudget_rejected_even_under_total_budget(self):
        c = case()
        c["questions"]["check"]["criteria"] = {"A": "word " * 9, "B": "word " * 9}
        self.rejects("OPTION_REBUDGET_REQUIRED", c, {"max_len": 512, "head_max_len": 32})

    def test_instruction_truncation(self):
        c = case()
        c["questions"]["check"]["instructions"] = "rule " * 200
        self.rejects("INSTRUCTIONS_WOULD_BE_TRUNCATED", c)

    def test_state_truncation(self):
        c = case()
        c["state"] = "evidence " * 600
        self.rejects("STATE_WOULD_BE_TRUNCATED", c)

    def test_exact_boundary_and_one_extra_token(self):
        c = case()
        c["state"] = ""
        room = self.check(c)["state_budget"]
        c["state"] = "fact " * room
        self.assertEqual(self.check(c)["input_tokens_retained"], 512)
        c["state"] += "overflow"
        self.rejects("STATE_WOULD_BE_TRUNCATED", c)

    def test_mask_replacement_in_any_segment(self):
        for location in ("state", "instructions", "option"):
            with self.subTest(location=location):
                c = case()
                if location == "state":
                    c["state"] = "literal [MASK] evidence"
                elif location == "instructions":
                    c["questions"]["check"]["instructions"] += " [MASK]"
                else:
                    c["questions"]["check"]["criteria"]["COMPLIES"] = "[MASK]"
                self.rejects("RESERVED_MASK_WOULD_BE_REPLACED", c)

    def test_zero_one_and_nine_options_rejected(self):
        for number in (0, 1, 9):
            c = case()
            c["questions"]["check"]["criteria"] = {str(i): "description" for i in range(number)}
            self.rejects("OPTION_COUNT_OUT_OF_RANGE", c)

    def test_empty_and_nonchoice_questions_rejected(self):
        c = case()
        c["questions"]["check"]["instructions"] = " "
        self.rejects("EMPTY_INSTRUCTIONS", c)
        c = case()
        c["questions"]["check"]["type"] = "noul"
        self.rejects("CHOICE_ONLY", c)

    def test_nonfinite_state_rejected(self):
        c = case()
        c["state"] = {"value": float("nan")}
        self.rejects("STATE_NOT_FINITE_JSON", c)

    def test_extra_question_metadata_rejected(self):
        c = case()
        c["questions"]["check"]["expected"] = "COMPLIES"
        self.rejects("UNSUPPORTED_QUESTION_FIELDS", c)

    def test_option_order_is_preserved_and_hashed(self):
        c = case()
        c["questions"]["check"]["criteria"] = dict(reversed(list(c["questions"]["check"]["criteria"].items())))
        tokenizer = FakeTokenizer()
        config = {"max_len": 512, "head_max_len": 192}
        original = preflight(case(), tokenizer, config, self.formatters)
        result = preflight(c, tokenizer, config, self.formatters)
        self.assertEqual(result["option_order"], ["UNDECIDABLE", "VIOLATES", "COMPLIES"])
        self.assertNotEqual(result["sequence_sha256"], original["sequence_sha256"])

    def test_upstream_serialization_disagreement_rejected(self):
        modified = dict(self.formatters)
        original = modified["build_sequence"]
        def bad(*args):
            ids, markers = original(*args)
            return ids[:-1], markers
        modified["build_sequence"] = bad
        with self.assertRaises(GuardError) as caught:
            preflight(case(), FakeTokenizer(), {"max_len": 512, "head_max_len": 192}, modified)
        self.assertEqual(caught.exception.code, "UPSTREAM_SERIALIZATION_MISMATCH")

    def test_changed_source_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "laya"
            p.mkdir()
            (p / "common.py").write_bytes((self.root / "laya/common.py").read_bytes() + b"\n")
            (p / "agent.py").write_bytes((self.root / "laya/agent.py").read_bytes())
            with self.assertRaises(GuardError) as caught:
                load_formatters(directory)
            self.assertEqual(caught.exception.code, "SOURCE_MISMATCH")

    def test_changed_package_initializer_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            shutil.copytree(self.root / "laya", Path(directory) / "laya")
            p = Path(directory) / "laya/__init__.py"
            p.write_bytes(p.read_bytes() + b"\n")
            with self.assertRaises(GuardError) as caught:
                load_formatters(directory)
            self.assertEqual(caught.exception.details["path"], "laya/__init__.py")


class InputAndRunnerTests(unittest.TestCase):
    def test_nonfinite_response_evidence_remains_writable(self):
        raw = {"probabilities": {"A": float("nan"), "B": float("inf")}}
        safe = run_local.preserve_json(raw)
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "result.json"
            run_local.write_report(p, {"raw_response": safe})
            reread = json.loads(p.read_text())
            self.assertEqual(reread["raw_response"]["probabilities"]["A"], {"__invalid_float__": "nan"})

    def test_invalid_or_failed_rows_never_produce_success(self):
        invalid = {"execution_status": "EXECUTED", "review_status": "INVALID_RESPONSE"}
        failed = {"execution_status": "FAILED", "review_status": "INFERENCE_ERROR"}
        valid = {"execution_status": "EXECUTED", "review_status": "OBSERVATION_ONLY"}
        self.assertEqual(run_local.final_status([invalid], []), "PARTIAL")
        self.assertEqual(run_local.final_status([failed], []), "PARTIAL")
        self.assertEqual(run_local.final_status([valid], [{"reason": "later error"}]), "PARTIAL")
        self.assertEqual(run_local.final_status([valid], []), "EXECUTED")

    def test_unlisted_model_assets_are_not_copied(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            runtime = Path(directory) / "runtime"
            model.mkdir()
            runtime.mkdir()
            required = ["model.safetensors", "rl_agent_config.json", "encoder/config.json",
                        "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]
            files = {}
            for name in required:
                p = model / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("unit fixture, not model weights")
                files[name] = run_local.sha256_file(p)
            (model / "tokenizer/special_tokens_map.json").write_text("unlisted conflicting value")
            manifest = {"model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
                        "sdk_commit": SDK_COMMIT, "checkpoint": "english", "files": files}
            (model / "pilot_model_manifest.json").write_text(json.dumps(manifest))
            verified = run_local.verify_model_dir(model)
            run_local.stage_model_copy(model, verified, runtime)
            self.assertFalse((runtime / "tokenizer/special_tokens_map.json").exists())
            self.assertTrue((runtime / "tokenizer/tokenizer.json").is_file())
            (model / "tokenizer/tokenizer.json").write_text("changed after manifest")
            with self.assertRaises(GuardError) as caught:
                run_local.verify_model_dir(model)
            self.assertEqual(caught.exception.code, "MODEL_FILE_HASH_MISMATCH")

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "input.json"
            p.write_text('{"state":1,"state":2}')
            with self.assertRaises(GuardError) as caught:
                load_json(p)
            self.assertEqual(caught.exception.code, "DUPLICATE_JSON_KEY")

    def test_json_nan_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "input.json"
            p.write_text('{"state":NaN}')
            with self.assertRaises(GuardError):
                load_json(p)

    def test_missing_dependencies_retains_all_rows(self):
        valid = case()
        second = copy.deepcopy(valid)
        second["case_id"] = "second"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cases.json").write_text(json.dumps({"cases": [valid, {"case_id": "bad"}, second]}))
            with mock.patch("run_local.importlib.util.find_spec", return_value=None), contextlib.redirect_stdout(io.StringIO()):
                code = run_local.main(["--cases", str(root / "cases.json"), "--model-dir", str(root / "absent"),
                                       "--output", str(root / "result.json")])
            result = json.loads((root / "result.json").read_text())
            self.assertEqual(code, 2)
            self.assertEqual(result["executed_cases"], 0)
            self.assertEqual(len(result["rows"]), 3)
            self.assertEqual([r["review_status"] for r in result["rows"]],
                             ["UNAVAILABLE", "INPUT_REJECTED", "UNAVAILABLE"])
            self.assertTrue(all(r["model_input_tokens"] is None for r in result["rows"]))

    def test_duplicate_case_ids_both_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cases.json").write_text(json.dumps({"cases": [case(), case()]}))
            with contextlib.redirect_stdout(io.StringIO()):
                run_local.main(["--cases", str(root / "cases.json"), "--model-dir", str(root / "absent"),
                                "--output", str(root / "result.json")])
            result = json.loads((root / "result.json").read_text())
            self.assertEqual([r["reason"] for r in result["rows"]], ["DUPLICATE_CASE_ID"] * 2)

    def test_response_validation_does_not_claim_correctness(self):
        raw = {"answers": {"check": {"type": "choice", "choice": "A",
                                      "probabilities": {"A": 0.99, "B": 0.01}, "confidence": 0.9192}}}
        self.assertEqual(run_local.response_errors(raw, {"A": "x", "B": "y"}), [])
        raw["answers"]["check"]["probabilities"]["A"] = 0.7
        self.assertEqual(run_local.response_errors(raw, {"A": "x", "B": "y"}),
                         ["PROBABILITIES_DO_NOT_SUM_TO_ONE"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
