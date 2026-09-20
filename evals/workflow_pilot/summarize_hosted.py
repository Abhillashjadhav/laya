"""Recompute descriptive agreement for the captured hosted run, not model accuracy."""
import json
from pathlib import Path

from guard import load_json
from run_local import response_errors


def main():
    root = Path(__file__).resolve().parent
    captured = load_json(root / "hosted_results.json")
    cases = load_json(root / "cases.json")["cases"]
    refs = load_json(root / "references.json")["references"]
    records = captured["records"]
    by_case = {c["case_id"]: c for c in cases}
    by_ref = {r["case_id"]: r for r in refs}
    assert len(by_case) == len(cases), "Duplicate case IDs"
    assert len(by_ref) == len(refs), "Duplicate reference IDs"
    assert len({r["case_id"] for r in records}) == len(records), "Duplicate result IDs"
    assert set(by_case) == {r["case_id"] for r in records}, "Missing or extra results"
    rows = []
    for record in records:
        case = by_case[record["case_id"]]
        assert case["state"] == record["state"] and case["questions"] == record["questions"], "Request mismatch"
        assert list(case["questions"]["check"]["criteria"]) == list(record["questions"]["check"]["criteria"]), "Option order mismatch"
        base = case.get("base_case_id", case["case_id"])
        ref = by_ref[base]
        errors = response_errors(record["result"], case["questions"]["check"]["criteria"])
        answer = record["result"].get("answers", {}).get("check", {})
        choice = answer.get("choice") if not errors else None
        rows.append({"case_id": case["case_id"], "base_case_id": base,
                     "is_perturbation": base != case["case_id"],
                     "provisional_reference": ref["label"], "choice": choice,
                     "agrees_with_reference": choice == ref["label"],
                     "false_approval": choice == "COMPLIES" and ref["label"] != "COMPLIES",
                     "reference_status": ref["reference_status"],
                     "confidence": answer.get("confidence"), "response_errors": errors})
    base_rows = [r for r in rows if not r["is_perturbation"]]
    agree = sum(r["agrees_with_reference"] for r in base_rows)
    summary = {
        "scope": "DESCRIPTIVE_SMOKE_TEST_NOT_ESTIMATED_PRODUCTION_ACCURACY",
        "unique_base_cases": len(base_rows), "base_reference_agreements": agree,
        "base_agreement_fraction": agree / len(base_rows),
        "base_false_approvals": sum(r["false_approval"] for r in base_rows),
        "perturbation_runs": len(rows) - len(base_rows),
        "human_time_saving": "NOT_MEASURED", "automatic_release_authority": False,
        "rows": rows,
    }
    (root / "hosted_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}))


if __name__ == "__main__":
    main()
