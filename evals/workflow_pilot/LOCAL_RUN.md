# Guarded local evaluation — not a production integration

This pilot tests one bounded Choice question per case. It does not route models, authorize actions, expose connectors, change repository release gates, repair an output or read reference labels. It supports 2–8 short options and at most 100 cases per invocation. It is intended for Python 3.11+ on macOS/Linux.

The exact reviewed SDK commit is `d113dca2512fb3eaca313534bc54c7162d87c1d4`; the Hugging Face model bundle revision is `1c5edc17a7acd8701df6fc341c0d179f1c62c982`. The guard binds the exact Git blob hashes of all seven upstream Python package files, including `laya/common.py`, `laya/agent.py` and `laya/__init__.py`. Changing any requires a new source review and updated guard; it does not silently accept new SDK behavior.

## What the guard protects

Laya's pinned serializer silently clips individual option texts after 48 tokens, can rebudget options further, clips instructions to the available head budget, and then clips state to remaining sequence room. It also replaces literal model mask tokens. This pilot rejects those cases before inference. It compares the resulting sequence with the actual upstream pure formatter and records original/retained segment counts, option order and hashes. Option rebudgeting is conservatively rejected even if a different shortening policy could work.

Token completeness is **not semantic evidence completeness**. A short packet can still omit the decisive fact or contain an unclear rule. Checkpoint language/domain suitability and independently reviewed acceptance examples are still required. The runner does not automatically choose a language model; run English cases against the English checkpoint and use separately approved checkpoint-specific case sets for other languages.

Choice confidence is `1 - entropy(probabilities)/log(number_of_options)`. It is a distribution concentration statistic, not measured accuracy. The full probability vector and separate learned `action.act_probability` remain in each raw response. No confidence threshold is used to approve or block an artifact.

## Obtain the model explicitly on an authorized local machine

Inspect the pinned source and install compatible PyTorch, Transformers, safetensors, huggingface_hub and numpy in an isolated environment using their supported installation procedures. Installation was not performed by this pilot in the current environment. Exact installed versions are recorded by the runner; a future validated environment should retain its dependency lock. Do not use unreviewed checkpoint Python files or `trust_remote_code=True`.

From the fork's root, run:

```bash
python evals/workflow_pilot/download_model.py --checkpoint english --out /absolute/path/to/laya-english-pinned
```

The helper downloads only the selected checkpoint's safetensors, configuration and tokenizer assets at the pinned public revision. It passes `token=False`, requests no account credentials and saves file hashes in `pilot_model_manifest.json`. The output directory must be empty. For multilingual or typed-decisions, the returned `model_dir` includes that subfolder; use the printed directory for inference. Download and inference are deliberately separate commands. The SDK's root downloader is not used, because its unfiltered root snapshot can fetch the entire model family.

Do not preload all three checkpoints on an 8GB laptop or assume the T4 GPU latency headline applies locally. One English checkpoint has about 1.684GB of FP32 parameters before framework, activations and load-time overhead. Measure actual peak memory and elapsed time on your intended host. This runner uses one resident checkpoint and one question per call, with no automatic retry. It does not install an OS memory limit or hard wall-time deadline; run under host-level limits if required and treat process termination as an incomplete evaluation.

## Run the evaluation

Input shape:

```json
{"cases":[{"case_id":"example","state":{"fact":"pending"},"questions":{"check":{"type":"choice","instructions":"Does the claim match the fact?","criteria":{"COMPLIES":"The evidence supports the claim.","VIOLATES":"The evidence contradicts the claim.","UNDECIDABLE":"The necessary evidence is missing."}}}}]}
```

Place independently reviewed labels in a **separate file**. The local runner has no reference-file argument and passes only `state` and `questions` to Laya. Ground-truth labels must never be embedded in either field. Dictionary insertion order defines option order; use separate case IDs for explicit order permutations and do not count them as independent base examples.

```bash
python evals/workflow_pilot/run_local.py \
  --cases evals/workflow_pilot/cases.json \
  --model-dir /absolute/path/to/laya-english-pinned \
  --output /absolute/path/to/local_results.json
```

The runner sets Hugging Face/Transformers offline flags, disables implicit HF credentials and telemetry, and requires local tokenizer, encoder configuration, weights and the pinned hash manifest. It makes no inference-time download call. Those flags are library protections, not an OS network firewall. Use a host network policy for a hard egress guarantee.

The upstream loader may normalize tokenizer configuration. The runner copies small configuration/tokenizer assets into a temporary directory and links the already-verified weights, so original downloaded files remain unchanged. It records the effective tokenizer config hash and whether normalization occurred. Original manifest hashes and runtime hashes remain distinct.

The report preserves rejected, unavailable, failed and successful rows. It records actual retained input tokens only after the real model tokenizer has run; unavailable rows say `NOT_MEASURED`. Successful inference rows retain raw response, probabilities, entropy confidence, input hashes, source/model/config versions, device, latency and process peak RSS. Transport-schema validation is not correctness adjudication. Review reference labels separately after the raw report is saved.

Exit code 0 means every case produced a response with the expected transport schema and token count; it does **not** mean every judgment was correct. Code 2 means some input was rejected, a response was invalid, inference failed, or required dependencies/model/source were unavailable. Nonfinite response values are preserved as explicit JSON tags and marked invalid. A malformed or duplicate-key JSON file is rejected as a whole because its original intent cannot be recovered safely. An interrupted process may not produce its final report: retain the previous completed report and never infer a pass from a missing file.

## Unit checks, not model evaluation

```bash
python -m unittest discover -s evals/workflow_pilot -p test_guard.py -v
```

The tests use a whitespace fake tokenizer and the actual pinned upstream formatting functions extracted from the reviewed AST. They test boundary control flow, malformed inputs, duplicate IDs, mask replacement, source changes and honest unavailable results. **Their counts are not Laya token counts, inference results or model accuracy.** To test against an exact source checkout elsewhere, set `LAYA_SOURCE_ROOT` to that checkout's root.

## What this pilot cannot establish

A small synthetic run cannot establish calibration, general model accuracy, defect-confidence correlation, production readiness or reducing the owner's self-reported 60–70% hands-on QA share to 5%. Compare the current workflow, deterministic-only improvements, and the same workflow with optional Laya observations. Keep false approvals, false blocks, missing evidence and total owner effort visible. Do not broadly integrate the model until the actual added value is demonstrated.
