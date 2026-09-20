# Workflow evaluation pilot

The first hosted diagnostic run matched **3 of 6 provisional reference answers**. It accepted all three deliberately flawed examples. Two option-order repeats retained the same incorrect verdicts. This configuration has not earned automatic approval authority.

Laya is a small model that returns choices and probabilities from supplied context and criteria. This pilot asks whether it can help check an output against a written requirement. It does not replace deterministic validators, evidence, or independently reviewed reference answers.

## What actually ran

Eight requests ran through the [official free Hugging Face playground](https://huggingface.co/spaces/convaiinnovations/laya-demo): six unique synthetic cases and two order perturbations. Exact submitted state, questions, and returned JSON are in [hosted_results.json](hosted_results.json). Only synthetic text was submitted. No paid API, connector data, or production workflow was used.

The UI reported model `laya`; it did not expose its checkpoint or deployed SDK revision. Hosted input token counts and latency are server-reported. Exact pre-truncation token counts were unavailable, so these results are a hosted smoke test, not a pinned local reproduction. Local inference was **not run** because this environment lacked the model/dependencies and its package/model download attempts timed out.

## Observations

References were proposed before the base runs. They are explicit diagnostic expectations, not an independent gold dataset. The model received the criteria, but not reference labels. All original results are retained.

| Case | Supplied situation | Expected | Laya chose | Choice probability | Reported confidence | Finding |
| --- | --- | --- | --- | ---: | ---: | --- |
| P01 | Reply accepts Friday; explanation says it refuses work before Monday | VIOLATES | COMPLIES | 58.99% | 13.67% | False approval |
| P02 | Reply and explanation both set Monday as earliest start | COMPLIES | COMPLIES | 81.11% | 44.75% | Agrees |
| P03 | User wants a light blanket; reply selects the explicitly heavy one | VIOLATES | COMPLIES | 66.72% | 21.43% | False approval |
| P04 | Reply selects the explicitly warm, lightweight blanket | COMPLIES | COMPLIES | 85.10% | 52.34% | Agrees |
| P05 | Claim has no supplied source | UNDECIDABLE | UNDECIDABLE | 61.59% | 15.68% | Correct abstention under the supplied rule |
| P06 | Single-answer quiz asks which animal supplies milk; both cow and goat fit the supplied evidence | VIOLATES | COMPLIES | 61.11% | 16.17% | False approval |

P01 and P03 were repeated with `VIOLATES, UNDECIDABLE, COMPLIES` instead of `COMPLIES, VIOLATES, UNDECIDABLE`. Both still chose COMPLIES, at 57.28% and 53.93% choice probability. Their probabilities changed, but their verdicts did not. These repeats are not independent examples and are excluded from the six-case agreement fraction.

**3/6 describes only this small selected set; it is not an estimate of production accuracy.** These are synthetic examples, so they do not prove any existing repository is broken. They show that these evaluation questions can miss deliberately supplied defects.

Low confidence also occurred on P05's correct abstention. It therefore does not establish whether the candidate is broken. The reviewed local SDK defines Choice confidence using entropy: `1 - H(probabilities) / log(number_of_options)`. It measures how concentrated the distribution is, not the probability of correctness. The hosted implementation version is unknown. Its separate `rl_agent.act_probability: 1` field is preserved as returned and grants no action authority here.

## Implemented boundary

- One `state` plus one Choice question per packet; 2–8 options. Cases and reference labels live in separate files.
- A local adapter binds reviewed upstream source and a model manifest, loads from an explicit local model directory, and disables implicit Hugging Face credentials/downloads for inference.
- Before inference, the actual tokenizer and upstream serializer must preserve every instruction, option, and state token. Clipping, rebudgeting, or replacement of a reserved mask token rejects the input.
- The runner records rejected, unavailable, failed, and executed observations. A model answer cannot publish, route, modify a repository, or approve a release.

This is the common evaluation boundary to test before adding workflow-specific adapters. It is not installed across existing projects or hosted chat surfaces. Reference agreement, valid transport, and model confidence are separate facts.

## Files and reproduction

| File | Purpose |
| --- | --- |
| `cases.json` | Exact synthetic inputs; includes the two perturbations |
| `references.json` | Separate proposed labels and reasons for the six base cases |
| `hosted_results.json` | Captured hosted requests and unmodified response objects |
| `hosted_summary.json` | Descriptive agreement and false approvals, recomputed from those files |
| `local_unavailable.json` | Actual local attempt: zero inferences, missing dependencies, all eight cases retained |
| `summarize_hosted.py` | Recompute the hosted summary without calling any model |
| `guard.py`, `run_local.py` | Lossless input checks and evaluation-only local runner |
| `download_model.py` | Explicit selective download helper for the pinned public checkpoint |
| `test_guard.py` | Guard and runner failure-path tests, without model inference |
| `LOCAL_RUN.md` | Local setup, source/model pins, commands, and limitations |
| `qa_effort_template.json` | Empty measurement template; no time savings asserted |

From the repository root, with Python 3.11 or newer:

```bash
python3 evals/workflow_pilot/summarize_hosted.py
python3 -m unittest discover -s evals/workflow_pilot -p 'test_guard.py' -v
```

Local inference instructions are in [LOCAL_RUN.md](LOCAL_RUN.md). The source files are additions under `evals/workflow_pilot`; upstream model behavior is unchanged.

Validation in the implementation environment: **23 unit checks passed**, using an exact source fixture from the pinned SDK commit. The checks use a fake tokenizer to exercise control flow; they do not validate real model accuracy or token fit. The hosted summary recomputation passed, and the local missing-dependency attempt correctly exited with code 2 and retained all eight unexecuted rows. A separate code review checked source/model provenance, clipping detection, reference separation, and failure reporting. Remote CI and local model inference have not been verified.

## What would justify using it in a workflow

First fix deterministic acceptance rules, then compare those rules alone with the same rules plus Laya on one bounded semantic task. Use independently reviewed labels, held-out cases, legitimate alternatives, missing evidence, and repeated/perturbed inputs. Include the general LLM already available in the session as a comparator when feasible; that comparison has not run here. Choose false-approval/false-block costs and thresholds before the held-out run, not from these six examples.

The owner's goal is a reduction in hands-on QA share from an estimated 60–70% to 5%. This run measured neither that baseline nor savings. Track QA minutes and total hands-on work for the same outputs and time window, including contract writing, labeling, false-alarm review, repair, and maintenance. Also track total minutes per accepted outcome and escaped defects so moving work into setup or enlarging the denominator cannot masquerade as savings.

The next deployment step depends on improved held-out results and reduced all-in human effort. These observations support keeping Laya experimental rather than placing it in every workflow's approval path.
