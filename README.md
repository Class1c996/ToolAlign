# ToolAlign: Executable Tool-Use Agent Post-Training

ToolAlign is a personal, independent two-week project completed in July 2026.
It studies tool-use post-training for a local Qwen3-1.7B agent: executable SFT,
interactive GRPO ablation, and evaluation that checks full tool trajectories
rather than only a predicted tool name.

The work was run locally on an NVIDIA RTX 5080 16GB with QLoRA. Model weights
and adapters are intentionally not included in this public-ready repository.

## Validated result

Selected checkpoint: `checkpoints/sft_multitask_state_v5`.

| Executable suite | Base end-to-end | Selected SFT end-to-end | Tool success | False success |
| --- | ---: | ---: | ---: | ---: |
| Broad challenge (120) | 55.83% | **74.17%** | **100.00%** | 0.00% |
| State/value holdout (20) | 45.00% | **60.00%** | **100.00%** | 0.00% |
| Wording holdout (20) | n/a | **70.00%** | **100.00%** | 0.00% |

The selected SFT has 99.17% JSON legality on the broad challenge. The small
interactive GRPO ablation retained 74.17% end-to-end success but reduced JSON
legality to 98.33%, so it is reported as a negative ablation and is not the
selected model.

## What makes the evaluation executable

- 16 deterministic order, calendar and travel tools with JSON-schema checks.
- Isolated episode state, reset/replay, JSON/SQLite snapshots and structured
  errors for unknown tools, missing fields and invalid parameters.
- Full-loop rollout: model output -> strict JSON parser -> tool execution ->
  tool result appended to context -> next model output.
- Exact-trace challenge scoring: expected calls, successful execution and a
  non-empty grounded final answer are all required.
- Separate state/value and wording holdouts. This caught an earlier apparent
  gain that was below the Base model on state/value generalization.

## Quick public verification (no model download)

Python 3.11+ is sufficient for the deterministic environment. Install pytest,
then run the low-cost explicit suite:

```powershell
cd ToolAlign
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/reproduce.py --suite public_smoke
```

The command regenerates the deterministic 1,200-task corpus, but validates and
replays only the named 120-row `public_smoke` suite. It does not scan arbitrary
JSONL files in `data/processed`.

## Model evaluation and demo

After placing the local base model and adapters in the ignored paths and
installing `requirements.txt`, evaluate a checkpoint with:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_checkpoint.py --checkpoint checkpoints\sft_multitask_state_v5 --suite challenge_v1 --output reports\challenge_v1_sft_multitask_state_v5.jsonl
```

The demo is genuine inference, not a gold-call animation. It exposes each raw
model JSON output, parsed action, tool result, reward breakdown, state and full
trace:

```powershell
.\.venv\Scripts\python.exe demo\app.py --model sft --task-id challenge-0000 --output reports\demo.json
```

`--model base`, `--model sft`, and `--model grpo` select local presets; a
missing checkpoint fails explicitly instead of substituting a scripted answer.

## Repository map

```text
envs/       deterministic tools, schema validation, parsers and state stores
rewards/    terminal and shaped reward decomposition
training/   SFT and interactive multi-turn GRPO loops
scripts/    data build, suites, replay, evaluation and reproduction commands
demo/       checkpoint-driven executable inference demo
tests/      parser, executor, state override, reward, replay and challenge tests
data/       task generator, explicit suites and small public examples
configs/    reproducible local training configurations
docs/       experiment report and public-release checklist
```

## Reproducibility boundaries

- Results use one training seed and one local Qwen3-1.7B setup; they are not
  multi-seed confidence intervals.
- State/value and wording holdouts each contain 20 tasks. They are meaningful
  adversarial checks, but not a large external benchmark.
- BFCL and xLAM are future external benchmark/data directions, not completed
  headline results in this repository.
- The canonical methodology, negative GRPO ablation and exact commands are in
  [docs/06-最终实验结论与复现.md](docs/06-最终实验结论与复现.md).

## License

Released under the [MIT License](LICENSE).
