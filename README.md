# Antigravity Training Monitor

A zero-dependency, pure-Python Antigravity side-pane UI plugin that monitors live and recently completed ML/RL training runs with auto-refreshing win-rate, loss/KL, timing, and ETA views.

## Install on Any Machine or Antigravity VM

Clone directly into Antigravity's global plugins directory (no `pip` or `npm` install needed — stdlib Python 3 only):

```bash
git clone https://github.com/AClaybrook/antigravity-training-monitor ~/.gemini/config/plugins/training-monitor
```

Then open **Training Monitor** from the side pane (`+` menu) or ask your Antigravity agent to open it.

To update later:

```bash
git -C ~/.gemini/config/plugins/training-monitor pull
```

## What It Auto-Discovers

1. **Antigravity background task logs** (`~/.gemini/antigravity/brain/*/.system_generated/tasks/*.log`):
   - Orbit Wars League PPO (`scripts/rl_finetune_transformer.py`)
   - Orbit Wars Soft-KD & Ensemble Distillation (`scripts/distill_soft_kd.py`, `scripts/distill_ensemble_4m.py`)
   - Galcon Gen4 Multi-Send PPO & Supervised training (`galcon/train.py`)
   - Generic Python training logs containing `epoch`/`iter`/`step` + numeric `key=value` metrics
2. **Universal `metrics.jsonl` files** in any workspace (`metrics.jsonl`, `train_metrics.jsonl`, `*.metrics.jsonl`) or conversation `scratch/` directory.

## Standalone Browser Mode

Works headless outside Antigravity too (e.g., over SSH port forwarding from a remote VM):

```bash
python3 ~/.gemini/config/plugins/training-monitor/sidecars/monitor/parser.py \
  --serve 8765 \
  --workspaces /path/to/your/project
```

## Logging Future Projects (`metrics.jsonl`)

Append one JSON object per epoch/iteration in your workspace (`metrics.jsonl` or `logs/<run>.metrics.jsonl`):

```python
with open("metrics.jsonl", "a") as f:
    f.write(json.dumps({"step": ep, "total_steps": epochs, "loss": loss, "val_acc": acc}) + "\n")
```

Optional helpful keys: `run_name` (or `title`), `total_steps` (or `total_iters`, `total_epochs`), `elapsed_s` (or `time_s`), `device`, `best` (`true`/`false`), `roll_s`, `upd_s`.

## Tests

```bash
python3 tests/test_parser.py
```
