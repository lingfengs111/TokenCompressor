# Remote Run Notes For `gpub005`

This repo has an older remote workflow rooted at:

- host: `gpub005.delta.ncsa.illinois.edu`
- repo root: `/u/lshi8/TokenCompressor`
- python: `/u/lshi8/miniconda3/envs/py313/bin/python`

The current local worktree is not clean. Do not push `main` blindly.

For the standard backbone runs discussed on `2026-04-25`, the minimum remote tree must include at least:

- `run_sasrec_taobao_standard.py`
- `run_hstu_taobao_standard.py`
- `train_backbone_standard.py`
- `core/loo_dataset.py`
- `core/streaming_eval.py`
- `backbones/SASRec.py`
- `backbones/HSTU.py`
- `backbones/HSTUOfficialish.py`
- `backbones/HSTUResearchAligned.py`
- `backbones/modules.py`
- `backbones/patch.py`

The fast path is:

1. Push a dedicated branch that contains the remote launch helpers in `scripts/remote/` plus the actual runner/dependency files above.
2. On remote:

```bash
cd /u/lshi8/TokenCompressor
git fetch origin
git checkout <branch>
git pull --ff-only origin <branch>
bash scripts/remote/check_gpub005_ready.sh
BRANCH=<branch> bash scripts/remote/launch_gpub005_ml10m_original_full_dotbce.sh
```

The first experiments worth moving are:

- `ML-10M / SASRec / original / full-view / dot-BCE`
- `ML-10M / HSTU / original / full-view / dot-BCE`

Why these first:

- `ml-10m` data is already on remote and small enough
- they are currently blocked locally only by GPU queueing
- they do not depend on large local `xlong` checkpoints being copied over

`xlong` is better treated separately after the `ml-10m` pair is running.
