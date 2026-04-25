# Remote Run Notes

This repo has an older remote workflow rooted at:

- example host: `gpub005.delta.ncsa.illinois.edu`
- repo root: `/u/lshi8/TokenCompressor`
- python: `/u/lshi8/miniconda3/envs/py313/bin/python`

The current local worktree is not clean. Do not push `main` blindly.

For the remote runs discussed on `2026-04-25`, the minimum remote tree should include at least:

- `run_sasrec_taobao_standard.py`
- `run_hstu_taobao_standard.py`
- `run_lru_taobao_standard.py`
- `run_longer_taobao_sample_softmax.py`
- `train_backbone_standard.py`
- `train_backbone_sample_softmax.py`
- `core/loo_dataset.py`
- `core/streaming_eval.py`
- `backbones/SASRec.py`
- `backbones/HSTU.py`
- `backbones/HSTUOfficialish.py`
- `backbones/HSTUResearchAligned.py`
- `backbones/modules.py`
- `backbones/patch.py`
- `id_patch/train_patch_first_order.py`
- `id_patch/train_Persrec.py`
- `scripts/remote/*`

The fast path is:

1. Push a dedicated branch that contains the remote launch helpers in `scripts/remote/` plus the actual runner/dependency files above.
2. On remote:

```bash
cd /u/lshi8/TokenCompressor
git fetch origin
git checkout <branch>
git pull --ff-only origin <branch>
bash scripts/remote/smoke_test_remote.sh
bash scripts/remote/launch_remote_round1_ml10m_original_table.sh
bash scripts/remote/launch_remote_round2_ml10m_original_prefix.sh
bash scripts/remote/launch_remote_round3_ml10m_persrec.sh
bash scripts/remote/launch_remote_round4_lru_backbones.sh
bash scripts/remote/launch_remote_round5_ml10m_longer.sh
```

Recommended execution order:

1. `smoke_test_remote.sh`
2. `launch_remote_round1_ml10m_original_table.sh`
3. `launch_remote_round2_ml10m_original_prefix.sh`
4. `launch_remote_round3_ml10m_persrec.sh`
5. `launch_remote_round4_lru_backbones.sh`
6. `launch_remote_round5_ml10m_longer.sh`

What each script does:

- `smoke_test_remote.sh`
  Runs file checks, import checks, `--help` checks for the patch/PersRec entrypoints, and a tiny one-epoch SASRec smoke run on `ml10m_loo202`.
- `launch_remote_round1_ml10m_original_table.sh`
  Launches the four `ML-10M / Original` jobs for the table:
  `SASRec/HSTU × full-view/short-view`.
- `launch_remote_round2_ml10m_original_prefix.sh`
  Launches the two `ML-10M / Original / Add prefix` jobs.
  This depends on the round1 full-view checkpoints already existing on remote.
- `launch_remote_round3_ml10m_persrec.sh`
  Launches `ML-10M / PersRec / SASRec` and `ML-10M / PersRec / HSTU`.
  This depends on the sampled-softmax warm-start checkpoints already being present on remote.
- `launch_remote_round4_lru_backbones.sh`
  Launches exploratory `LRU` backbone runs for `ML-10M` and `Xlong`, each with `full` and `short` training.
- `launch_remote_round5_ml10m_longer.sh`
  Launches the missing `ML-10M / others / LONGER` run using the current better recipe:
  `sampled-softmax + similarity`.

Queue helper:

- `queue_remote_round5_ml10m_longer_after_lru_ml10m_short.sh`
  Waits for `round4`'s `lru_ml10m_short_remote` process to finish, then launches `round5` on that freed GPU.
  This is the easiest way to append the missing table cell after the already-running remote scheduler.

Checkpoint prerequisites for `launch_remote_round3_ml10m_persrec.sh`:

- `$ROOT/checkpoints/sasrec_loo_sample_softmax/sasrec_ml10m_oldbest_recipe_softmax_sample_softmax_bs512_evalseed2026/sasrec_ml10m_loo202_seq200_dim128_L2_H1_best.pt`
- `$ROOT/checkpoints/hstu_loo_sample_softmax/hstu_true_mh_ml10m_sm1_sampledsoftmax_backbone_20260419/hstu_ml10m_loo202_seq202_dim128_L4_H4_best.pt`

If those paths differ on remote, override them with env vars:

```bash
SASREC_WARMSTART_CKPT=/path/to/sasrec_best.pt \
HSTU_WARMSTART_CKPT=/path/to/hstu_best.pt \
bash scripts/remote/launch_remote_round3_ml10m_persrec.sh
```

Current limitation:

- `Mamba4Rec` is not wired into this repo right now, so there is no remote launcher for it yet.
- `LRU + A2` is also not exposed through the current `patch_first_order` pipeline, so the provided `LRU` script covers backbone experiments only.
