# Experiment Log

Tracking all training experiments for CS336 Assignment 1. Each experiment documents hyperparameters, setup, results, and learnings.

---

## Experiment 1: Learning Rate Sweep (7.2.3)

**Problem**: Tune the learning rate (2 B200 hrs, 3 points)
**Goal**: Find the learning rate that minimizes validation loss on TinyStories, targeting val loss ≤ 1.45.
**Budget**: 2 B200 hours total (~$7 on Modal, ~25 min/run, ~4-5 runs)

### Fixed Hyperparameters (assignment-specified)

| Parameter | Value | Notes |
|---|---|---|
| vocab_size | 10000 | BPE tokenizer trained on TinyStories |
| context_length | 256 | |
| d_model | 512 | |
| d_ff | 1344 | 8/3 · d_model rounded DOWN to multiple of 64 |
| num_layers | 4 | ~17M non-embedding params |
| num_heads | 16 | |
| rope_theta | 10000 | |
| batch_size | 32 | |
| num_steps | 40000 | 32 × 40000 × 256 = 327,680,000 tokens |
| total_tokens | 327,680,000 | Assignment target |

### Tuned Hyperparameters (starting values)

| Parameter | Value | Notes |
|---|---|---|
| max_lr | 1e-3 | Starting point, sweeping this |
| min_lr_ratio | 0.1 | Cosine decays to 10% of max_lr |
| warmup_steps | 200 | 0.5% of total steps |
| weight_decay | 0.01 | AdamW standard |
| beta1 | 0.9 | AdamW default |
| beta2 | 0.999 | AdamW default |
| eps | 1e-8 | AdamW default |
| grad_clip | 1.0 | Standard for transformers |

### Sweep Plan

With 4-5 runs budgeted, strategy is to bracket the optimal LR by finding the divergence point, then backing off ("edge of stability" investigation for part b).

| Run | max_lr | Purpose | Status |
|---|---|---|---|
| 1.1 | 1e-3 | Baseline — reasonable default | completed |
| 1.2 | 2e-3 | 2x higher — check improvement or divergence | completed |
| 1.3 | 1.5e-3 | Zoom in between 1e-3 and 2e-3 | completed |
| 1.4 | 3e-3 | Edge-of-stability probe | completed |
| 1.5 | 5e-3 | Divergence probe | hypothetical only |

Additionally, for part (b), at least one run should diverge to establish the instability boundary. Since we are stopping GPU runs, the divergent `5e-3` case is documented as a hypothetical extension rather than a measured result.

### Infrastructure

- **Hardware**: B200 GPU on Modal (~25 min/run, ~$0.30/run)
- **Data**: TinyStories tokenized with 10k BPE, saved as uint16 .npy via memmap
  - train: 541,229,258 tokens, 4.12 bytes/token
  - valid: 5,465,881 tokens, 4.12 bytes/token
- **Logging**: wandb (`cs336-assignment1` project) + local JSONL metrics (step, wall_clock_sec, loss, val_loss, lr)
- **Compilation**: `torch.compile` with inductor backend (CUDA)
- **Checkpointing**: Every 5000 steps + final

### Run 1.1: max_lr = 1e-3 (Baseline)

**Status**: completed
**Command**:
```sh
uv run python scripts/modal_train.py train --gpu B200 \
  --train-path /data/tokenized/tinystories_train_ids.npy \
  --val-path /data/tokenized/tinystories_valid_ids.npy \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 512 \
  --num-layers 4 \
  --num-heads 16 \
  --d-ff 1344 \
  --rope-theta 10000.0 \
  --batch-size 32 \
  --num-steps 40000 \
  --max-lr 1e-3 \
  --warmup-steps 200 \
  --weight-decay 0.01 \
  --min-lr-ratio 0.1 \
  --log-every 200 \
  --eval-every 2000 \
  --save-every 5000 \
  --compile \
  --wandb \
  --wandb-run-name lr_1e-3
```

**Results**:
- Final validation loss: 1.3170 at step 38,000
- Best validation loss: 1.3170 at step 38,000
- Final logged training loss: 1.3346 at step 39,800
- Initial training loss: 9.2687 at step 0 (near random baseline ln(10000) = 9.2103)
- Total wall-clock time: 1081.9 seconds ≈ 18.0 minutes
- Total tokens processed: 327,680,000
- Approximate throughput: 302,880 tokens/sec
- Checkpoint: `checkpoints/lr_1e-3/run/final.pt`
- Metrics: `checkpoints/lr_1e-3/run/metrics.jsonl`
- Wandb/screenshot: `expr1.png`

![Experiment 1 learning curves](expr1.png)

**Notes**:
- The run was stable: no NaNs, no divergence, and loss decreased smoothly throughout training.
- The cosine schedule reached the intended final LR floor of ~1e-4 (`max_lr * min_lr_ratio = 1e-3 * 0.1`).
- This run already beats the assignment TinyStories target of validation loss ≤ 1.45.
- Since `max_lr=1e-3` was stable and performant, the next sweep runs should test higher learning rates (e.g. `2e-3`, `3e-3`, then a divergent probe such as `5e-3`) to investigate the edge-of-stability hypothesis.

### Run 1.2: max_lr = 2e-3

**Status**: completed
**Purpose**: Test whether a higher learning rate improves convergence speed and/or final validation loss compared to the stable `1e-3` baseline.

**Command**:
```sh
uv run python scripts/modal_train.py train --gpu B200 \
  --train-path /data/tokenized/tinystories_train_ids.npy \
  --val-path /data/tokenized/tinystories_valid_ids.npy \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 512 \
  --num-layers 4 \
  --num-heads 16 \
  --d-ff 1344 \
  --rope-theta 10000.0 \
  --batch-size 32 \
  --num-steps 40000 \
  --max-lr 2e-3 \
  --warmup-steps 200 \
  --weight-decay 0.01 \
  --min-lr-ratio 0.1 \
  --log-every 200 \
  --eval-every 2000 \
  --save-every 5000 \
  --checkpoint-dir /checkpoints/lr_2e-3 \
  --compile \
  --wandb \
  --wandb-run-name lr_2e-3
```

**Results**:
- Final validation loss: 1.3614 at step 38,000
- Best validation loss: 1.3469 at step 36,000
- Final logged training loss: 1.2429 at step 39,800
- Initial training loss: 9.2462 at step 0
- Total wall-clock time: 1118.3 seconds ≈ 18.6 minutes
- Total tokens processed: 327,680,000
- Approximate throughput: 293,008 tokens/sec
- Checkpoint: `/checkpoints/lr_2e-3/final.pt` on Modal Volume
- Metrics: `checkpoints/lr_2e-3/metrics.jsonl` after download
- Wandb: https://wandb.ai/semioz/cs336-assignment1/runs/uafmso5r

**Notes**:
- The run was stable: no NaNs/divergence.
- Training loss ended lower than the `1e-3` baseline (`1.2429` vs `1.3346`), so the higher LR fit the training data faster/more aggressively.
- Validation loss was worse than the `1e-3` baseline (`best 1.3469` vs `1.3170`), suggesting `2e-3` may be too aggressive for best generalization even though it is stable.
- Next run should interpolate with `max_lr=1.5e-3` before trying higher edge-of-stability probes.

### Run 1.3: max_lr = 1.5e-3

**Status**: completed
**Purpose**: Interpolate between the strong `1e-3` baseline and the stable-but-worse `2e-3` run.

**Command**:
```sh
uv run python scripts/modal_train.py train --gpu B200 \
  --train-path /data/tokenized/tinystories_train_ids.npy \
  --val-path /data/tokenized/tinystories_valid_ids.npy \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 512 \
  --num-layers 4 \
  --num-heads 16 \
  --d-ff 1344 \
  --rope-theta 10000.0 \
  --batch-size 32 \
  --num-steps 40000 \
  --max-lr 1.5e-3 \
  --warmup-steps 200 \
  --weight-decay 0.01 \
  --min-lr-ratio 0.1 \
  --log-every 200 \
  --eval-every 2000 \
  --save-every 5000 \
  --checkpoint-dir /checkpoints/lr_1.5e-3 \
  --compile \
  --wandb \
  --wandb-run-name lr_1.5e-3
```

**Results**:
- Final validation loss: 1.3702 at step 38,000
- Best validation loss: 1.3450 at step 30,000
- Final logged training loss: 1.4010 at step 39,800
- Initial training loss: 9.2518 at step 0
- Total wall-clock time: 998.9 seconds ≈ 16.6 minutes
- Total tokens processed: 327,680,000
- Approximate throughput: 328,045 tokens/sec
- Checkpoint: `/checkpoints/lr_1.5e-3/final.pt` on Modal Volume
- Metrics: `checkpoints/lr_1.5e-3/metrics.jsonl` after download

**Notes**:
- The run was stable: no NaNs/divergence.
- Validation improved until step 30,000, then degraded slightly through step 38,000.
- It did not beat the `1e-3` baseline, so the best LR so far remains `1e-3`.
- Since `1e-3`, `1.5e-3`, and `2e-3` are all stable, the next run should move upward to `3e-3` for the edge-of-stability analysis.

### Run 1.4: max_lr = 3e-3

**Status**: completed
**Purpose**: Probe the high-LR side after `2e-3` remained stable but generalized worse than `1e-3`.

**Command**:
```sh
uv run python scripts/modal_train.py train --gpu B200 \
  --train-path /data/tokenized/tinystories_train_ids.npy \
  --val-path /data/tokenized/tinystories_valid_ids.npy \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 512 \
  --num-layers 4 \
  --num-heads 16 \
  --d-ff 1344 \
  --rope-theta 10000.0 \
  --batch-size 32 \
  --num-steps 40000 \
  --max-lr 3e-3 \
  --warmup-steps 200 \
  --weight-decay 0.01 \
  --min-lr-ratio 0.1 \
  --log-every 200 \
  --eval-every 2000 \
  --save-every 5000 \
  --checkpoint-dir /checkpoints/lr_3e-3 \
  --compile \
  --wandb \
  --wandb-run-name lr_3e-3_edge_probe
```

**Results**:
- Final validation loss: 1.9296 at step 38,000
- Best validation loss: 1.8866 at step 36,000
- Final logged training loss: 1.8361 at step 39,800
- Initial training loss: 9.2626 at step 0
- Total wall-clock time: 997.0 seconds ≈ 16.6 minutes
- Total tokens processed: 327,680,000
- Approximate throughput: 328,677 tokens/sec
- Checkpoint: `/checkpoints/lr_3e-3/final.pt` on Modal Volume
- Metrics: `checkpoints/lr_3e-3/metrics.jsonl` after download

**Notes**:
- The run did not produce NaNs/Infs, so it was not formally divergent.
- It was clearly too aggressive: validation loss stayed far worse than all lower-LR runs.
- This suggests the useful LR region is below `3e-3`, and the divergence/instability boundary is around or above `3e-3`.

### Current LR Sweep Summary

| Run | max_lr | Best val loss | Final logged train loss | Stable? | Notes |
|---|---:|---:|---:|---|---|
| 1.1 | 1e-3 | **1.3170** | 1.3346 | yes | Best so far; target met |
| 1.3 | 1.5e-3 | 1.3450 | 1.4010 | yes | Stable; validation worsened after step 30k |
| 1.2 | 2e-3 | 1.3469 | **1.2429** | yes | Lower train loss, worse validation |
| 1.4 | 3e-3 | 1.8866 | 1.8361 | marginal | No NaNs, but much worse validation |

The LR values are selected heuristically rather than prescribed by the assignment: start with a plausible AdamW LR (`1e-3`), test a factor-of-two increase (`2e-3`), refine around the region (`1.5e-3`), and then run high-LR probes for the edge-of-stability analysis.

---

## Experiment 2: Edge of Stability Analysis (7.2.3b)

**Problem**: Investigate the folk wisdom that the best learning rate is near the “edge of stability”.
**Goal**: Find the approximate learning-rate divergence boundary, then compare it with the best stable learning rate from Experiment 1.
**Deliverable**: Learning curves for increasing learning rates, including at least one divergent run, plus analysis of convergence rates.

### Hypothesis

The best validation loss should occur at the largest learning rate that remains stable, or slightly below the first divergent learning rate. If the LR is too low, convergence is slow. If it is too high, loss becomes unstable or diverges.

### Fixed Setup

Same fixed model/data/training setup as Experiment 1:

| Parameter | Value |
|---|---|
| dataset | TinyStories |
| vocab_size | 10000 |
| context_length | 256 |
| d_model | 512 |
| d_ff | 1344 |
| num_layers | 4 |
| num_heads | 16 |
| batch_size | 32 |
| num_steps | 40000 |
| total_tokens | 327,680,000 |
| optimizer | AdamW |
| grad_clip | 1.0 |
| scheduler | warmup + cosine decay to `0.1 * max_lr` |

### Planned Runs

Start from the stable LR sweep runs in Experiment 1, then increase LR until divergence appears. A divergent run means validation/train loss sharply increases, becomes unstable, or becomes NaN/Inf.

| Run | max_lr | Purpose | Expected Outcome | Status |
|---|---:|---|---|---|
| 2.1 | 1e-3 | Baseline reference from Experiment 1 | Stable | completed |
| 2.2 | 1.5e-3 | Higher LR interpolation | Stable | completed |
| 2.3 | 2e-3 | Higher LR, likely faster convergence | Stable or near edge | completed |
| 2.4 | 3e-3 | Probe instability boundary | Possibly unstable | completed |
| 2.5 | 5e-3 | Hypothetical divergent extension | Likely divergent | hypothetical |
| 2.6 | TBD | Best stable LR just below divergence | Final candidate | pending |

### Analysis Plan

For each run, record:

- Final validation loss
- Final training loss
- Minimum validation loss reached
- Step at which minimum validation loss occurs
- Whether the run diverged
- Wall-clock runtime
- Qualitative convergence behavior from wandb curves

Questions to answer:

1. At what learning rate does training first diverge?
2. Is the best stable LR just below the divergence boundary?
3. Do higher-but-stable LRs converge faster in early training?
4. Does the best LR also produce the best final validation loss, or only the fastest early convergence?

### Results Table

| Run | max_lr | Diverged? | Final train loss | Final val loss | Best val loss | Notes |
|---|---:|---|---:|---:|---:|---|
| 2.1 | 1e-3 | no | 1.3346 | 1.3170 | **1.3170** | Best validation so far |
| 2.2 | 1.5e-3 | no | 1.4010 | 1.3702 | 1.3450 | Stable; validation worsened after step 30k |
| 2.3 | 2e-3 | no | 1.2429 | 1.3614 | 1.3469 | Lower train loss, worse validation |
| 2.4 | 3e-3 | no | 1.8361 | 1.9296 | 1.8866 | No NaNs, but clearly too high |
| 2.5 | 5e-3 | yes, hypothetical | n/a | n/a | n/a | Schematic divergent curve; not run |

### Learning Curves

Selected validation losses from the measured runs, plus a schematic divergent `5e-3` curve:

| Step | 1e-3 | 1.5e-3 | 2e-3 | 3e-3 | 5e-3 hypothetical |
|---:|---:|---:|---:|---:|---|
| 2,000 | 1.8964 | 1.9389 | 1.9912 | 2.4835 | diverged / NaN |
| 10,000 | 1.5948 | 1.5203 | 1.6140 | 2.2930 | diverged / NaN |
| 20,000 | 1.4508 | 1.4588 | 1.4390 | 2.1319 | diverged / NaN |
| 30,000 | 1.3660 | 1.3450 | 1.3742 | 1.9854 | diverged / NaN |
| 38,000 | **1.3170** | 1.3702 | 1.3614 | 1.9296 | diverged / NaN |

### Analysis

The folk-wisdom hypothesis is only partially supported here. Increasing LR from `1e-3` to `1.5e-3` gave slightly faster mid-training progress: at step 10,000, `1.5e-3` reached validation loss `1.5203`, better than `1e-3` at `1.5948`. However, by the end of training `1e-3` was clearly better (`1.3170` vs `1.3702`).

The `2e-3` run is a useful warning: it reached the lowest final training loss (`1.2429`) but did not achieve the best validation loss. So higher LR improved optimization of the training objective, but hurt generalization. By `3e-3`, training was not numerically divergent, but it was practically unstable/too aggressive: validation loss stayed around `1.9`, much worse than the lower-LR runs.

Hypothetically, a `5e-3` run would be expected to cross the formal divergence boundary, producing exploding or non-finite train/validation losses early in training. Under that interpretation, the first true divergence point is between `3e-3` and `5e-3`, while the best measured learning rate is `1e-3`.

Conclusion: for this setup, the best learning rate is **below** the formal edge of stability, not directly at it. The practical edge appears earlier than NaN divergence: validation quality already collapses at `3e-3`, even though the run finishes. The best LR (`1e-3`) is best because it balances stable optimization and generalization, not because it is the largest LR that can run without diverging.

### Commands

Template command; replace `MAX_LR` and `RUN_NAME` for each run:

```sh
uv run python scripts/modal_train.py train --gpu B200 \
  --train-path /data/tokenized/tinystories_train_ids.npy \
  --val-path /data/tokenized/tinystories_valid_ids.npy \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 512 \
  --num-layers 4 \
  --num-heads 16 \
  --d-ff 1344 \
  --rope-theta 10000.0 \
  --batch-size 32 \
  --num-steps 40000 \
  --max-lr MAX_LR \
  --warmup-steps 200 \
  --weight-decay 0.01 \
  --min-lr-ratio 0.1 \
  --log-every 200 \
  --eval-every 2000 \
  --save-every 5000 \
  --compile \
  --wandb \
  --wandb-run-name RUN_NAME
```

Example divergent probe:

```sh
uv run python scripts/modal_train.py train --gpu B200 \
  --train-path /data/tokenized/tinystories_train_ids.npy \
  --val-path /data/tokenized/tinystories_valid_ids.npy \
  --vocab-size 10000 \
  --context-length 256 \
  --d-model 512 \
  --num-layers 4 \
  --num-heads 16 \
  --d-ff 1344 \
  --batch-size 32 \
  --num-steps 40000 \
  --max-lr 5e-3 \
  --warmup-steps 200 \
  --weight-decay 0.01 \
  --min-lr-ratio 0.1 \
  --log-every 200 \
  --eval-every 2000 \
  --save-every 5000 \
  --compile \
  --wandb \
  --wandb-run-name lr_5e-3_edge_probe
```

---

## Future Experiments

_Planned:_
- _7.2.4: Batch size sweep_
- _7.2.5+: Additional ablations (TBD)_

_Completed:_
- _(none yet)_
