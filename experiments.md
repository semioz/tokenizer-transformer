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

## Experiment 3: Batch Size Sweep (batch_size_experiment)

**Problem**: Vary batch size from 1 up to the B200 memory limit and study the effect on training.
**Goal**: Compare learning curves across batch sizes while controlling for total token budget as much as possible.
**Budget**: 1 B200 hour.
**Deliverable**: Learning curves for multiple batch sizes, including typical sizes 64 and 128, plus a short discussion of throughput/generalization tradeoffs.

### Setup

Use the same model/data setup as the LR sweep:

| Parameter | Value |
|---|---:|
| dataset | TinyStories |
| vocab_size | 10000 |
| context_length | 256 |
| d_model | 512 |
| d_ff | 1344 |
| num_layers | 4 |
| num_heads | 16 |
| optimizer | AdamW |
| grad_clip | 1.0 |
| scheduler | warmup + cosine decay to `0.1 * max_lr` |

For fairer comparison, the main sweep keeps the token budget approximately fixed instead of keeping the number of optimizer steps fixed. The target budget is 67,108,864 tokens, about 20% of the full 40k-step batch-32 run.

```text
tokens = batch_size × num_steps × context_length
num_steps = target_tokens / (batch_size × context_length)
```

### Learning Rate Strategy

The best full-run LR so far is `1e-3` at batch size 32. For the first batch-size sweep, use conservative square-root LR scaling from that baseline:

```text
max_lr(B) = 1e-3 × sqrt(B / 32)
```

Because the LR sweep showed that very high LRs hurt validation loss, the high-batch LRs are capped/conservative. If a batch size looks obviously LR-limited or unstable, rerun only that batch size with an adjusted LR.

### Planned Runs

| Run | batch_size | num_steps | max_lr | target tokens | Purpose | Status |
|---|---:|---:|---:|---:|---|---|
| 3.1 | 1 | 262,144 | 2e-4 | 67,108,864 | Tiny batch / noisy gradients | pending |
| 3.2 | 8 | 32,768 | 5e-4 | 67,108,864 | Small batch | completed |
| 3.3 | 32 | 8,192 | 1e-3 | 67,108,864 | Baseline batch size | completed |
| 3.4 | 64 | 4,096 | 1.4e-3 | 67,108,864 | Typical larger batch | completed |
| 3.5 | 128 | 2,048 | 2e-3 | 67,108,864 | Typical large batch | completed |
| 3.6 | 256 | 200 | 1e-3 | n/a | Memory-limit/throughput probe | completed |

### Command Helper

Batch-size commands are generated by:

```sh
uv run python scripts/batch_size_experiment.py
```

Run one selected batch size:

```sh
uv run python scripts/batch_size_experiment.py --batch-size 64 --execute
```

Print a short memory-limit probe command:

```sh
uv run python scripts/batch_size_experiment.py --memory-probe 256
```

If the probe fits, try larger candidates such as 512 or 1024 until the run OOMs; record the largest successful batch size as the approximate B200 memory limit for this model/context length.

### Results Table

| Run | batch_size | max_lr | tokens | best val loss | final train loss | wall-clock | tokens/sec | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 3.1 | 1 | 2e-4 | 67,108,864 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 3.2 | 8 | 5e-4 | 67,108,864 | **1.4393** | 1.6477 | 472.1s | 142,160 | Best val so far, much slower |
| 3.3 | 32 | 1e-3 | 67,108,864 | 1.5285 | 1.5586 | 219.2s | 306,147 | Baseline short-token run |
| 3.4 | 64 | 1.4e-3 | 67,108,864 | 1.5620 | 1.6158 | 203.0s | 330,664 | Faster than bs=32, slightly worse val |
| 3.5 | 128 | 2e-3 | 67,108,864 | 1.6266 | 1.5837 | 243.8s | 275,222 | Latest segment; worse val than bs=32/64 |
| 3.6 | 256 | 1e-3 | 13,107,200 planned | n/a | 2.9662 | 36.1s logged | 273,763 logged | Fit probe only; no validation |

### Run 3.2: batch_size = 8

**Status**: completed  
**Command**:

```sh
uv run python scripts/batch_size_experiment.py --batch-size 8 --execute
```

**Results**:
- Total tokens processed: 67,108,864
- Final validation loss: 1.5191 at step 32,256
- Best validation loss: 1.4393 at step 27,648
- Final logged training loss: 1.6477 at step 32,750
- Initial training loss: 9.2490 at step 0
- Total wall-clock time: 472.1 seconds ≈ 7.9 minutes
- Approximate throughput: 142,160 tokens/sec
- Checkpoint: `/checkpoints/batch_size/bs_8_lr_5e-4/final.pt` on Modal Volume
- Metrics: `checkpoints/batch_size/bs_8_lr_5e-4/metrics.jsonl` after download

**Notes**:
- `batch_size=8` achieved the best validation loss so far under the fixed-token budget (`1.4393`), beating `batch_size=32` (`1.5285`).
- It was much less hardware-efficient: throughput was only `142,160` tokens/sec versus `306,147` for `batch_size=32` and `330,664` for `batch_size=64`.
- This is the clearest tradeoff so far: smaller batches give more optimizer updates for the same token budget and can improve validation, but wall-clock efficiency drops sharply.

### Run 3.3: batch_size = 32

**Status**: completed  
**Command**:

```sh
uv run python scripts/batch_size_experiment.py --batch-size 32 --execute
```

**Results**:
- Total tokens processed: 67,108,864
- Final validation loss: 1.5285 at step 7,680
- Best validation loss: 1.5285 at step 7,680
- Final logged training loss: 1.5586 at step 8,150
- Initial training loss: 9.2648 at step 0
- Total wall-clock time: 219.2 seconds ≈ 3.7 minutes
- Approximate throughput: 306,147 tokens/sec
- Checkpoint: `/checkpoints/batch_size/bs_32_lr_1e-3/final.pt` on Modal Volume
- Metrics: `checkpoints/batch_size/bs_32_lr_1e-3/metrics.jsonl` after download

**Notes**:
- This run is intentionally much shorter than the full LR sweep: 67.1M tokens instead of 327.7M tokens.
- The final validation loss is therefore worse than the full `batch_size=32` run (`1.5285` vs `1.3170`), which is expected because it trained on about 20% as many tokens.
- Use this run as the baseline for comparing other batch sizes under the same fixed-token budget.

### Run 3.4: batch_size = 64

**Status**: completed  
**Command**:

```sh
uv run python scripts/batch_size_experiment.py --batch-size 64 --execute
```

**Results**:
- Total tokens processed: 67,108,864
- Final validation loss: 1.5620 at step 3,584
- Best validation loss: 1.5620 at step 3,584
- Final logged training loss: 1.6158 at step 4,050
- Initial training loss: 9.2594 at step 0
- Total wall-clock time: 203.0 seconds ≈ 3.4 minutes
- Approximate throughput: 330,664 tokens/sec
- Checkpoint: `/checkpoints/batch_size/bs_64_lr_1.4e-3/final.pt` on Modal Volume
- Metrics: `checkpoints/batch_size/bs_64_lr_1.4e-3/metrics.jsonl` after download

**Notes**:
- With the same 67.1M-token budget, `batch_size=64` finished faster than `batch_size=32` (`203.0s` vs `219.2s`) and processed more tokens/sec (`330,664` vs `306,147`).
- Validation loss was slightly worse than the `batch_size=32` baseline (`1.5620` vs `1.5285`).
- This suggests the larger batch improved hardware efficiency, but the reduced number of optimizer updates and/or LR choice hurt validation quality slightly.

### Run 3.5: batch_size = 128

**Status**: completed  
**Command**:

```sh
uv run python scripts/batch_size_experiment.py --batch-size 128 --execute
```

**Results**:
- Total tokens processed: 67,108,864
- Final validation loss: 1.6266 at step 1,536
- Best validation loss: 1.6266 at step 1,536
- Final logged training loss: 1.5837 at step 2,000
- Initial training loss: 9.2609 at step 0
- Total wall-clock time: 243.8 seconds ≈ 4.1 minutes
- Approximate throughput: 275,222 tokens/sec
- Checkpoint: `/checkpoints/batch_size/bs_128_lr_2e-3/final.pt` on Modal Volume
- Metrics: `checkpoints/batch_size/bs_128_lr_2e-3/metrics.jsonl` after download

**Notes**:
- The metrics file contains two appended attempts because this checkpoint directory was reused. The table reports the latest segment.
- The earlier segment had nearly identical validation behavior: best validation loss `1.6252` at step 1,536.
- `batch_size=128` performed worse than both `batch_size=32` and `batch_size=64` on validation loss under the fixed-token budget.
- This strengthens the early trend that larger batches can reduce the number of optimizer updates enough to hurt validation quality, even when they process the same number of tokens.

### Run 3.6: batch_size = 256 memory probe

**Status**: completed  
**Command**:

```sh
uv run python scripts/batch_size_experiment.py --memory-probe 256 --execute
```

**Results**:
- Planned tokens: 13,107,200 (`256 × 200 × 256`)
- Validation loss: n/a; the probe uses only 200 steps and `eval_every=512`
- Final logged training loss: 2.9662 at step 150
- Initial training loss: 9.2636 at step 0
- Last logged wall-clock time: 36.1 seconds
- Approximate logged throughput: 273,763 tokens/sec through step 150
- Checkpoint: `/checkpoints/batch_size/bs_256_lr_1e-3/final.pt` on Modal Volume
- Metrics: `checkpoints/batch_size/bs_256_lr_1e-3/metrics.jsonl` after download

**Notes**:
- `batch_size=256` fits on the B200 for this model and context length.
- This is a memory/throughput probe, not a learning-curve run, because it is too short and has no validation evaluations.
- If a stricter memory-limit estimate is needed, probe larger batch sizes such as 512 and 1024 until OOM.

### Analysis Plan

Compare both optimization quality and hardware efficiency:

1. Validation loss vs tokens processed: does a larger batch converge better or worse for the same token budget?
2. Validation loss vs wall-clock: do larger batches win because they use the GPU more efficiently?
3. Training loss variance: do tiny batches show noisier learning curves?
4. Memory limit: what is the largest batch size that fits on B200 for this model and context length?
5. LR sensitivity: did any batch size need a rerun with a different LR?

Expected pattern: very small batches should have noisy gradients and poor GPU utilization; medium batches should train efficiently; very large batches may improve throughput but can reduce the number of optimizer updates for a fixed token budget and may require careful LR tuning.

---

## Experiment 4: Text Generation (generate)

**Problem**: Generate text from the trained TinyStories language model.
**Goal**: Use the decoder and trained checkpoint to produce a TinyStories-style sample, then comment on fluency and factors affecting output quality.

### Setup

| Parameter | Value |
|---|---|
| checkpoint | `checkpoints/lr_1e-3/run/final.pt` |
| source run | LR sweep, `max_lr=1e-3` |
| validation loss | 1.3170 |
| tokenizer | `artifacts/tinystories_bpe_10k` |
| prompt | `Once upon a time` |
| max_new_tokens | 256 |
| temperature | 0.8 |
| top_p | 0.9 |
| seed | 1 |
| device | CPU |

**Command**:

```sh
uv run python scripts/generate_text.py \
  --checkpoint checkpoints/lr_1e-3/run/final.pt \
  --prompt "Once upon a time" \
  --max-new-tokens 256 \
  --temperature 0.8 \
  --top-p 0.9 \
  --seed 1 \
  --device cpu
```

### Generated Text

The sample is 175 tokens including the first `<|endoftext|>` token.

```text
Once upon a time, there was a little boy named Tim. Tim loved to play with his toys. One day, he saw a big box with a label on it. The label had a picture of a ball. Tim wanted to play with the ball.
Tim took the ball and started to play. He kicked the ball and it went very far. Tim ran after the ball and caught it. He was so happy. But then, he saw a little girl who was sad. She did not have a toy to play with. Tim wanted to help, so he gave her the ball.
The little girl was very grateful. She said "thank you" to Tim. They played together with the ball and the ball. They had so much fun. Tim was happy to help his new friend. They both went home with big smiles on their faces.
<|endoftext|>
```

### Fluency Comment

The output is fluent and clearly resembles a TinyStories-style children’s story: it has a simple protagonist, a toy-centered conflict, a kind action, and a positive ending. Grammar is mostly correct, and the model produces coherent sentence boundaries and an explicit end-of-text token. The main weakness is mild repetition: “the ball and the ball” is unnatural, and the story is simple/redundant.

Factors affecting output quality:

1. **Checkpoint quality / validation loss**: this sample uses the best LR-sweep checkpoint (`val loss=1.3170`). Worse checkpoints, such as the `3e-3` run, would likely produce less coherent text.
2. **Decoding parameters**: temperature and top-p strongly affect the tradeoff between boring/repetitive and chaotic output. Here, `temperature=0.8` and `top_p=0.9` gave a stable, fluent sample.
3. **Training budget and model size**: the model trained on 327.7M tokens with the assignment TinyStories architecture. More data/steps or a larger model would likely improve long-range consistency and reduce repetition.

---

## Experiment 5: RMSNorm Ablation (layer_norm_ablation)

**Problem**: Remove all RMSNorm from the Transformer and train. What happens at the previous optimal LR? Can stability be recovered with a lower LR?
**Goal**: Compare learning curves of no-RMSNorm runs at the optimal LR and a lower LR, against the baseline.
**Budget**: 0.5 B200 hours.

### Setup

Same model/data as LR sweep. The only change is `--no-rmsnorm` which removes `ln1`, `ln2` (per-block) and `ln_final` (pre-LM-head).

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
| num_steps | 5000 |
| total_tokens | 40,960,000 |
| warmup_steps | 200 |
| weight_decay | 0.01 |
| grad_clip | 1.0 |

Shorter run (5000 steps) to fit 0.5 B200 hr budget.

### Planned Runs

| Run | rmsnorm | max_lr | Purpose | Status |
|---|---|---:|---|---|
| 5.1 | yes (baseline) | 1e-3 | Reference: baseline with RMSNorm | completed |
| 5.2 | no | 1e-3 | Ablation at optimal LR — expect instability | completed |
| 5.3 | no | 3e-4 | Ablation at lower LR — seek stability | completed |

### Commands

Run 5.2 (no RMSNorm, optimal LR):

```sh
uv run python scripts/modal_train.py train --gpu B200 \
  --train-path /data/tokenized/tinystories_train_ids.npy \
  --val-path /data/tokenized/tinystories_valid_ids.npy \
  --vocab-size 10000 --context-length 256 --d-model 512 --num-layers 4 \
  --num-heads 16 --d-ff 1344 --rope-theta 10000.0 --batch-size 32 \
  --num-steps 5000 --max-lr 1e-3 --warmup-steps 200 --weight-decay 0.01 \
  --min-lr-ratio 0.1 --log-every 100 --eval-every 1000 --save-every 5000 \
  --no-rmsnorm --checkpoint-dir /checkpoints/ablation_no_rmsnorm_lr_1e-3 \
  --compile --wandb --wandb-run-name ablation_no_rmsnorm_lr_1e-3
```

Run 5.3 (no RMSNorm, lower LR):

```sh
uv run python scripts/modal_train.py train --gpu B200 \
  --train-path /data/tokenized/tinystories_train_ids.npy \
  --val-path /data/tokenized/tinystories_valid_ids.npy \
  --vocab-size 10000 --context-length 256 --d-model 512 --num-layers 4 \
  --num-heads 16 --d-ff 1344 --rope-theta 10000.0 --batch-size 32 \
  --num-steps 5000 --max-lr 3e-4 --warmup-steps 200 --weight-decay 0.01 \
  --min-lr-ratio 0.1 --log-every 100 --eval-every 1000 --save-every 5000 \
  --no-rmsnorm --checkpoint-dir /checkpoints/ablation_no_rmsnorm_lr_3e-4 \
  --compile --wandb --wandb-run-name ablation_no_rmsnorm_lr_3e-4
```

### Results Table

| Run | rmsnorm | max_lr | best val loss | final train loss | diverged? | wall-clock | Notes |
|---|---|---:|---:|---:|---|---:|---|
| 5.1 | yes | 1e-3 | 1.6319 | 1.7130 | no | 156.4s | Baseline with RMSNorm (pre-norm) |
| 5.2 | no | 1e-3 | 1.7277 | 1.7102 | no | 149.2s | Stable but worse; high initial loss 15.78 |
| 5.3 | no | 3e-4 | 1.8804 | 1.8503 | no | 125.3s | Lower LR did NOT help; slower and worse |

### Run 5.2: no RMSNorm, max_lr = 1e-3

**Status**: completed
**Command**:

```sh
uv run python scripts/modal_train.py train --gpu B200 \
  --train-path /data/tokenized/tinystories_train_ids.npy \
  --val-path /data/tokenized/tinystories_valid_ids.npy \
  --vocab-size 10000 --context-length 256 --d-model 512 --num-layers 4 \
  --num-heads 16 --d-ff 1344 --rope-theta 10000.0 --batch-size 32 \
  --num-steps 5000 --max-lr 1e-3 --warmup-steps 200 --weight-decay 0.01 \
  --min-lr-ratio 0.1 --log-every 100 --eval-every 1000 --save-every 5000 \
  --no-rmsnorm --checkpoint-dir /checkpoints/ablation_no_rmsnorm_lr_1e-3 \
  --compile --wandb --wandb-run-name ablation_no_rmsnorm_lr_1e-3
```

**Results**:
- Total tokens processed: 40,960,000
- Best validation loss: 1.7277 at step 4,000
- Final validation loss: 1.7277 at step 4,000
- Final logged training loss: 1.7102 at step 4,900
- Initial training loss: 15.7777 at step 0
- Total wall-clock time: 149.2 seconds ≈ 2.5 minutes
- Checkpoint: `/checkpoints/ablation_no_rmsnorm_lr_1e-3/final.pt` on Modal Volume
- Metrics: `checkpoints/ablation_no_rmsnorm_lr_1e-3/metrics.jsonl` after download

**Notes**:
- Surprisingly, training did **not** diverge at `lr=1e-3` without RMSNorm.
- The initial training loss was much higher than the baseline (`15.78` vs `9.25`), indicating unnormalized activations produce very poor initial logits.
- Training loss decreased steadily but was noisier than the baseline, and validation loss converged much higher (`1.73` vs expected `~1.5` for the baseline at the same step count).
- This suggests RMSNorm is not strictly required for numerical stability at this LR, but it is critical for convergence quality and speed: without it, the model trains slower and reaches worse validation loss.

### Run 5.3: no RMSNorm, max_lr = 3e-4

**Status**: completed
**Command**:

```sh
uv run python scripts/modal_train.py train --gpu B200 \
  --train-path /data/tokenized/tinystories_train_ids.npy \
  --val-path /data/tokenized/tinystories_valid_ids.npy \
  --vocab-size 10000 --context-length 256 --d-model 512 --num-layers 4 \
  --num-heads 16 --d-ff 1344 --rope-theta 10000.0 --batch-size 32 \
  --num-steps 5000 --max-lr 3e-4 --warmup-steps 200 --weight-decay 0.01 \
  --min-lr-ratio 0.1 --log-every 100 --eval-every 1000 --save-every 5000 \
  --no-rmsnorm --checkpoint-dir /checkpoints/ablation_no_rmsnorm_lr_3e-4 \
  --compile --wandb --wandb-run-name ablation_no_rmsnorm_lr_3e-4
```

**Results** (latest segment):
- Total tokens processed: 40,960,000
- Best validation loss: 1.8804 at step 4,000
- Final validation loss: 1.8804 at step 4,000
- Final logged training loss: 1.8503 at step 4,500
- Initial training loss: 16.5494 at step 0
- Total wall-clock time: 125.3 seconds
- Checkpoint: `/checkpoints/ablation_no_rmsnorm_lr_3e-4/final.pt` on Modal Volume
- Metrics: `checkpoints/ablation_no_rmsnorm_lr_3e-4/metrics.jsonl` after download

**Notes**:
- Lowering the LR did **not** help. `lr=3e-4` without RMSNorm was worse than `lr=1e-3` without RMSNorm (`1.8804` vs `1.7277` best val loss).
- This confirms that the problem without RMSNorm is not just "LR too high" — it's a fundamental deficit in activation normalization. Lowering LR simply makes the model train slower without fixing the root cause.
- Both no-RMSNorm runs had very high initial loss (`16-17` vs `9.25` baseline), showing unnormalized activations produce poor initial logits regardless of LR.

### Run 5.1: baseline with RMSNorm (pre-norm), max_lr = 1e-3

**Status**: completed (also serves as Run 6.1 for Experiment 6)
**Command**:

```sh
uv run python scripts/modal_train.py train --gpu B200 \
  --train-path /data/tokenized/tinystories_train_ids.npy \
  --val-path /data/tokenized/tinystories_valid_ids.npy \
  --vocab-size 10000 --context-length 256 --d-model 512 --num-layers 4 \
  --num-heads 16 --d-ff 1344 --rope-theta 10000.0 --batch-size 32 \
  --num-steps 5000 --max-lr 1e-3 --warmup-steps 200 --weight-decay 0.01 \
  --min-lr-ratio 0.1 --log-every 100 --eval-every 1000 --save-every 5000 \
  --checkpoint-dir /checkpoints/ablation_pre_norm_lr_1e-3 \
  --compile --wandb --wandb-run-name ablation_pre_norm_lr_1e-3
```

**Results**:
- Best validation loss: 1.6319 at step 4,000
- Final validation loss: 1.6319 at step 4,000
- Final logged training loss: 1.7130 at step 4,900
- Initial training loss: 9.2446 at step 0
- Total wall-clock time: 156.4 seconds
- Metrics: `checkpoints/ablation_pre_norm_lr_1e-3/metrics.jsonl`

### RMSNorm Ablation Analysis

| Run | rmsnorm | lr | best val | initial train | wall-clock |
|---|---|---:|---:|---:|---:|
| 5.1 (baseline) | yes | 1e-3 | **1.6319** | 9.24 | 156.4s |
| 5.2 | no | 1e-3 | 1.7277 | 15.78 | 149.2s |
| 5.3 | no | 3e-4 | 1.8804 | 17.38 | 125.3s |

1. Does removing RMSNorm cause divergence at the previously optimal LR? — **No**, but convergence quality degrades significantly. Best val loss went from `1.63` to `1.73`.
2. Can a lower LR recover stability without RMSNorm? — **No**. Lowering LR to `3e-4` made things worse (`1.88`), not better. The problem is not "LR too high" — it's a fundamental lack of activation normalization.
3. RMSNorm's impact: RMSNorm is not strictly required for numerical stability at this scale, but it is critical for convergence quality. Without it, initial loss is much higher (`15-17` vs `9.25`), meaning unnormalized activations produce poor initial logits. The model can still learn, but slower and to a worse minimum. Lowering LR does not fix this — it just slows training further without addressing the root cause.

---

## Experiment 6: Pre-norm vs Post-norm Ablation (pre_norm_ablation)

**Problem**: Modify pre-norm Transformer into post-norm and train. Compare learning curves.
**Goal**: Show that the position of layer normalization matters.
**Budget**: 0.5 B200 hours.

### Architecture Difference

Pre-norm (default, our baseline):
```text
z = x + Attn(RMSNorm(x))
y = z + FFN(RMSNorm(z))
```

Post-norm (original Transformer):
```text
z = RMSNorm(x + Attn(x))
y = RMSNorm(z + FFN(z))
```

In post-norm, normalization happens **after** the residual addition, which makes gradient flow through the residual path harder. Pre-norm normalizes **before** each sublayer, keeping the residual stream clean.

### Setup

Same model/data as Experiment 5. 5000 steps, `lr=1e-3`, `batch_size=32`.

| Run | norm type | max_lr | Purpose | Status |
|---|---|---:|---|---|
| 6.1 | pre-norm | 1e-3 | Baseline reference (same as Run 5.1) | completed |
| 6.2 | post-norm | 1e-3 | Post-norm at optimal LR | completed |

### Commands

Run 6.2 (post-norm):

```sh
uv run python scripts/modal_train.py train --gpu B200 \
  --train-path /data/tokenized/tinystories_train_ids.npy \
  --val-path /data/tokenized/tinystories_valid_ids.npy \
  --vocab-size 10000 --context-length 256 --d-model 512 --num-layers 4 \
  --num-heads 16 --d-ff 1344 --rope-theta 10000.0 --batch-size 32 \
  --num-steps 5000 --max-lr 1e-3 --warmup-steps 200 --weight-decay 0.01 \
  --min-lr-ratio 0.1 --log-every 100 --eval-every 1000 --save-every 5000 \
  --post-norm --checkpoint-dir /checkpoints/ablation_post_norm_lr_1e-3 \
  --compile --wandb --wandb-run-name ablation_post_norm_lr_1e-3
```

Run 6.1 (pre-norm baseline, same as 5.1):

```sh
uv run python scripts/modal_train.py train --gpu B200 \
  --train-path /data/tokenized/tinystories_train_ids.npy \
  --val-path /data/tokenized/tinystories_valid_ids.npy \
  --vocab-size 10000 --context-length 256 --d-model 512 --num-layers 4 \
  --num-heads 16 --d-ff 1344 --rope-theta 10000.0 --batch-size 32 \
  --num-steps 5000 --max-lr 1e-3 --warmup-steps 200 --weight-decay 0.01 \
  --min-lr-ratio 0.1 --log-every 100 --eval-every 1000 --save-every 5000 \
  --checkpoint-dir /checkpoints/ablation_pre_norm_lr_1e-3 \
  --compile --wandb --wandb-run-name ablation_pre_norm_lr_1e-3
```

### Results Table

| Run | norm type | max_lr | best val loss | final train loss | diverged? | wall-clock | Notes |
|---|---|---:|---:|---:|---|---:|---|
| 6.1 | pre-norm | 1e-3 | 1.6319 | 1.7130 | no | 156.4s | Baseline |
| 6.2 | post-norm | 1e-3 | 1.7364 | 1.6356 | no | 159.9s | Stable but worse than pre-norm |

### Run 6.2: post-norm, max_lr = 1e-3

**Status**: completed
**Command**:

```sh
uv run python scripts/modal_train.py train --gpu B200 \
  --train-path /data/tokenized/tinystories_train_ids.npy \
  --val-path /data/tokenized/tinystories_valid_ids.npy \
  --vocab-size 10000 --context-length 256 --d-model 512 --num-layers 4 \
  --num-heads 16 --d-ff 1344 --rope-theta 10000.0 --batch-size 32 \
  --num-steps 5000 --max-lr 1e-3 --warmup-steps 200 --weight-decay 0.01 \
  --min-lr-ratio 0.1 --log-every 100 --eval-every 1000 --save-every 5000 \
  --post-norm --checkpoint-dir /checkpoints/ablation_post_norm_lr_1e-3 \
  --compile --wandb --wandb-run-name ablation_post_norm_lr_1e-3
```

**Results**:
- Total tokens processed: 40,960,000
- Best validation loss: 1.7364 at step 4,000
- Final validation loss: 1.7364 at step 4,000
- Final logged training loss: 1.6356 at step 4,900
- Initial training loss: 9.2561 at step 0
- Total wall-clock time: 159.9 seconds ≈ 2.7 minutes
- Checkpoint: `/checkpoints/ablation_post_norm_lr_1e-3/final.pt` on Modal Volume
- Metrics: `checkpoints/ablation_post_norm_lr_1e-3/metrics.jsonl` after download

**Notes**:
- Post-norm did **not** diverge at `lr=1e-3`, unlike what the original Transformer literature might suggest.
- Initial training loss (`9.26`) was normal, unlike the no-RMSNorm ablation (`15.78`), because RMSNorm is still present — just moved to after the residual.
- Validation loss (`1.74`) is worse than the no-RMSNorm run at the same LR (`1.73`), suggesting post-norm hurts convergence quality even when normalization exists.
- This is surprising: post-norm should be better than no-norm, but at this short run length it's comparable or slightly worse. The pre-norm baseline (Run 6.1) is needed for a fair comparison.

### Pre-norm vs Post-norm Analysis

| Run | norm type | best val | final train | wall-clock |
|---|---|---:|---:|---:|
| 6.1 (pre-norm) | pre-norm | **1.6319** | 1.7130 | 156.4s |
| 6.2 (post-norm) | post-norm | 1.7364 | **1.6356** | 159.9s |

1. Does post-norm diverge or train worse than pre-norm at the same LR? — Post-norm did **not** diverge, but validation loss was clearly worse (`1.74` vs `1.63`). Interestingly, post-norm achieved lower training loss (`1.64` vs `1.71`), suggesting it fits the training data better but generalizes worse.
2. Is post-norm stable but slower, or does it actually diverge? — Post-norm was stable at `lr=1e-3`. No NaNs or divergence. However, it trains to worse validation loss.
3. Why pre-norm is better: In pre-norm, the residual stream is never normalized directly, so gradient can flow through the residual path without being rescaled. In post-norm, every residual addition is followed by normalization, which reshapes the gradient signal at each layer. This makes optimization harder and can lead to worse generalization even when training loss looks good.

### Combined Summary: Layer Normalization Matters

Across both ablations (Experiments 5 and 6), the ranking at 5000 steps is:

| config | best val loss |
|---|---:|
| pre-norm with RMSNorm (baseline) | **1.6319** |
| no RMSNorm, lr=1e-3 | 1.7277 |
| post-norm with RMSNorm, lr=1e-3 | 1.7364 |
| no RMSNorm, lr=3e-4 | 1.8804 |

RMSNorm is essential for good convergence quality, and its **position** matters: pre-norm (normalizing before sublayers) works better than post-norm (normalizing after residual additions). Removing RMSNorm entirely hurts convergence but does not cause divergence at this scale. Lowering LR without RMSNorm makes things worse, not better.

---

## Experiment 7: Position Embedding Ablation — RoPE vs NoPE

**Problem**: Ablation 2: position embeddings / `no_pos_emb` (0.5 B200 hrs, 1 point)
**Goal**: Compare the base Transformer with RoPE against the same model with no positional information at all (NoPE).
**Deliverable**: Learning curve comparing RoPE and NoPE validation performance.

### Implementation

RoPE is disabled by passing `rope_theta=None` into the model. The attention module already skips rotary embeddings when no RoPE module is constructed, so NoPE keeps the causal mask but removes all explicit positional embedding information.

Even without RoPE, the decoder-only Transformer can infer some position information from the causal mask. Position 0 can only attend to itself, position 1 can attend to two tokens, position 2 to three tokens, and so on. This gives each position a different visibility pattern, so the model can indirectly learn whether it is early or late in the sequence and which tokens came before it. However, this signal is weaker than RoPE because RoPE gives attention direct relative-position information, while NoPE must recover it indirectly from the triangular mask and context.

Training flag:

```sh
--no-rope
```

### Comparison Setup

Use the same 5000-step ablation setup as Experiments 5 and 6:

| Parameter | Value |
|---|---:|
| vocab size | 10,000 |
| context length | 256 |
| d_model | 512 |
| layers | 4 |
| heads | 16 |
| d_ff | 1344 |
| batch size | 32 |
| max LR | 1e-3 |
| steps | 5,000 |
| warmup | 200 |
| eval every | 1,000 |

### Run 7.1: RoPE Baseline

Reusing the pre-norm/RMSNorm baseline from Run 5.1 / Run 6.1:

| step | val loss |
|---:|---:|
| 1000 | 2.5662 |
| 2000 | 2.0914 |
| 3000 | 1.8089 |
| 4000 | 1.6319 |

Best validation loss: **1.6319** at step 4000.

### Run 7.2: NoPE

Command:

```sh
uv run python scripts/modal_train.py train --gpu B200 \
  --train-path /data/tokenized/tinystories_train_ids.npy \
  --val-path /data/tokenized/tinystories_valid_ids.npy \
  --vocab-size 10000 --context-length 256 --d-model 512 --num-layers 4 \
  --num-heads 16 --d-ff 1344 --rope-theta 10000.0 --batch-size 32 \
  --num-steps 5000 --max-lr 1e-3 --warmup-steps 200 --weight-decay 0.01 \
  --min-lr-ratio 0.1 --log-every 100 --eval-every 1000 --save-every 5000 \
  --no-rope --checkpoint-dir /checkpoints/ablation_nope_lr_1e-3 \
  --compile --wandb --wandb-run-name ablation_nope_lr_1e-3
```

**Results**:

| step | train loss | val loss |
|---:|---:|---:|
| 1000 | 2.4049 | 2.2494 |
| 2000 | 1.9282 | 2.0012 |
| 3000 | 1.9373 | 1.8843 |
| 4000 | 1.8048 | 1.7748 |
| 4900 | 1.8361 | - |

- Best validation loss: **1.7748** at step 4000
- Final logged training loss: **1.8361** at step 4900
- Initial training loss: **9.2863** at step 0
- Total wall-clock time: **136.4 seconds** ≈ 2.3 minutes

### RoPE vs NoPE Learning Curve

| step | RoPE val loss | NoPE val loss | better |
|---:|---:|---:|---|
| 1000 | 2.5662 | **2.2494** | NoPE |
| 2000 | 2.0914 | **2.0012** | NoPE |
| 3000 | **1.8089** | 1.8843 | RoPE |
| 4000 | **1.6319** | 1.7748 | RoPE |

### Analysis

NoPE trains and improves normally, so explicit positional embeddings are not strictly required for this small decoder-only model to learn TinyStories. The causal mask gives the model an indirect position signal because each timestep has a different visible prefix length. Early in training, NoPE even has slightly better validation loss than RoPE, which may be because it has a simpler attention computation to optimize initially.

However, RoPE becomes better as training progresses. By step 4000, RoPE reaches `1.6319` validation loss while NoPE reaches `1.7748`, a gap of `0.1429`. This suggests the causal mask alone provides enough weak positional information to learn, but RoPE gives a stronger and more useful relative-position signal once the model has learned the basics. The final NoPE training loss is also worse than the RoPE baseline, so the gap is not just overfitting; NoPE is optimizing worse too.

Conclusion: NoPE is viable but inferior here. The causal mask lets the model infer some position information, but explicit RoPE substantially improves later-stage convergence and validation quality.

---

## Experiment 8: Feed-Forward Ablation — SwiGLU vs SiLU

**Problem**: Ablation 3: SwiGLU vs SiLU / `swiglu_ablation` (0.5 B200 hrs, 1 point)
**Goal**: Test whether the gating mechanism in SwiGLU improves performance compared to a plain SiLU feed-forward network with approximately matched parameter count.
**Deliverable**: Learning curve comparing SwiGLU and SiLU FFNs, plus discussion.

### Explanation

The baseline feed-forward network uses SwiGLU:

```text
SwiGLU(x) = W2( SiLU(W1 x) ⊙ W3 x )
```

This has two inner projections. `SiLU(W1 x)` computes activated features, while `W3 x` acts like a learned gate/value stream. Multiplying them lets the FFN conditionally amplify or suppress features before projecting back to `d_model`.

The ablation replaces this with a plain SiLU FFN:

```text
FFN_SiLU(x) = W2( SiLU(W1 x) )
```

This removes the gate and uses only one inner feature stream. Because SwiGLU has three matrices while SiLU has two, the SiLU baseline should use a larger hidden dimension, `d_ff = 4 * d_model`, to approximately match the parameter count of SwiGLU with `d_ff ≈ 8/3 * d_model`. This makes the comparison mostly about the effect of gating, not model size.

### Implementation

Training flag:

```sh
--ffn-type swiglu  # default
--ffn-type silu    # ablation
```

For the matched-parameter SiLU run, set `--d-ff 2048` because `4 * d_model = 4 * 512 = 2048`.

### Run 8.1: SwiGLU Baseline

Reusing the pre-norm/RMSNorm/SwiGLU baseline from Run 5.1 / Run 6.1:

| step | val loss |
|---:|---:|
| 1000 | 2.5662 |
| 2000 | 2.0914 |
| 3000 | 1.8089 |
| 4000 | 1.6319 |

Best validation loss: **1.6319** at step 4000.

### Run 8.2: SiLU FFN

Command:

```sh
uv run python scripts/modal_train.py train --gpu B200 \
  --train-path /data/tokenized/tinystories_train_ids.npy \
  --val-path /data/tokenized/tinystories_valid_ids.npy \
  --vocab-size 10000 --context-length 256 --d-model 512 --num-layers 4 \
  --num-heads 16 --d-ff 2048 --rope-theta 10000.0 --batch-size 32 \
  --num-steps 5000 --max-lr 1e-3 --warmup-steps 200 --weight-decay 0.01 \
  --min-lr-ratio 0.1 --log-every 100 --eval-every 1000 --save-every 5000 \
  --ffn-type silu --checkpoint-dir /checkpoints/ablation_silu_ffn_lr_1e-3 \
  --compile --wandb --wandb-run-name ablation_silu_ffn_lr_1e-3
```

**Results**:

| step | train loss | val loss |
|---:|---:|---:|
| 1000 | 2.1290 | 2.2696 |
| 2000 | 1.8287 | 1.9141 |
| 3000 | 1.8344 | 1.7906 |
| 4000 | 1.7544 | 1.7137 |
| 4900 | 1.6635 | - |

- Best validation loss: **1.7137** at step 4000
- Final logged training loss: **1.6635** at step 4900
- Initial training loss: **9.2413** at step 0
- Total wall-clock time: **202.0 seconds** ≈ 3.4 minutes

### SwiGLU vs SiLU Learning Curve

| step | SwiGLU val loss | SiLU val loss | better |
|---:|---:|---:|---|
| 1000 | 2.5662 | **2.2696** | SiLU |
| 2000 | 2.0914 | **1.9141** | SiLU |
| 3000 | 1.8089 | **1.7906** | SiLU |
| 4000 | **1.6319** | 1.7137 | SwiGLU |

### Analysis

The matched-parameter SiLU FFN trains successfully and is actually better than SwiGLU early in the run. At 1000-3000 steps, SiLU has lower validation loss, suggesting the simpler non-gated FFN can optimize quickly at the start despite lacking multiplicative gating.

By step 4000, however, SwiGLU overtakes SiLU. SwiGLU reaches `1.6319` validation loss while SiLU reaches `1.7137`, a gap of `0.0818`. This suggests the gate becomes useful later in training: once the model has learned basic token patterns, the multiplicative interaction in SwiGLU helps represent richer conditional features than a plain SiLU MLP with similar parameter count.

Interestingly, SiLU has lower final training loss (`1.6635`) than the SwiGLU baseline (`1.7130`) but worse validation loss. This may indicate that the larger plain SiLU FFN fits the training batches well but generalizes slightly worse than the gated SwiGLU FFN.

Conclusion: the SiLU FFN is a strong baseline and trains faster early, but SwiGLU gives better validation quality by the end of this 5000-step ablation. With approximately matched parameter counts, the gating mechanism appears beneficial for generalization and later-stage convergence.

---

## Future Experiments

_Planned:_
- _Complete Run 8.2 SiLU FFN and add SwiGLU vs SiLU learning curve._

_Completed:_
- _(none yet)_
