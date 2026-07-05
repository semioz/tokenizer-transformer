import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.data_loader import get_batch
from cs336_basics.modules import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.utils import (
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)


def get_device() -> str:
    """Auto-detect: mps on Apple Silicon, cuda if available, else cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MetricsLogger:
    def __init__(self, metrics_path, use_wandb=False, wandb_project=None, wandb_run_name=None, config=None):
        self.metrics_path = metrics_path
        self.use_wandb = use_wandb
        self._wandb = None
        self.t0 = time.perf_counter()

        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = metrics_path.open("a", encoding="utf-8")

        if use_wandb:
            import wandb
            self._wandb = wandb
            wandb.init(project=wandb_project, config=config)
            if wandb_run_name:
                wandb.run.name = wandb_run_name

    def log(self, metrics: dict, step: int) -> None:
        """Add wall_clock_sec and step, write to JSONL + wandb."""
        entry = {"step": step, "wall_clock_sec": time.perf_counter() - self.t0, **metrics}
        self._fh.write(json.dumps(entry) + "\n")
        self._fh.flush()
        if self._wandb is not None:
            self._wandb.log(entry, step=step)

    def close(self) -> None:
        self._fh.close()
        if self._wandb is not None:
            self._wandb.finish()


@torch.no_grad()
def evaluate(model, val_data, batch_size, context_length, vocab_size, device, n_batches=5):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(val_data, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def train(cfg):
    device = get_device() if cfg.device == "auto" else cfg.device
    print(f"device={device}")

    # Memory-efficient loading: memmap lazily reads from disk on access.
    train_data = np.load(cfg.train_path, mmap_mode="r")
    val_data = np.load(cfg.val_path, mmap_mode="r")
    print(f"train_tokens={len(train_data)} val_tokens={len(val_data)} dtype={train_data.dtype}")

    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=cfg.rope_theta,
        device=device,
    )

    # torch.compile: inductor on cuda/cpu, aot_eager on mps (inductor unsupported).
    # ~20-40% speedup on backward pass. Compile happens lazily on first forward.
    if cfg.compile:
        backend = "aot_eager" if device == "mps" else "inductor"
        model = torch.compile(model, backend=backend)
        print(f"compiled model with backend={backend}")

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.max_lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )
    min_lr = cfg.max_lr * cfg.min_lr_ratio

    start_step = 0
    if cfg.resume is not None:
        start_step = load_checkpoint(cfg.resume, model, optimizer)
        print(f"resumed from {cfg.resume} at step {start_step}")

    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(
        metrics_path=cfg.checkpoint_dir / "metrics.jsonl",
        use_wandb=cfg.wandb,
        wandb_project=cfg.wandb_project,
        wandb_run_name=cfg.wandb_run_name,
        config=vars(cfg),
    )

    model.train()
    for step in range(start_step, cfg.num_steps):
        # LR schedule: T_c = num_steps (cosine annealing horizon = training length)
        lr = get_lr_cosine_schedule(step, cfg.max_lr, min_lr, cfg.warmup_steps, cfg.num_steps)
        optimizer.param_groups[0]["lr"] = lr

        x, y = get_batch(train_data, cfg.batch_size, cfg.context_length, device)

        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        loss.backward()

        gradient_clipping(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_every == 0:
            logger.log({"train/loss": loss.item(), "lr": lr}, step=step)
            print(f"step={step} loss={loss.item():.4f} lr={lr:.2e}")

        if step > 0 and step % cfg.eval_every == 0:
            val_loss = evaluate(
                model, val_data, cfg.batch_size, cfg.context_length, cfg.vocab_size, device
            )
            logger.log({"val/loss": val_loss}, step=step)
            print(f"step={step} val_loss={val_loss:.4f}")

        if step > 0 and step % cfg.save_every == 0:
            save_checkpoint(model, optimizer, step, cfg.checkpoint_dir / f"step_{step}.pt")
            print(f"saved checkpoint at step={step}")

    save_checkpoint(model, optimizer, cfg.num_steps, cfg.checkpoint_dir / "final.pt")
    print(f"saved final checkpoint at step={cfg.num_steps}")
    logger.close()


def main():
    p = argparse.ArgumentParser(description="Train a Transformer LM.")
    # data
    p.add_argument("--train-path", type=Path, required=True)
    p.add_argument("--val-path", type=Path, required=True)
    p.add_argument("--vocab-size", type=int, required=True)
    # model
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--d-ff", type=int, default=None, help="None = auto (8*d_model/3 rounded to 64)")
    p.add_argument("--rope-theta", type=float, default=10000.0)
    # optimizer
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=10000)
    p.add_argument("--max-lr", type=float, default=1e-3)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # loop control
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume from")
    # device
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|mps|cuda:N")
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    # wandb
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="cs336-assignment1")
    p.add_argument("--wandb-run-name", type=str, default=None)
    cfg = p.parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
