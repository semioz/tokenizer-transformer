import argparse
import subprocess
from dataclasses import dataclass


TARGET_TOKENS = 67_108_864
CONTEXT_LENGTH = 256

TRAIN_PATH = "/data/tokenized/tinystories_train_ids.npy"
VAL_PATH = "/data/tokenized/tinystories_valid_ids.npy"

LR_BY_BATCH_SIZE = {
    1: 2e-4,
    8: 5e-4,
    32: 1e-3,
    64: 1.4e-3,
    128: 2e-3,
}


@dataclass(frozen=True)
class BatchRun:
    batch_size: int
    num_steps: int
    max_lr: float
    context_length: int = CONTEXT_LENGTH


def lr_label(max_lr: float) -> str:
    label = f"{max_lr:.1e}".replace(".0e", "e")
    return label.replace("e-0", "e-").replace("e+0", "e+")


def default_runs(target_tokens: int = TARGET_TOKENS, context_length: int = CONTEXT_LENGTH) -> list[BatchRun]:
    return [
        BatchRun(
            batch_size=batch_size,
            num_steps=target_tokens // (batch_size * context_length),
            max_lr=max_lr,
            context_length=context_length,
        )
        for batch_size, max_lr in LR_BY_BATCH_SIZE.items()
    ]


def build_modal_command(
    run: BatchRun,
    *,
    gpu: str = "B200",
    log_every: int = 50,
    eval_every: int = 512,
    save_every: int = 100_000,
) -> list[str]:
    lr = lr_label(run.max_lr)
    run_name = f"batch_size_bs_{run.batch_size}_lr_{lr}"
    checkpoint_dir = f"/checkpoints/batch_size/bs_{run.batch_size}_lr_{lr}"

    return [
        "uv", "run", "python", "scripts/modal_train.py", "train",
        "--gpu", gpu,
        "--train-path", TRAIN_PATH,
        "--val-path", VAL_PATH,
        "--vocab-size", "10000",
        "--context-length", str(run.context_length),
        "--d-model", "512",
        "--num-layers", "4",
        "--num-heads", "16",
        "--d-ff", "1344",
        "--rope-theta", "10000.0",
        "--batch-size", str(run.batch_size),
        "--num-steps", str(run.num_steps),
        "--max-lr", f"{run.max_lr:g}",
        "--warmup-steps", "200",
        "--weight-decay", "0.01",
        "--min-lr-ratio", "0.1",
        "--log-every", str(log_every),
        "--eval-every", str(eval_every),
        "--save-every", str(save_every),
        "--checkpoint-dir", checkpoint_dir,
        "--compile",
        "--wandb",
        "--wandb-run-name", run_name,
    ]


def print_commands(runs: list[BatchRun], gpu: str) -> None:
    for run in runs:
        tokens = run.batch_size * run.num_steps * run.context_length
        print(f"# batch_size={run.batch_size} steps={run.num_steps} lr={run.max_lr:g} tokens={tokens:,}")
        print(" ".join(build_modal_command(run, gpu=gpu)))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare/run Modal batch-size sweep commands.")
    parser.add_argument("--gpu", default="B200")
    parser.add_argument("--target-tokens", type=int, default=TARGET_TOKENS)
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH)
    parser.add_argument("--batch-size", type=int, default=None, help="Run/print only one batch size")
    parser.add_argument("--memory-probe", type=int, default=None, help="Print a short probe command for a candidate max batch size")
    parser.add_argument("--execute", action="store_true", help="Actually run the selected command")
    args = parser.parse_args()

    if args.memory_probe is not None:
        runs = [BatchRun(batch_size=args.memory_probe, num_steps=200, max_lr=1e-3, context_length=args.context_length)]
    else:
        runs = default_runs(args.target_tokens, args.context_length)

    if args.batch_size is not None:
        runs = [run for run in runs if run.batch_size == args.batch_size]
        if not runs:
            raise SystemExit(f"batch size {args.batch_size} is not configured")

    if not args.execute:
        print_commands(runs, args.gpu)
        return

    if len(runs) != 1:
        raise SystemExit("--execute requires --batch-size or --memory-probe")

    command = build_modal_command(runs[0], gpu=args.gpu)
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
