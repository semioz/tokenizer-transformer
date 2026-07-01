import argparse
import pickle
import time
import tracemalloc
from pathlib import Path

from cs336_basics.train_bpe import train_bpe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/TinyStoriesV2-GPT4-train.txt")
    parser.add_argument("--output", default="artifacts/tinystories_bpe_10k")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    args = parser.parse_args()

    tracemalloc.start()
    start = time.perf_counter()
    vocab, merges = train_bpe(args.input, args.vocab_size, ["<|endoftext|>"])
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "vocab.pkl").open("wb") as f:
        pickle.dump(vocab, f)
    with (output_dir / "merges.pkl").open("wb") as f:
        pickle.dump(merges, f)

    longest = max(vocab.values(), key=len)
    print(f"time_seconds={elapsed:.2f}")
    print(f"peak_tracemalloc_mib={peak / 1024 / 1024:.2f}")
    print(f"longest_token={longest.decode('utf-8', errors='replace')!r}")
    print(f"longest_token_bytes={len(longest)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
