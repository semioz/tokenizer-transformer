import argparse
import pickle
import time
from pathlib import Path

from cs336_basics.train_bpe import train_bpe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/owt_train.txt")
    parser.add_argument("--output", default="artifacts/owt_bpe_32k")
    parser.add_argument("--vocab-size", type=int, default=32_000)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    args = parser.parse_args()

    start = time.perf_counter()
    vocab, merges = train_bpe(args.input, args.vocab_size, args.special_token)
    elapsed = time.perf_counter() - start

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "vocab.pkl").open("wb") as f:
        pickle.dump(vocab, f)
    with (output_dir / "merges.pkl").open("wb") as f:
        pickle.dump(merges, f)

    longest = max(vocab.values(), key=len)
    print(f"time_seconds={elapsed:.2f}")
    print(f"longest_token={longest.decode('utf-8', errors='replace')!r}")
    print(f"longest_token_bytes={len(longest)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
