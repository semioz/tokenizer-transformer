import argparse
import random
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenizer import Tokenizer

END_OF_TEXT = "<|endoftext|>"

def load_tokenizer(artifact_dir: Path) -> Tokenizer | None:
    vocab_path = artifact_dir / "vocab.pkl"
    merges_path = artifact_dir / "merges.pkl"
    if not vocab_path.exists() or not merges_path.exists():
        print(f"skip_missing_tokenizer={artifact_dir}")
        return None
    return Tokenizer.from_files(vocab_path, merges_path, special_tokens=[END_OF_TEXT])


def sample_documents(path: Path, n: int, seed: int) -> list[str]:
    if not path.exists():
        print(f"skip_missing_dataset={path}")
        return []

    rng = random.Random(seed)
    samples = []
    seen = 0
    buffer = ""

    with path.open("r", encoding="utf-8") as f:
        while chunk := f.read(8 * 1024 * 1024):
            buffer += chunk
            docs = buffer.split(END_OF_TEXT)
            buffer = docs.pop()

            for doc in docs:
                if not doc:
                    continue
                seen += 1
                if len(samples) < n:
                    samples.append(doc)
                else:
                    j = rng.randrange(seen)
                    if j < n:
                        samples[j] = doc

    if buffer:
        seen += 1
        if len(samples) < n:
            samples.append(buffer)
        else:
            j = rng.randrange(seen)
            if j < n:
                samples[j] = buffer

    return samples


def compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> float:
    total_bytes = sum(len(doc.encode("utf-8")) for doc in docs)
    total_tokens = sum(len(tokenizer.encode(doc)) for doc in docs)
    return total_bytes / total_tokens


def throughput_bytes_per_second(tokenizer: Tokenizer, docs: list[str]) -> float:
    text = END_OF_TEXT.join(docs)
    num_bytes = len(text.encode("utf-8"))
    start = time.perf_counter()
    tokenizer.encode(text)
    elapsed = time.perf_counter() - start
    return num_bytes / elapsed


def save_token_ids_uint16(tokenizer: Tokenizer, input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        print(f"skip_missing_encode_input={input_path}")
        return

    token_count = 0
    with input_path.open("r", encoding="utf-8") as f:
        for _ in tokenizer.encode_iterable(f):
            token_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ids = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint16, shape=(token_count,))

    i = 0
    with input_path.open("r", encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            ids[i] = token_id
            i += 1
    ids.flush()
    print(f"saved_ids={output_path} tokens={token_count} dtype=uint16")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tinystories-train", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--tinystories-valid", type=Path, default=Path("data/TinyStoriesV2-GPT4-valid.txt"))
    parser.add_argument("--owt-train", type=Path, default=Path("data/owt_train.txt"))
    parser.add_argument("--owt-valid", type=Path, default=Path("data/owt_valid.txt"))
    parser.add_argument("--tinystories-tokenizer", type=Path, default=Path("artifacts/tinystories_bpe_10k"))
    parser.add_argument("--owt-tokenizer", type=Path, default=Path("artifacts/owt_bpe_32k"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tokenized"))
    parser.add_argument("--num-docs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encode-datasets", action="store_true")
    args = parser.parse_args()

    tiny_tokenizer = load_tokenizer(args.tinystories_tokenizer)
    owt_tokenizer = load_tokenizer(args.owt_tokenizer)

    tiny_docs = sample_documents(args.tinystories_train, args.num_docs, args.seed)
    owt_docs = sample_documents(args.owt_train, args.num_docs, args.seed)

    if tiny_tokenizer and tiny_docs:
        print(f"tinystories_on_tinystories_bytes_per_token={compression_ratio(tiny_tokenizer, tiny_docs):.4f}")
        tiny_throughput = throughput_bytes_per_second(tiny_tokenizer, tiny_docs)
        pile_seconds = 825 * 1024**3 / tiny_throughput
        print(f"tinystories_tokenizer_throughput_bytes_per_second={tiny_throughput:.2f}")
        print(f"pile_estimate_hours_with_tinystories_tokenizer={pile_seconds / 3600:.2f}")

    if owt_tokenizer and owt_docs:
        print(f"owt_on_owt_bytes_per_token={compression_ratio(owt_tokenizer, owt_docs):.4f}")
        owt_throughput = throughput_bytes_per_second(owt_tokenizer, owt_docs)
        pile_seconds = 825 * 1024**3 / owt_throughput
        print(f"owt_tokenizer_throughput_bytes_per_second={owt_throughput:.2f}")
        print(f"pile_estimate_hours_with_owt_tokenizer={pile_seconds / 3600:.2f}")

    if tiny_tokenizer and owt_docs:
        print(f"tinystories_on_owt_bytes_per_token={compression_ratio(tiny_tokenizer, owt_docs):.4f}")

    if args.encode_datasets:
        if tiny_tokenizer:
            save_token_ids_uint16(tiny_tokenizer, args.tinystories_train, args.output_dir / "tinystories_train_ids.npy")
            save_token_ids_uint16(tiny_tokenizer, args.tinystories_valid, args.output_dir / "tinystories_valid_ids.npy")
        if owt_tokenizer:
            save_token_ids_uint16(owt_tokenizer, args.owt_train, args.output_dir / "owt_train_ids.npy")
            save_token_ids_uint16(owt_tokenizer, args.owt_valid, args.output_dir / "owt_valid_ids.npy")


if __name__ == "__main__":
    main()
