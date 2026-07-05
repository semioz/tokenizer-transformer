import argparse
import subprocess

import numpy as np
import modal

app = modal.App("cs336-train")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("einops>=0.8", "einx>=0.4", "jaxtyping>=0.3", "numpy>=2.4", "regex", "tiktoken", "tqdm", "wandb", "psutil")
    .workdir("/root")
    .add_local_dir("cs336_basics", "/root/cs336_basics")
    .add_local_dir("scripts", "/root/scripts")
    .add_local_dir("artifacts/tinystories_bpe_10k", "/root/artifacts/tinystories_bpe_10k")
)

data_vol = modal.Volume.from_name("cs336-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("cs336-checkpoints", create_if_missing=True)

END_OF_TEXT = "<|endoftext|>"
TINYSTORIES_URLS = {
    "train": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "valid": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
}

def _encode_chunk(chunk):
    from cs336_basics.tokenizer import Tokenizer

    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = Tokenizer.from_files(
            "/root/artifacts/tinystories_bpe_10k/vocab.pkl",
            "/root/artifacts/tinystories_bpe_10k/merges.pkl",
            special_tokens=[END_OF_TEXT],
        )
    return np.array(_TOKENIZER.encode(chunk), dtype=np.uint16)


_TOKENIZER = None


def _init_worker():
    from cs336_basics.tokenizer import Tokenizer

    global _TOKENIZER
    _TOKENIZER = Tokenizer.from_files(
        "/root/artifacts/tinystories_bpe_10k/vocab.pkl",
        "/root/artifacts/tinystories_bpe_10k/merges.pkl",
        special_tokens=[END_OF_TEXT],
    )


@app.function(image=image, volumes={"/data": data_vol}, timeout=3600, cpu=8, memory=16 * 1024)
def setup_data():
    import urllib.request
    from pathlib import Path
    from multiprocessing import Pool
    import numpy as np

    Path("/data/tokenized").mkdir(parents=True, exist_ok=True)

    for split, url in TINYSTORIES_URLS.items():
        out_npy = Path(f"/data/tokenized/tinystories_{split}_ids.npy")
        if out_npy.exists():
            print(f"{split}: already tokenized, skipping")
            continue

        tmp_txt = Path(f"/tmp/tinystories_{split}.txt")
        if not tmp_txt.exists():
            print(f"{split}: downloading from {url}")
            urllib.request.urlretrieve(url, tmp_txt)
            print(f"{split}: downloaded {tmp_txt.stat().st_size / 1e6:.1f} MB")

        chunks = []
        batch = []
        with open(tmp_txt, encoding="utf-8") as f:
            for line in f:
                batch.append(line)
                if len(batch) >= 10_000:
                    chunks.append("".join(batch))
                    batch = []
        if batch:
            chunks.append("".join(batch))

        print(f"{split}: {len(chunks)} chunks, encoding on 8 cores...")

        arrays = []
        with Pool(8, initializer=_init_worker) as pool:
            for i, arr in enumerate(pool.imap(_encode_chunk, chunks), start=1):
                arrays.append(arr)
                if i % 25 == 0 or i == len(chunks):
                    print(f"{split}: encoded {i}/{len(chunks)} chunks", flush=True)

        total_tokens = sum(len(a) for a in arrays)
        bytes_per_token = tmp_txt.stat().st_size / total_tokens
        print(f"{split}: {total_tokens} tokens ({bytes_per_token:.2f} bytes/token)")

        if bytes_per_token < 1.5:
            raise RuntimeError(
                f"Suspicious tokenization for {split}: {bytes_per_token:.2f} bytes/token. "
                "This likely means BPE special-token splitting is broken. Refusing to save bad data."
            )

        ids = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.uint16, shape=(total_tokens,))
        offset = 0
        for a in arrays:
            ids[offset:offset + len(a)] = a
            offset += len(a)
        ids.flush()
        print(f"{split}: saved {out_npy}")

        tmp_txt.unlink(missing_ok=True)

    data_vol.commit()
    print("data volume committed")


@app.function(image=image, volumes={"/data": data_vol}, timeout=300, cpu=1, memory=1024)
def clear_tokenized_data():
    """Delete tokenized TinyStories files from the Modal data volume."""
    from pathlib import Path

    tokenized_dir = Path("/data/tokenized")
    for path in tokenized_dir.glob("tinystories_*_ids.npy"):
        print(f"deleting {path}")
        path.unlink()
    data_vol.commit()
    print("tokenized data cleared")


def train_remote(train_args: list[str]) -> dict:
    """Run training on GPU. Checkpoints saved to /checkpoints volume."""
    import os

    cmd = ["python", "/root/scripts/train.py"] + train_args

    if not any(a.startswith("--checkpoint-dir") for a in train_args):
        cmd += ["--checkpoint-dir", "/checkpoints/run"]
    cmd += ["--device", "cuda"]

    if "--wandb" in cmd and "WANDB_API_KEY" not in os.environ:
        print("WARNING: --wandb passed but WANDB_API_KEY not set. Disabling wandb.")
        cmd.remove("--wandb")

    print(f"running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd="/root", capture_output=False, text=True)

    ckpt_vol.commit()
    print("checkpoints committed to volume")
    return {"returncode": result.returncode}


def _make_train_fn(gpu: str):
    """Create a train function bound to a specific GPU type."""
    return app.function(
        image=image,
        gpu=gpu,
        volumes={"/data": data_vol, "/checkpoints": ckpt_vol},
        timeout=7200,
        secrets=[modal.Secret.from_name("wandb")],
    )(train_remote)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    sub = sys.argv[1]
    rest = sys.argv[2:]

    if sub == "setup":
        with app.run():
            setup_data.remote()
        print("done: data tokenized and saved to cs336-data volume")

    elif sub == "train":
        gpu = "T4"
        rest_filtered = []
        i = 0
        while i < len(rest):
            if rest[i] == "--gpu":
                gpu = rest[i + 1]
                i += 2
            elif rest[i].startswith("--gpu="):
                gpu = rest[i].split("=", 1)[1]
                i += 1
            else:
                rest_filtered.append(rest[i])
                i += 1

        print(f"using gpu={gpu}")
        train_fn = _make_train_fn(gpu)

        with app.run():
            result = train_fn.remote(rest_filtered)
        print(f"training finished with returncode={result['returncode']}")

    else:
        print(f"unknown command: {sub}. Use 'setup' or 'train'.")
        sys.exit(1)
