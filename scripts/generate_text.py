import argparse
from dataclasses import dataclass
from pathlib import Path

import torch

from cs336_basics.generation import generate
from cs336_basics.modules import TransformerLM
from cs336_basics.tokenizer import Tokenizer


END_OF_TEXT = "<|endoftext|>"


@dataclass(frozen=True)
class GenerationConfig:
    vocab_size: int
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10000.0


def model_kwargs(cfg: GenerationConfig, device: str) -> dict:
    return {
        "vocab_size": cfg.vocab_size,
        "context_length": cfg.context_length,
        "d_model": cfg.d_model,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "d_ff": cfg.d_ff,
        "rope_theta": cfg.rope_theta,
        "device": device,
    }


def clean_state_dict_keys(state_dict: dict) -> dict:
    return {
        key.removeprefix("_orig_mod."): value
        for key, value in state_dict.items()
    }


def get_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_tokenizer(tokenizer_dir: Path) -> Tokenizer:
    return Tokenizer.from_files(
        tokenizer_dir / "vocab.pkl",
        tokenizer_dir / "merges.pkl",
        special_tokens=[END_OF_TEXT],
    )


def load_model(checkpoint_path: Path, cfg: GenerationConfig, device: str) -> TransformerLM:
    model = TransformerLM(**model_kwargs(cfg, device=device))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = clean_state_dict_keys(checkpoint["model"])
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained TinyStories checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("artifacts/tinystories_bpe_10k"))
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer_dir)
    eos_token_id = tokenizer.bytes_to_id[END_OF_TEXT.encode("utf-8")]

    cfg = GenerationConfig(vocab_size=len(tokenizer.vocab))
    model = load_model(args.checkpoint, cfg, device)

    prompt_ids = tokenizer.encode(args.prompt)
    output_ids = generate(
        model,
        prompt_ids,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
        context_length=cfg.context_length,
    )

    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
