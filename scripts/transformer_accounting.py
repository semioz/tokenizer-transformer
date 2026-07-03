from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int


def params(config: ModelConfig) -> dict[str, int]:
    embeddings = config.vocab_size * config.d_model
    attention = config.num_layers * 4 * config.d_model * config.d_model
    ffn = config.num_layers * 3 * config.d_model * config.d_ff
    norms = config.num_layers * 2 * config.d_model + config.d_model
    lm_head = config.vocab_size * config.d_model

    return {
        "token_embeddings": embeddings,
        "attention": attention,
        "ffn": ffn,
        "rmsnorm": norms,
        "lm_head": lm_head,
        "total": embeddings + attention + ffn + norms + lm_head,
    }


def flops(config: ModelConfig) -> dict[str, int]:
    s = config.context_length
    l = config.num_layers
    d = config.d_model
    f = config.d_ff
    v = config.vocab_size

    qkv_o_projections = l * 8 * s * d * d
    attention_scores = l * 2 * s * s * d
    attention_values = l * 2 * s * s * d
    ffn = l * 6 * s * d * f
    lm_head = 2 * s * d * v

    return {
        "attention_projections": qkv_o_projections,
        "attention_scores_qk": attention_scores,
        "attention_weighted_values": attention_values,
        "ffn": ffn,
        "lm_head": lm_head,
        "total": qkv_o_projections + attention_scores + attention_values + ffn + lm_head,
    }


def fmt_count(value: int) -> str:
    return f"{value:,}"


def fmt_billions(value: int) -> str:
    return f"{value / 1e9:.3f}B"


def fmt_tflops(value: int) -> str:
    return f"{value / 1e12:.3f}T"


def print_params(config: ModelConfig) -> None:
    p = params(config)
    print(f"\n{config.name} parameters")
    for key, value in p.items():
        print(f"  {key:24s} {fmt_count(value):>18s}  {value / p['total'] * 100:6.2f}%")

    bytes_fp32 = p["total"] * 4
    print(f"  fp32_memory_gb          {bytes_fp32 / 1e9:18.3f} GB")
    print(f"  fp32_memory_gib         {bytes_fp32 / 1024**3:18.3f} GiB")


def print_flops(config: ModelConfig) -> None:
    f = flops(config)
    print(f"\n{config.name} FLOPs, S={config.context_length}")
    for key, value in f.items():
        print(f"  {key:28s} {fmt_count(value):>22s}  {fmt_tflops(value):>10s}  {value / f['total'] * 100:6.2f}%")


def nearest_multiple_of_64(value: float) -> int:
    lower = int(value // 64) * 64
    upper = lower + 64
    return lower if value - lower < upper - value else upper


def main() -> None:
    vocab_size = 50_257
    context_length = 1_024

    configs = [
        ModelConfig("GPT-2 small-shaped", vocab_size, context_length, 12, 768, 12, nearest_multiple_of_64(8 / 3 * 768)),
        ModelConfig("GPT-2 medium-shaped", vocab_size, context_length, 24, 1024, 16, nearest_multiple_of_64(8 / 3 * 1024)),
        ModelConfig("GPT-2 large-shaped", vocab_size, context_length, 36, 1280, 20, nearest_multiple_of_64(8 / 3 * 1280)),
        ModelConfig("GPT-2 XL-shaped", vocab_size, context_length, 48, 1600, 25, 4288),
        ModelConfig("GPT-2 XL-shaped long context", vocab_size, 16_384, 48, 1600, 25, 4288),
    ]

    for config in configs:
        print_params(config)
        print_flops(config)


if __name__ == "__main__":
    main()
