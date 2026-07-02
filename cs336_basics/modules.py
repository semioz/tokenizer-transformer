import torch
from torch import nn
import math
from einops import rearrange

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        std = math.sqrt(2 / (in_features + out_features))
        # modifies the tensor inplace 
        # normal distribution with mean 0 and std std but values are clipped/resampled to stay between [-3*std, 3*std]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        # upcasting for numerical stability when squaring and averaging, especially for low-precision inputs
        x = x.to(torch.float32) 
        
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight
        
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()

        if d_ff is None:
            d_ff = 64 * math.ceil((8 * d_model / 3) / 64)

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = self.w1(x)
        return self.w2(w1_x * torch.sigmoid(w1_x) * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, got {d_k}")

        position = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        dim = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        angles = position[:, None] / (theta ** (dim[None, :] / d_k))

        # Fixed RoPE lookup tables via register_buffer: move with the module device, but are not learned or saved in state_dict
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_pairs = rearrange(x, "... seq (d_pair two) -> ... seq d_pair two", two=2)

        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]

        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)

        rotated = torch.stack(
            (
                x_even * cos - x_odd * sin,
                x_even * sin + x_odd * cos,
            ),
            dim=-1
        )

        return rearrange(rotated, "... seq d_pair two -> ... seq (d_pair two)")

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / torch.sum(exp, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    attention_weights = softmax(scores, dim=-1)
    return attention_weights @ V

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads, got {d_model=} and {num_heads=}")

        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = None
        if theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        q = rearrange(self.q_proj(x), "... seq (heads d_head) -> ... heads seq d_head", heads=self.num_heads)
        k = rearrange(self.k_proj(x), "... seq (heads d_head) -> ... heads seq d_head", heads=self.num_heads)
        v = rearrange(self.v_proj(x), "... seq (heads d_head) -> ... heads seq d_head", heads=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2], device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        output = scaled_dot_product_attention(q, k, v, mask)
        output = rearrange(output, "... heads seq d_head -> ... seq (heads d_head)")
        return self.output_proj(output)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), token_positions)
        return x + self.ffn(self.ln2(x))

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(token_ids)
        token_positions = torch.arange(token_ids.shape[-1], device=token_ids.device)

        for layer in self.layers:
            x = layer(x, token_positions)

        return self.lm_head(self.ln_final(x))
