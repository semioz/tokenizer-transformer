import torch
from cs336_basics.modules import softmax

def softmax_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        probs = torch.zeros_like(logits)
        probs[logits.argmax()] = 1.0
        return probs
    return softmax(logits / temperature, dim=-1) 


def top_p_filter(probs: torch.Tensor, p: float) -> torch.Tensor:
    if p >= 1.0:
        return probs

    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    sorted_remove = cumsum > p
    sorted_remove[1:] = sorted_remove[:-1].clone()
    sorted_remove[0] = False

    remove_mask = torch.zeros_like(probs, dtype=torch.bool)
    remove_mask[sorted_idx] = sorted_remove

    probs = probs.clone()
    probs[remove_mask] = 0.0
    probs /= probs.sum()
    return probs


def generate(
    model: torch.nn.Module,
    prompt: list[int],
    max_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    context_length: int | None = None,
) -> list[int]:
    model.eval()
    device = next(model.parameters()).device
    tokens = list(prompt)

    with torch.no_grad():
        for _ in range(max_tokens):
            x = tokens[-context_length:] if context_length is not None else tokens
            input_ids = torch.tensor([x], dtype=torch.long, device=device)

            logits = model(input_ids)  # (1, seq_len, vocab_size)
            next_logits = logits[0, -1]  # (vocab_size,)

            probs = softmax_temperature(next_logits, temperature)
            probs = top_p_filter(probs, top_p)

            next_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(next_token)

            if eos_token_id is not None and next_token == eos_token_id:
                break

    return tokens
