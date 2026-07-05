import numpy as np
import torch

def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    offsets = np.arange(context_length)
    idx = starts[:, None] + offsets  # (batch_size, context_length)

    x = dataset[idx]
    y = dataset[idx + 1]

    return (
        torch.from_numpy(x).to(device=device, dtype=torch.long),
        torch.from_numpy(y).to(device=device, dtype=torch.long),
    )
