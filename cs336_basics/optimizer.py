from collections.abc import Callable
from typing import Optional
import math

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data

                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta values: {betas}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0) + 1
                # compute the gradient of the loss
                grad = p.grad.data

                if "m" not in state:
                    # initialize first and second moment estimates.
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m = state["m"]
                v = state["v"]

                # decoupled weight decay.
                p.data -= lr * weight_decay * p.data
                # update first moment estimate.
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # update second moment estimate.
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # bias-corrected learning rate for iteration t.
                adjusted_lr = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                # moment-adjusted parameter update.
                p.data -= adjusted_lr * m / (torch.sqrt(v) + eps)
                state["t"] = t

        return loss
