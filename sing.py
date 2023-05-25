from typing import Tuple
import torch
import torch.nn.functional as F
import collections


def centralize_gradient(x):
    """credit - https://github.com/Yonghongwei/Gradient-Centralization """

    size = x.dim()

    if size > 1:
        x.data.add_(-x.mean(dim=tuple(range(1, size)), keepdim=True))


def normalize_gradient(x, eps=1e-8):
    x.data.div_(x.norm() + eps)


class SING(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 lr: float = 5e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 weight_decay: float = 0,
                 eps: float = 1e-8,
                 softplus: bool = True,
                 beta_softplus: int = 50,
                 grad_central: bool = True,
                 grad_norm: bool = True,
                 lookahead_active: bool = True,
                 la_mergetime: int = 5,
                 la_alpha: float = 0.5
                 ):

        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay,
            grad_central=grad_central,
            grad_norm=grad_norm,
            softplus=softplus, beta_softplus=beta_softplus,
            lookahead_active=lookahead_active,
            la_mergetime=la_mergetime, la_alpha=la_alpha,
            la_step=0
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('la_step', 0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None and isinstance(closure, collections.Callable):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            lr = group["lr"]
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise NotImplementedError()

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.)

                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format)

                    if group["lookahead_active"]:
                        state["lookahead_params"] = torch.zeros_like(p)
                        state["lookahead_params"].copy_(p)

                # Gradient centralization
                if group["grad_central"]:
                    centralize_gradient(p.grad)

                # Gradient normalization
                if group["grad_norm"]:
                    normalize_gradient(p.grad, eps)

                state["step"] += 1

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Adam update
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    p.grad, p.grad, value=1 - beta2)

                bias_correction1 = 1 - torch.pow(beta1, state["step"])
                bias_correction2 = 1 - torch.pow(beta2, state["step"])

                # Weight decay (decoupled like AdamW)
                # Only apply weight decay to weights: https://arxiv.org/pdf/1812.01187.pdf
                if weight_decay and p.dim() > 1:
                    p.data.mul_(1 - lr * weight_decay)

                # Computing the denominator (Adam)
                denom = exp_avg_sq.sqrt() / bias_correction2.sqrt()

                # SAdam - https://arxiv.org/abs/1908.00700
                if group["softplus"]:
                    denom = F.softplus(denom, beta=group["beta_softplus"])
                else:
                    denom.add_(eps)

                # Update the parameter
                p.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)

        # LookAhead - https://arxiv.org/abs/1907.08610
        for group in self.param_groups:
            if not group['lookahead_active']:
                continue

            group['la_step'] += 1
            la_alpha = group['la_alpha']

            if group['la_step'] >= group['la_mergetime']:
                group['la_step'] = 0

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    state = self.state[p]

                    p.data.mul_(la_alpha).add_(
                        state["lookahead_params"], alpha=1 - la_alpha)
                    state["lookahead_params"].copy_(p)

        return loss
