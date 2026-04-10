from __future__ import print_function
import torch
from torch.optim import Optimizer
import math


class AdamNormGrad(Optimizer):
    """Adam with L2-normalized gradients.

    Normalizes each parameter's gradient by its L2 norm before applying
    standard Adam momentum and adaptive learning rate. This prevents
    individual large gradients from dominating the update, which
    stabilises training for VAEs with complex priors.

    Used in the VampPrior paper (Tomczak & Welling, 2018) and the
    Exemplar VAE paper (Aneja et al., 2021):

    * https://arxiv.org/abs/1705.07120 (VampPrior)
    * https://arxiv.org/abs/2004.04795 (Exemplar VAE)

    The gradient normalisation technique itself is discussed in:

    * https://arxiv.org/abs/1711.02257 (Brock et al., 2018)

    Note: the original implementation used deprecated PyTorch API calls
    (positional args for in-place ops). Updated to keyword-arg form:

    * ``tensor.add(scalar, other)``       → ``tensor.add(other, alpha=scalar)``
    * ``tensor.addcmul_(scalar, t1, t2)`` → ``tensor.addcmul_(t1, t2, value=scalar)``
    * ``tensor.addcdiv_(scalar, t1, t2)`` → ``tensor.addcdiv_(t1, t2, value=scalar)``
    * ``grad.new().resize_as_(grad).zero_()`` → ``torch.zeros_like(grad)``

    The positional forms were deprecated in PyTorch 1.5 and removed in 2.x.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamNormGrad, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                # normalize gradients by L2 norm
                grad = grad / (torch.norm(grad, 2) + 1e-7)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(grad)
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
