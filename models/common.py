# common.py
import math
import torch


def trunc_normal_init_(tensor: torch.Tensor,
                       std: float = 0.02,
                       a: float = -2.0,
                       b: float = 2.0) -> torch.Tensor:

    with torch.no_grad():
        if std == 0.0:
            tensor.zero_()
            return tensor

        sqrt2 = math.sqrt(2.0)
        alpha = math.erf(a / sqrt2)
        beta  = math.erf(b / sqrt2)
        z = (beta - alpha) / 2.0

        c = 1.0 / math.sqrt(2.0 * math.pi)
        pdf_a = c * math.exp(-0.5 * a * a)
        pdf_b = c * math.exp(-0.5 * b * b)
        comp_std = std / math.sqrt(
            1.0 - (b * pdf_b - a * pdf_a) / z - ((pdf_b - pdf_a) / z) ** 2
        )

        tensor.uniform_(alpha, beta)
        tensor.erfinv_()
        tensor.mul_(sqrt2 * comp_std)
        tensor.clamp_(a * comp_std, b * comp_std)

    return tensor