from typing import Tuple

from torch import LongTensor, FloatTensor, arange, bernoulli


def convert_rel_to_gamma(
    relevance: LongTensor, max_rel_value: int = 4, eps: float = 0.1
) -> FloatTensor:
    """元データの嗜好度合いラベルを[0,1]-スケールに変換する."""
    gamma = 1.0 - eps
    gamma *= (2 ** relevance.float()) - 1
    gamma /= (2 ** max_rel_value) - 1
    gamma += eps
    return gamma


def convert_gamma_to_implicit(
    relevance: LongTensor,
    pow_true: float = 1.0,
    pow_used: float = 1.0,
) -> Tuple[FloatTensor, FloatTensor]:
    """[0,1]-スケールの嗜好度合いをPosition-based Modelをもとにクリックデータに変換する."""
    gamma = convert_rel_to_gamma(relevance=relevance)
    theta_true = (0.9 / arange(1, gamma.shape[1] + 1)) ** pow_true
    theta_used = (0.9 / arange(1, gamma.shape[1] + 1)) ** pow_used
    click = bernoulli(gamma * theta_true)
    return click, theta_used
