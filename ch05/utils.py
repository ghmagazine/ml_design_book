from typing import Tuple

from torch import LongTensor, FloatTensor, bernoulli, arange


def convert_rel_to_mu(
    relevance: LongTensor,
    max_rel_value: int = 4,
    eps: float = 0.1,
) -> Tuple[FloatTensor, FloatTensor]:
    """元データの嗜好度合いラベルを[0,1]-スケールの\mu(u,i)に変換する."""
    mu = 1 - eps
    mu *= (2 ** relevance.float()) - 1
    mu /= (2 ** max_rel_value) - 1
    mu += eps
    conversion = bernoulli(mu)
    return mu, conversion  # \mu(u,i), R(u,i) | C(u,i,\cdot)=1


def convert_rel_to_mu_zero(
    relevance: LongTensor,
    max_rel_value: int = 4,
    eps: float = 0.1,
) -> Tuple[FloatTensor, FloatTensor]:
    """元データの嗜好度合いラベルを[0,1]-スケールの\mu^{(0)}(u,i)に変換する."""
    mu_zero = 1 - eps
    mu_zero *= (2 ** relevance.float()) - 1
    mu_zero /= (2 ** max_rel_value) - 1
    mu_zero += eps
    mu_zero += 0.1 * (relevance == 3).float()
    mu_zero += 0.05 * (relevance == 2).float()
    mu_zero -= 0.1 * (relevance == 1).float()
    mu_zero -= 0.05 * (relevance == 0).float()
    conversion_zero = bernoulli(mu_zero)
    return mu_zero, conversion_zero  # \mu^{(0)}(u,i), R(u,i) | C(u,i,\cdot)=0


def generate_click_and_recommend(
    relevance: LongTensor,
) -> Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor]:
    """推薦枠内でのクリック発生有無・推薦確率・推薦有無情報を適当に生成する（他の生成方法も十分あり得る）."""
    num_items = relevance.shape[1]
    pscore = 0.9 / arange(1, num_items + 1)
    recommend = bernoulli(pscore)
    mu = convert_rel_to_mu(relevance)[0]
    click = bernoulli(mu) * recommend  # \mu(u,i)が大きいとクリックが発生しやすいとする
    pscore_zero = 1.0 - pscore
    pscore = mu * pscore  # 推薦枠内でクリックが発生する確率x推薦される確率
    return click, pscore, recommend, pscore_zero
