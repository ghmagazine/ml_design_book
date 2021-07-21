from typing import Optional

from torch import ones_like, FloatTensor
from torch.nn.functional import log_softmax


def listwise_loss(
    scores: FloatTensor,  # f_{\phi}
    click: FloatTensor,  # C(u,i,k)
    conversion: FloatTensor,  # R(u,i)
    num_docs: FloatTensor,
    recommend: Optional[FloatTensor] = None,  # \mathbb{I}\{ K(u,i) \neq 0 \}
    pscore: Optional[FloatTensor] = None,  # CTR(u,i)
    pscore_zero: Optional[FloatTensor] = None,  # e(u,i,0)
) -> FloatTensor:
    """リストワイズ損失.

    パラメータ
    ----------
    scores: FloatTensor
        スコアリング関数の出力. f_{\phi}.

    click: FloatTensor
        クリック発生有無データ. C(u,i,k).

    conversion: FloatTensor
        コンバージョン発生有無データ（クリック発生あとに観測される目的変数）. R(u,i).

    num_docs: FloatTensor
        クエリごとのドキュメントの数.

    recommend: FloatTensor, default=None.
        推薦有無を表すインディケータ. \mathbb{I}\{ K(u,i) \neq 0 \}

    pscore: Optional[FloatTensor], default=None.
        傾向スコア. CTR(u,i).
        Noneが与えられた場合はナイーブ推定量に基づいた損失が計算される.

    pscore_zero: Optional[FloatTensor], default=None.
        アイテムがユーザに推薦されない確率. e(u,i,0).
        Noneが与えられた場合はナイーブ推定量に基づいた損失が計算される.

    """
    if recommend is None:
        recommend = ones_like(click)
    if pscore is None:
        pscore = ones_like(click)
    if pscore_zero is None:
        pscore_zero = ones_like(click)
    listwise_loss = 0
    for scores_, click_, conv_, num_docs_, recommend_, pscore_, pscore_zero_ in zip(
        scores, click, conversion, num_docs, recommend, pscore, pscore_zero
    ):
        weight = ((click_ / pscore_) - ((1 - recommend_) / pscore_zero_)) * conv_
        listwise_loss -= (weight * log_softmax(scores_, dim=-1))[:num_docs_].sum()
    return listwise_loss / len(scores)
