from typing import Optional

from torch import ones, FloatTensor
from torch.nn.functional import log_softmax


def listwise_loss(
    scores: FloatTensor,  # f_{\phi}(u,i)
    click: FloatTensor,  # CTR(u,i,k)
    num_docs: FloatTensor,
    pscore: Optional[FloatTensor] = None,  # \theta(k)
) -> FloatTensor:
    """リストワイズ損失.

    パラメータ
    ----------
    scores: FloatTensor
        スコアリング関数の出力.

    click: FloatTensor
        クリック有無データ. Implicit Feedback.

    num_docs: FloatTensor
        クエリごとのドキュメントの数.

    pscore: Optional[FloatTensor], default=None.
        傾向スコア. Noneの場合は、ナイーブ推定量に基づいた損失が計算される.

    """
    if pscore is None:
        pscore = ones(click.shape[1])
    listwise_loss = 0
    for scores_, click_, num_docs_ in zip(scores, click, num_docs):
        listwise_loss_ = (click_ / pscore) * log_softmax(scores_, dim=0)
        listwise_loss -= listwise_loss_[:num_docs_].sum()
    return listwise_loss / len(scores)
