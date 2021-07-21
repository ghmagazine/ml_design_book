from typing import List, Optional

from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorchltr.datasets.svmrank.svmrank import SVMRankDataset

from evaluate import evaluate_test_performance
from loss import listwise_loss
from utils import convert_rel_to_gamma, convert_gamma_to_implicit


def train_ranker(
    score_fn: nn.Module,
    optimizer: optim,
    estimator: str,
    train: SVMRankDataset,
    test: SVMRankDataset,
    batch_size: int = 32,
    n_epochs: int = 30,
    pow_true: float = 1.0,
    pow_used: Optional[float] = None,
) -> List:
    """ランキングモデルを学習するための関数.

    パラメータ
    ----------
    score_fn: nn.Module
        スコアリング関数.

    optimizer: optim
        パラメータ最適化アルゴリズム.

    estimator: str
        スコアリング関数を学習するための目的関数を観測データから近似する推定量.
        'naive', 'ips', 'ideal'のいずれかしか与えることができない.
        'ideal'が与えられた場合は、真の嗜好度合いデータ（Explicit Feedback）をもとに、ランキングモデルを学習する.

    train: SVMRankDataset
        （オリジナルの）トレーニングデータ.

    test: SVMRankDataset
        （オリジナルの）テストデータ.

    batch_size: int, default=32
        バッチサイズ.

    n_epochs: int, default=30
        エポック数.

    pow_true: float, default=1.0
        ポジションバイアスの大きさを決定するパラメータ. クリックデータの生成に用いられる.
        pow_trueが大きいほど、ポジションバイアスの影響（真の嗜好度合いとクリックデータの乖離）が大きくなる.

    pow_used: Optional[float], default=None
        ポジションバイアスの大きさを決定するパラメータ. ランキングモデルの学習に用いられる.
        Noneが与えられた場合は、pow_trueと同じ値が設定される.
        pow_trueと違う値を与えると、ポジションバイアスの大きさを見誤ったケースにおけるランキングモデルの学習を再現できる.

    """
    assert estimator in [
        "naive",
        "ips",
        "ideal",
    ], f"estimator must be 'naive', 'ips', or 'ideal', but {estimator} is given"
    if pow_used is None:
        pow_used = pow_true

    ndcg_score_list = list()
    for _ in tqdm(range(n_epochs)):
        loader = DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train.collate_fn(),
        )
        score_fn.train()
        for batch in loader:
            if estimator == "naive":
                click, theta = convert_gamma_to_implicit(
                    relevance=batch.relevance, pow_true=pow_true, pow_used=pow_used
                )
                loss = listwise_loss(
                    scores=score_fn(batch.features), click=click, num_docs=batch.n
                )
            elif estimator == "ips":
                click, theta = convert_gamma_to_implicit(
                    relevance=batch.relevance, pow_true=pow_true, pow_used=pow_used
                )
                loss = listwise_loss(
                    scores=score_fn(batch.features),
                    click=click,
                    num_docs=batch.n,
                    pscore=theta,
                )
            elif estimator == "ideal":
                gamma = convert_rel_to_gamma(relevance=batch.relevance)
                loss = listwise_loss(
                    scores=score_fn(batch.features), click=gamma, num_docs=batch.n
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        score_fn.eval()
        ndcg_score = evaluate_test_performance(score_fn=score_fn, test=test)
        ndcg_score_list.append(ndcg_score)

    return ndcg_score_list
