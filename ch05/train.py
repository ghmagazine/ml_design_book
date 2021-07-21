from typing import List

from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorchltr.datasets.svmrank.svmrank import SVMRankDataset

from evaluate import evaluate_test_performance
from loss import listwise_loss
from utils import (
    convert_rel_to_mu,
    convert_rel_to_mu_zero,
    generate_click_and_recommend,
)


def train_ranker(
    score_fn: nn.Module,
    optimizer: optim,
    estimator: str,
    objective: str,
    train: SVMRankDataset,
    test: SVMRankDataset,
    batch_size: int = 32,
    n_epochs: int = 30,
) -> List:
    """ランキングモデルを学習するための関数.

    パラメータ
    ----------
    score_fn: nn.Module
        スコアリング関数.

    optimizer: optim
        パラメータ最適化アルゴリズム.

    estimator: str
        スコアリング関数を学習するための目的関数を近似する推定量.
        'naive', 'ips-via-rec', 'ips-platform'のいずれかしか与えることができない.

    objective: str
        推薦枠内経由('via_rec')のKPIを扱う場面か、プラットフォーム全体('platform')で定義されたKPI扱う場面かを指定.
        'via_rec', 'platform'のいずれかしか与えることができない.

    train: SVMRankDataset
        （オリジナルの）トレーニングデータ.

    test: SVMRankDataset
        （オリジナルの）テストデータ.

    batch_size: int, default=32
        バッチサイズ.

    n_epochs: int, default=30
        エポック数.

    """
    assert estimator in [
        "naive",
        "ips-via-rec",
        "ips-platform",
    ], f"estimator must be 'naive', 'ips-via-rec', 'ips-platform', but {estimator} is given"
    assert objective in [
        "via-rec",
        "platform",
    ], f"objective must be 'via-rec' or 'objective', but {objective} is given"

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
            conversion = convert_rel_to_mu(batch.relevance)[1]
            conversion_zero = convert_rel_to_mu_zero(batch.relevance)[1]
            click, pscore, recommend, pscore_zero = generate_click_and_recommend(
                batch.relevance
            )
            conversion_obs = conversion * click + conversion_zero * (1 - recommend)
            scores = score_fn(batch.features)
            if estimator == "naive":
                loss = listwise_loss(
                    scores=scores,
                    click=click,
                    conversion=conversion_obs,
                    num_docs=batch.n,
                )
            elif estimator == "ips-via-rec":
                loss = listwise_loss(
                    scores=scores,
                    click=click,
                    conversion=conversion_obs,
                    num_docs=batch.n,
                    recommend=None,
                    pscore=pscore,
                    pscore_zero=None,
                )
            elif estimator == "ips-platform":
                loss = listwise_loss(
                    scores=scores,
                    click=click,
                    conversion=conversion_obs,
                    num_docs=batch.n,
                    recommend=recommend,
                    pscore=pscore,
                    pscore_zero=pscore_zero,
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        score_fn.eval()
        ndcg_score = evaluate_test_performance(
            score_fn=score_fn, test=test, objective=objective
        )
        ndcg_score_list.append(ndcg_score)

    return ndcg_score_list
