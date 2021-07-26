from torch import nn
from torch.utils.data import DataLoader
from pytorchltr.evaluation.dcg import ndcg
from pytorchltr.datasets.svmrank.svmrank import SVMRankDataset

from utils import (
    convert_rel_to_mu,
    convert_rel_to_mu_zero,
)


def evaluate_test_performance(
    score_fn: nn.Module, test: SVMRankDataset, objective: str
) -> float:
    """与えられたスコアリング関数のランキング性能をテストデータにおける目的変数の期待値を使ってnDCG@10で評価する."""
    loader = DataLoader(
        test, batch_size=1024, shuffle=False, collate_fn=test.collate_fn()
    )
    ndcg_score = 0.0
    for batch in loader:
        mu = convert_rel_to_mu(batch.relevance)[0]
        mu_zero = convert_rel_to_mu_zero(batch.relevance)[0]
        outcome = mu if objective == "via-rec" else (mu - mu_zero)
        ndcg_score += ndcg(
            score_fn(batch.features), outcome, batch.n, k=10, exp=False
        ).sum()
    return float(ndcg_score / len(test))
