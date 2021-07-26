from torch import nn
from torch.utils.data import DataLoader
from pytorchltr.evaluation.dcg import ndcg
from pytorchltr.datasets.svmrank.svmrank import SVMRankDataset

from utils import convert_rel_to_gamma


def evaluate_test_performance(score_fn: nn.Module, test: SVMRankDataset) -> float:
    """与えられたmodelのランキング性能をテストデータにおける真の嗜好度合い情報(\gamma)を使ってnDCG@10で評価する."""
    loader = DataLoader(
        test, batch_size=1024, shuffle=False, collate_fn=test.collate_fn()
    )
    ndcg_score = 0.0
    for batch in loader:
        gamma = convert_rel_to_gamma(relevance=batch.relevance)
        ndcg_score += ndcg(
            score_fn(batch.features), gamma, batch.n, k=10, exp=False
        ).sum()
    return float(ndcg_score / len(test))
