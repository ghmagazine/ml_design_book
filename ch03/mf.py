from typing import Tuple, Optional, List
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_squared_error as calc_mse
from sklearn.utils import check_random_state
from tqdm import tqdm


@dataclass
class MatrixFactorization:
    """MatrixFactorization.

    パラメータ
    ----------
    k: int
        ユーザ・アイテムベクトルの次元数.

    learning_rate: float
        学習率.

    reg_param: float
        正則化項のハイパーパラメータ.

    random_state: int
        モデルパラメータの初期化を司る乱数.

    """

    k: int
    learning_rate: float
    reg_param: float
    alpha: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    random_state: int = 12345

    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)

    def fit(
        self,
        train: np.ndarray,
        val: np.ndarray,
        test: np.ndarray,
        pscore: Optional[np.ndarray] = None,  # 傾向スコア (Propensity Score; pscore)
        n_epochs: int = 10,
    ) -> Tuple[List[float], List[float]]:
        """トレーニングデータを用いてモデルパラメータを学習し、バリデーションとテストデータに対する予測誤差の推移を出力.

        パラメータ
        ----------
        train: array-like of shape (データ数, 3)
            トレーニングデータ. (ユーザインデックス, アイテムインデックス, 嗜好度合いデータ)が3つのカラムに格納された2次元numpy配列.

        val: array-like of shape (データ数, 3)
            バリデーションデータ. (ユーザインデックス, アイテムインデックス, 嗜好度合いデータ)が3つのカラムに格納された2次元numpy配列.

        test: array-like of shape (データ数, 3)
            テストデータ. (ユーザインデックス, アイテムインデックス, 嗜好度合いデータ)が3つのカラムに格納された2次元numpy配列.

        pscore: array-like of shape (ユニークな嗜好度合い数,), default=None.
            事前に推定された嗜好度合いごとの観測されやすさ, 傾向スコア. P(O=1|R=r).
            Noneが与えられた場合, ナイーブ推定量が用いられる.

        n_epochs: int, default=10.
            学習におけるエポック数.

        """

        # 傾向スコアが設定されない場合は、ナイーブ推定量を用いる
        if pscore is None:
            pscore = np.ones(np.unique(train[:, 2]).shape[0])

        # ユニークユーザとユニークアイテムの数を数える
        n_users = np.unique(train[:, 0]).shape[0]
        n_items = np.unique(train[:, 1]).shape[0]

        # モデルパラメータを初期化
        self._initialize_model_parameters(n_users=n_users, n_items=n_items)

        # トレーニングデータを用いてモデルパラメータを学習
        val_loss, test_loss = [], []
        for _ in tqdm(range(n_epochs)):
            self.random_.shuffle(train)
            for user, item, rating in train:
                # 傾向スコアの逆数で予測誤差を重み付けて計算
                err = rating - self._predict_pair(user, item)
                err /= pscore[rating - 1]
                grad_P = err * self.Q[item] - self.reg_param * self.P[user]
                self._update_P(user=user, grad=grad_P)
                grad_Q = err * self.P[user] - self.reg_param * self.Q[item]
                self._update_Q(item=item, grad=grad_Q)

            # バリデーションデータに対する嗜好度合いの予測誤差を計算
            # 傾向スコアが与えられた場合はそれを用いたIPS推定量で、そうでない場合はナイーブ推定量を用いる
            r_hat_val = self.predict(data=val)
            inv_pscore_val = 1.0 / pscore[val[:, 2] - 1]  # 傾向スコアの逆数
            val_loss.append(
                calc_mse(val[:, 2], r_hat_val, sample_weight=inv_pscore_val)
            )
            # テストデータにおける嗜好度合いの予測誤差を計算
            r_hat_test = self.predict(data=test)
            test_loss.append(calc_mse(test[:, 2], r_hat_test))

        return val_loss, test_loss

    def _initialize_model_parameters(self, n_users: int, n_items: int) -> None:
        """モデルパラメータを初期化."""
        self.P = self.random_.rand(n_users, self.k) / self.k
        self.Q = self.random_.rand(n_items, self.k) / self.k
        self.M_P = np.zeros_like(self.P)
        self.M_Q = np.zeros_like(self.Q)
        self.V_P = np.zeros_like(self.P)
        self.V_Q = np.zeros_like(self.Q)

    def _update_P(self, user: int, grad: np.ndarray) -> None:
        "与えられたユーザのベクトルp_uを与えられた勾配に基づき更新."
        self.M_P[user] = self.beta1 * self.M_P[user] + (1 - self.beta1) * grad
        self.V_P[user] = self.beta2 * self.V_P[user] + (1 - self.beta2) * (grad ** 2)
        M_P_hat = self.M_P[user] / (1 - self.beta1)
        V_P_hat = self.V_P[user] / (1 - self.beta2)
        self.P[user] += self.alpha * M_P_hat / ((V_P_hat ** 0.5) + self.eps)

    def _update_Q(self, item: int, grad: np.ndarray) -> None:
        "与えられたアイテムのベクトルq_iを与えられた勾配に基づき更新."
        self.M_Q[item] = self.beta1 * self.M_Q[item] + (1 - self.beta1) * grad
        self.V_Q[item] = self.beta2 * self.V_Q[item] + (1 - self.beta2) * (grad ** 2)
        M_Q_hat = self.M_Q[item] / (1 - self.beta1)
        V_Q_hat = self.V_Q[item] / (1 - self.beta2)
        self.Q[item] += self.alpha * M_Q_hat / ((V_Q_hat ** 0.5) + self.eps)

    def _predict_pair(self, user: int, item: int) -> float:
        """与えられたユーザ・アイテムペア(u,i)の嗜好度合いを予測する."""
        return self.P[user] @ self.Q[item]

    def predict(self, data: np.ndarray) -> np.ndarray:
        """与えられたデータセットに含まれる全ユーザ・アイテムペアの嗜好度合いを予測する."""
        r_hat_arr = np.empty(data.shape[0])
        for i, row in enumerate(data):
            r_hat_arr[i] = self._predict_pair(user=row[0], item=row[1])
        return r_hat_arr
