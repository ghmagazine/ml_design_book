## 第4章

### PyTorchを用いた実装
- [`evaluate.py`](./evaluate.py): テストデータにおけるnDCG@10を計算するための関数を実装.
- [`loss.py`](./loss.py): IPS推定量に基づくリストワイズ損失関数を実装.
- [`model.py`](./model.py): 多層パーセプトロンに基づくスコアリング関数を実装.
- [`utils.py`](./utils.py): ポジションバイアスが存在するクリックデータを生成するための関数を実装.


### 半人工データを用いた簡易実験
- [`naive-vs-ips.ipynb`](./naive-vs-ips.ipynb): ポジションバイアスが存在する状況においてナイーブ推定量とIPS推定量の性能差を検証.
- [`position-bias-effects.ipynb`](./position-bias-effects.ipynb): ポジションバイアスの大きさがランキング性能に与える影響を検証.
- [`theta-misspecification.ipynb`](./theta-misspecification.ipynb): ポジションバイアスの大きさを見誤ったときのランキング性能の変化を検証.
