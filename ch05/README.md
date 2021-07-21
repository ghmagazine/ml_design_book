## 第5章

### PyTorchを用いた実装
- [`evaluate.py`](./evaluate.py): テストデータにおけるnDCG@10を計算するための関数を実装.
- [`loss.py`](./loss.py): IPS推定量に基づくリストワイズ損失関数を実装.
- [`model.py`](./model.py): 多層パーセプトロンに基づくスコアリング関数を実装.
- [`utils.py`](./utils.py): 半人工データを生成するための関数を実装.


### 半人工データを用いた簡易実験
- [`naive-vs-ips.ipynb`](./naive-vs-ips.ipynb): 推薦枠内で定義されるKPIを扱う状況においてナイーブ推定量とIPS推定量の性能差を検証.
- [`objective-misspecification.ipynb`](./objective-misspecification.ipynb): ランキングシステム構築のための方針の誤設定が性能に与える影響を検証.
