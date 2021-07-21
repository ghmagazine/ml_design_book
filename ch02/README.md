## 第2章

### データセット

実データを用いた簡易実験を行うには、Open Bandit Datasetを[https://research.zozo.com/data.html](https://research.zozo.com/data.html)から取得し、ディレクトリを以下の通りに配置します。

```
ch2/
 ├──open_bandit_dataset/
```
なお本書でも補足した通りオリジナルのデータセットは11GBあるため、最初はお試しのスモールサイズデータを使ってみると良いかもしれません。お試しのデータセットの使い方も本書にて補足しています。
### Open Bandit Pipelineを用いた実装

- [`synthetic-data.ipynb`](./synthetic-data.ipynb): 人工データを用いて意思決定モデルの学習とその性能評価を行う流れを実装.
- [`real-data.ipynb`](./real-data.ipynb): 実データ（Open Bandit Dataset）を用いて意思決定モデルの学習とその性能評価を行う流れを実装.
