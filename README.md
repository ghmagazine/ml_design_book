## 施策デザインのための機械学習入門

技術評論社発行の書籍『施策デザインのための機械学習入門』（[Amazon](https://www.amazon.co.jp/dp/4297122243/)）のサンプルコードです。

<img src="http://image.gihyo.co.jp/assets/images/cover/2021/9784297122249.jpg" width="200">


### ディレクトリ構成

|ディレクトリ| 内容 |
|:----|:-------|
| [ch02](ch02/) |「2.3節 Open Bandit Pipelineを用いた実装」で用いた実装 |
| [ch03](ch03/) |「3.5節 Pythonによる実装とYahoo! R3データを用いた性能検証」で用いた実装 |
| [ch04](ch04/) |「4.3節 PyTorchを用いた実装と簡易実験」で用いた実装 |
| [ch05](ch05/) |「5.4節 PyTorchを用いた実装と簡易実験」で用いた実装 |


### 環境
本書で用いたPython環境は[poetry](https://python-poetry.org/docs/)を用いて構築しています。Pythonおよび利用パッケージのバージョンは以下の通りです。

```
[tool.poetry.dependencies]
python = "^3.9"
torch = "^1.9.0"
scikit-learn = "^0.24.2"
numpy = "^1.20.3"
matplotlib = "^3.4.2"
seaborn = "^0.11.1"
tqdm = "^4.61.1"
pytorchltr = "^0.2.1"
pandas = "^1.2.4"
obp = "^0.4.1"
jupyterlab = "^3.0.16"
```

これらのパッケージのバージョンが異なると、使用方法や挙動が本書執筆時点と異なる場合があるので、注意してください。
