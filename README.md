## 施策デザインのための機械学習入門

技術評論社発行の書籍『施策デザインのための機械学習入門』（[Amazon](https://www.amazon.co.jp/dp/4297122243/)）のサンプルコードです。

<img src="http://image.gihyo.co.jp/assets/images/cover/2021/9784297122249.jpg" width="200">

## 書籍情報

- 紙版発売: 2021年8月4日 / 電子版発売: 2021年7月30日
- 齋藤優太，安井翔太　著，株式会社ホクソエム　監修
- A5判／336ページ
- 定価3,278円（本体2,980円＋税10%）
- ISBN 978-4-297-12224-9
- 出版社サポートサイト: https://gihyo.jp/book/2021/978-4-297-12224-9

## ディレクトリ構成

|ディレクトリ| 内容 |
|:----|:-------|
| [ch02](ch02/) |「2.3節 Open Bandit Pipelineを用いた実装」で用いた実装 |
| [ch03](ch03/) |「3.5節 Pythonによる実装とYahoo! R3データを用いた性能検証」で用いた実装 |
| [ch04](ch04/) |「4.3節 PyTorchを用いた実装と簡易実験」で用いた実装 |
| [ch05](ch05/) |「5.4節 PyTorchを用いた実装と簡易実験」で用いた実装 |


## 動作環境
本書で用いたPython環境は[poetry](https://python-poetry.org/docs/)を用いて構築しています。リポジトリを`git clone`し、フォルダ直下で`poetry install`を実行すると、本書と同じ環境を構築できます。

```bash
# リポジトリをclone
git clone https://github.com/ghmagazine/ml_design_book.git
cd ml_design_book

# poetryで環境構築
poetry install

# jupyter labを立ち上げ
poetry run jupyter lab
```

Pythonおよび利用パッケージのバージョンは以下の通りです。

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
