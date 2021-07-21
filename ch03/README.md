## 第3章

### Pythonによる実装
- [`mf.py`](./mf.py): IPS推定量に対応できるMatrix Factorizationを実装.

### 簡易実験
- [`naive-vs-ips.ipynb`](./naive-vs-ips.ipynb): 嗜好度合いデータの観測構造にバイアスが存在する状況で、ナイーブ推定量とIPS推定量の挙動を検証.

`naive-vs-ips.ipynb`では、以下の通り`data/`ディレクトリにデータファイルが配置されていることを想定。

```
ch3/
 ├──data/
     ├──test.txt
     ├──train.txt
```
