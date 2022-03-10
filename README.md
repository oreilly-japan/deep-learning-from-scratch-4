[<img src="https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-4/images/deep-learning-from-scratch-4.png" width="200px">](https://www.amazon.co.jp/dp/4873119758)

書籍『[ゼロから作るDeep Learning ❹ 強化学習編](https://www.amazon.co.jp/dp/4873119758)』(オライリー・ジャパン)のサポートサイトです。本書籍で使用するソースコードがまとめられています。



## ファイル構成

|フォルダ名 |説明                         |
|:--        |:--                          |
|ch01       |1章で使用するソースコード    |
|ch02       |2章で使用するソースコード    |
|...        |...                          |
|ch09       |9章で使用するソースコード    |
|common     |共通で使用するソースコード   |
|pytorch     |PyTorchに移植したソースコード   |


ソースコードの解説は、本書籍をご覧ください。


## Pythonと外部ライブラリ
ソースコードを実行するには、下記のソフトウェアが必要です。

* Python 3.x（バージョン3系）
* NumPy
* Matplotlib
* OpenAI Gym
* DeZero （または PyTorch）


本書では、ディープラーニングのフレームワークとしてDeZeroを使います。DeZeroは「ゼロから作るDeep Learning」シリーズの3作目で作ったフレームワークです（ `pip install dezero` からインストールできます）。

PyTorchを使った実装は[pytorchフォルダ](https://github.com/oreilly-japan/deep-learning-from-scratch-4/tree/master/pytorch)にて提供しています。

## 実行方法
各章のフォルダに該当するコードがあります。
実行するためには、下記のとおりPythonコマンドを実行します（どのディレクトリからでも実行できます）。

```
$ python ch01/avg.py
$ python ch08/dqn.py

$ cd ch09
$ python actor_critic.py
```

## ライセンス

本リポジトリのソースコードは[MITライセンス](http://www.opensource.org/licenses/MIT)です。
商用・非商用問わず、自由にご利用ください。

## 正誤表

本書の正誤情報は以下のページで公開しています。

https://github.com/oreilly-japan/deep-learning-from-scratch-4/wiki/errata

本ページに掲載されていない誤植など間違いを見つけた方は、[japan@oreilly.co.jp](<mailto:japan@oreilly.co.jp>)までお知らせください。
