# -*- coding: utf-8 -*-
"""
skl_perceptron.py
@author Ito Hajime

sk-learnのperceptronを使ってみる
"""
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt

# Irisデータセットをロード
iris = datasets.load_iris()
# 3, 4列目の特徴量を抽出
X = iris.data[:, [2, 3]]
# クラスラベルを取得
y = iris.target
# 一意なクラスラベルを出力
print('Class labels:', np.unique(y))
# トレーニングデータとテストデータに分割
# 全体の30％をテストデータにする
# stratifyで層化サンプリングにする
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
# numpy.bincountのクラスラベルの比率の出現回数を数えてみる
# クラスラベル全体の出現回数
print('labels counts in y:', np.bincount(y))
# トレーニングデータのクラスラベルの出現回数
print('labels counts in y_train', np.bincount(y_train))
# テストデータのクラスラベルの出現回数
print('labels counts in y_test', np.bincount(y_test))
# 標準化のためのオブジェクト宣言
sc = StandardScaler()
# トレーニングデータの平均と標準偏差を計算
sc.fit(X_train)
# 平均と標準偏差をもって標準化
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
# エポック数40, 学習率0.1でパーセプトロンのインスタンスを生成
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
# トレーニングデータをモデルに適合させる
ppn.fit(X_train_std, y_train)
# テストデータで予測を実施
y_pred = ppn.predict(X_test_std)
# 誤分類のサンプルの個数を表示
print('Misclassified samples: %d' % (y_test != y_pred).sum())
# 分類の正解率を表示(下二桁)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
print('Accuracy: %.2f' % ppn.score(X_test_std, y_test))

# トレーニングデータセットとテストデータセットの特徴量を行方向に結合
X_combined_std = np.vstack((X_train_std, X_test_std))
# トレーニングデータとテストデータのクラスラベルを結合
y_combined = np.hstack((y_train, y_test))
# 決定境界のプロット
plot_decision_regions(training_dataset_matrix=X_combined_std, class_label_vector=y_combined,
                      classifier=ppn, test_idx=range(105, 150))
# 軸ラベルの設定
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
# 凡例の設定(左上に配置)
plt.legend(loc='upper left')
# グラフを表示
plt.tight_layout()
plt.show()
