# -*- coding: utf-8 -*-
"""
plot_adaline.py
@author Ito Hajime
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import adaline_sgd as ads
import adaline_gd as adg
import plot_decision_regions as pdr

# csvファイルの読み込み
dataframe = pd.read_csv('https://archive.ics.uci.edu/ml/'
                        'machine-learning-databases/iris/iris.data', header=None)
# 末尾5行を表示(読み込み確認のため)
print(dataframe.tail())
# 1-100行目の目的変数の抽出
class_label_vector = dataframe.iloc[0:100, 4].values
# Iris-Setosaを-1に、Iris-versicolorを1に変換
class_label_vector = np.where(class_label_vector == 'Iris-setosa', -1, 1)
# 1-100行目の1,3列目の抽出
training_data_matrix = dataframe.iloc[0:100, [0, 2]].values

# データのコピー
training_data_matrix_std = np.copy(training_data_matrix)
# 各列の標準化
training_data_matrix_std[:, 0] = (
    (training_data_matrix[:, 0] - training_data_matrix[:, 0].mean()) / training_data_matrix[:, 0].std())
training_data_matrix_std[:, 1] = (
    (training_data_matrix[:, 1] - training_data_matrix[:, 1].mean()) / training_data_matrix[:, 1].std())

# 確率的勾配降下法によるADALINEの学習
ada = ads.AdalineSGD(epoch=15, learning_rate=0.01, random_state=1)
# モデルへの適合
ada.fit(training_data_matrix_std, class_label_vector)
# 境界領域のプロット
pdr.plot_decision_regions(training_data_matrix_std, class_label_vector, classifier=ada)
# タイトルの設定
plt.title('Adaline - Stochastic Gradient Descent')
# 軸のラベル設定
plt.xlabel('sepal length')
plt.ylabel('petal length')
# 凡例の設定
plt.legend(loc='upper left')
plt.tight_layout()
# プロットの表示
plt.show()

# エポックとコストの折れ線グラフのプロット
plt.plot(range(1, len(ada.cost_list) + 1), ada.cost_list, marker='o')
# 軸ラベルの設定
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
# プロットの表示
plt.show()

# 勾配降下法によるADALINEの学習
ada = adg.AdalineGD(epoch=15, learning_rate=0.01)
# モデルへの適合
ada.fit(training_data_matrix_std, class_label_vector)
# エポックとコストの折れ線グラフのプロット
plt.plot(range(1, len(ada.cost_list) + 1), ada.cost_list, marker='o')
# 軸ラベルの設定
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
# プロットの表示
plt.show()
