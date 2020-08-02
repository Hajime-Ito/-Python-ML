# -*- coding: utf-8 -*-
"""
plot_iris_dataset.py
@author Ito Hajime
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import perceptron as pt

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
# 品種setosaのプロット(赤のO)
plt.scatter(training_data_matrix[:50, 0],
            training_data_matrix[:50, 1],
            color="red",
            marker='o',
            label='setosa')
# 品種versicolorのプロット(青のX)
plt.scatter(training_data_matrix[50:100, 0],
            training_data_matrix[50:100, 1],
            color="blue",
            marker='x',
            label='versicolor')
# 軸のラベル設定
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
# 凡例の指定(左上に配置)
plt.legend(loc='upper left')
# 図の表示
plt.show()

# パーセプトロンのオブジェクトの生成(インスタンス化)
ppn = pt.Perceptron(learning_rate=0.1, epoch=10, random_state=1)
# トレーニングデータへのモデル適合
ppn.fit(training_data_matrix, class_label_vector)
# エポックと誤分類誤差の関係の折れ線グラフをプロット
plt.plot(range(1, len(ppn.error_list) + 1), ppn.error_list, marker='o')
# 軸のラベル設定
plt.xlabel('Epochs')
plt.ylabel('Number of update')
# 図の表示
plt.show()
