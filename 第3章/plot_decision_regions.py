# -*- coding: utf-8 -*-
"""
plot_decision_regions.py
@author Ito Hajime
"""
from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_decision_regions(training_dataset_matrix: "np.ndarray[[float]]",
                          class_label_vector: "np.ndarray[float]",
                          classifier: object,
                          resolution: float = 0.02,
                          test_idx: "np.ndarray[int]" = None) -> None:
    """決定領域のプロット関数

    Args:
        training_dataset_matrix (np.ndarray[[float]]): [shape = [[サンプルの数, 特徴ベクトルの要素数]]]
        class_label_vector (np.ndarray[float]): [shape = [特徴ベクトルの要素数]]
        classifier (object): [分類機オブジェクト]
        resolution ([type], optional): [格子分解能(格子線の幅)]. Defaults to 0.02:float.
    """

    # マーカーとカラーマップの準備
    markers: tuple = ('s', 'o', '^', 'v')
    colors: tuple = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(class_label_vector))])

    # 決定領域のプロット
    x1_min, x1_max = training_dataset_matrix[:, 0].min() - 1, training_dataset_matrix[:, 0].max() + 1
    x2_min, x2_max = training_dataset_matrix[:, 1].min() - 1, training_dataset_matrix[:, 1].max() + 1
    # グリッドポイントの生成
    # arangeで[x_min, x_min + resolution, x_min + resolution * 2, ....., x.max]というベクトルを生成
    # meshgridでそれらを, n×n行列にする
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    # 各特徴量を1次元配列に変換して予測を実行
    # [[x1, x2, x3, ...], [y1, y2, y3, ...]]t　で　n行2列の行列にする
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # 予測結果を元のグリッドポイントのデータサイズに変換
    Z = Z.reshape(xx1.shape)
    # グリッドポイントの等高線のプロット
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    # 軸の範囲の設定
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # クラスごとにサンプルをプロット
    for idx, cl in enumerate(np.unique(class_label_vector)):
        plt.scatter(x=training_dataset_matrix[class_label_vector == cl, 0],
                    y=training_dataset_matrix[class_label_vector == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    label=cl,
                    edgecolor='black')

    # テストサンプルを目立たせる(点を〇で表示)
    if test_idx:
        # 全てのサンプルをプロット
        tdm_test, clv_test = training_dataset_matrix[test_idx, :], class_label_vector[test_idx]
        plt.scatter(tdm_test[:, 0], tdm_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')