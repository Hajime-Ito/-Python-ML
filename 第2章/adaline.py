# -*- coding: utf-8 -*-
"""
adaline.py
@author Ito Hajime
"""
import numpy as np


class AdalineGD():
    """Adaptive Linear Neuron 分類機"""

    def __init__(self, learning_rate: float = 0.01, epoch: int = 50, random_state: int = 1) -> None:
        """イニシャライザ

        Args:
            learning_rate (float, optional): [学習率]. Defaults to 0.01.
            epoch (int, optional): [学習回数]. Defaults to 50.
            random_state (int, optional): [ランダムシード値]. Defaults to 1.
        """
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.random_state = random_state

    def fit(self, training_data_matrix: "np.ndarray[[float]]", class_label_vector: "np.ndarray[float]") -> object:
        """トレーニングデータに適合させる。(重み学習)

        Args:
            training_data_matrix (np.ndarray[[float]]): [トレーニングデータ行列, shape = [サンプルの数, 特徴ベクトルの要素数]]
            class_label_vector (np.ndarray[float]): [正解クラスラベルベクトル, shape = [サンプルの数]]

        Returns:
            object: [自分自身]
        """
        # ランダム関数の生成オブジェクト
        random_generator = np.random.RandomState(self.random_state)
        # 平均値0, 標準偏差0.01の正規分布で[特徴ベクトルの数+1]の大きさの乱数生成
        self.weight_vector = random_generator.normal(loc=0.0, scale=0.01, size=1 + training_data_matrix.shape[1])
        # 重み更新ごとのコストを記録
        self.cost_list = []

        for _ in range(self.epoch):  # トレーニング回数分トレーニングデータを反復
            net_input = self.__net_input(training_data_matrix)  # 総入力の計算
            # activationメソッドは単なる恒等関数なので、今回特に意味はない
            output = self.__activation(net_input)
            # 誤差(真のクラスラベル　- 線形活性化関数)の計算
            errors = (class_label_vector - output)
            # 重み更新(trainingdataの大きさとweight_vector[1:]の大きさは同じ)
            self.weight_vector[1:] += self.learning_rate * training_data_matrix.T.dot(errors)
            # バイアスユニットの更新
            self.weight_vector[0] += self.learning_rate * errors.sum()
            # コスト関数の計算
            cost = (errors ** 2).sum() / 2.0
            # コストの格納
            self.cost_list.append(cost)

        return self

     def __net_input(self, training_data_matrix: "np.ndarray[[float]]") -> "np.ndarray[float]":
        """総入力(特徴ベクトル * 重みベクトル)を計算
        プライベートメソッド

        Args:
            training_data_matrix ([np.ndarray]): [shape = [サンプルの数, 特徴ベクトルの要素数]]

        Returns:
            np.ndarray[float]: [shape = [特徴ベクトルの要素数]] 
        """
        # バイアスユニットは特徴ベクトルと積をとらない
        return np.dot(training_data_matrix, self.weight_vector[1:]) + self.weight_vector[0]

    def __activation(self, net_input: "np.ndarray[float]") -> "np.ndarray[float]":
        """線形活性化関数の出力を計算
        プライベートメソッド
        Args:
            net_input (np.ndarray[float]): [総入力, shape = [特徴ベクトルの要素数]]

        Returns:
            np.ndarray[float]: [shape = [特徴ベクトルの要素数]]
        """
        return training_data_matrix

    def predict(self, training_data_matrix: "np.ndarray[[float]]") -> "np.ndarray[int]":
        """予測クラスラベルを返す

        Args:
            training_data_matrix (np.ndarray[[float]]): [shape = [サンプルの数, 特徴ベクトルの要素数]]

        Returns:
            np.ndarray[int]: [shape = [特徴ベクトルの要素数]]
        """
        return np.where(self.__net_input(training_data_matrix) > 0.0, 1, -1)
