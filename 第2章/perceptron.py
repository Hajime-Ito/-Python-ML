# -*- coding: utf-8 -*-
"""
perceptron.py
@author Ito Hajime
"""
import numpy as np


class Perceptron(object):
    """ パーセプトロンの分類機 """

    def __init__(self, learning_rate: float, epoch: int, random_state: int) -> None:
        """イニシャライザ

        Args:
            learning_rate ([float]): [学習率]
            epoch ([int]): [トレーニング回数]
            random_state ([int]): [ランダムシード値]
        """
        self.learning_rate: float = learning_rate
        self.epoch: int = epoch
        self.random_state: int = random_state
        self.weight_vector: "np.ndarray[float]" = None
        self.error_list: list[int] = None

    def fit(self, training_data_matrix: "np.ndarray[[float]]", class_label_vector: "np.ndarray[float]") -> object:
        """トレーニングデータに適合(重み決定)

        例)
        トレーニングデータ行列
        (サンプルの特徴ベクトル3つの線形結合)
        [
        [1,1,1,1]
        [1,1,1,0]
        [0,0,0,1]
        ]

        例)
        重みベクトル
        (特徴ベクトルの要素数 + バイアスユニット1つ分 = (4 + 1 = 5))
        [1,0,0,1,1]

        例)
        更新用重みベクトル(Δw = 学習率(クラスラベル - 予測クラスラベル) * 入力(特徴ベクトル)) | 数式

        今回はプログラムで表現しやすいように以下のようにしている。

        更新重み値 Δw = 学習率(クラスラベル - 予測クラスラベル)
        w[1:] = w + Δw * 入力(特徴ベクトル)
        w[0]  = w + Δw

        Args:
            training_data_matrix (np.ndarray): [shape = [サンプルの数, 特徴ベクトルの要素数]]
            class_label_vector (np.ndarray): [shape = [サンプルの数]]

        Returns:
            self
        """
        # ランダム関数の生成オブジェクト
        random_generator = np.random.RandomState(self.random_state)

        # 平均値0, 標準偏差0.01の正規分布で[特徴ベクトル+1]の大きさの乱数生成
        self.weight_vector = random_generator.normal(
            loc=0.0,
            scale=0.01,
            size=1 + training_data_matrix.shape[1])

        self.error_list = []

        # トレーニング回数分トレーニングデータを反復する
        for _ in range(self.epoch):
            errors = 0
            # データセットの各行(サンプル)について重みの更新
            for training_data_vector, class_label in zip(training_data_matrix, class_label_vector):
                # Δwを計算(正のラベルが負と識別された場合 2 or 負のラベルが正と識別された場合 -2 or 正しく識別された場合 0)
                update_weight_value = self.learning_rate * (class_label - self.__predict(training_data_vector))
                # 重みの更新 Δw(定数) * training_data_vector(ベクトル)
                self.weight_vector[1:] += update_weight_value * training_data_vector
                # バイアスユニットの更新
                self.weight_vector[0] += update_weight_value
                # 重みΔwが正しく更新されていない(Δw != 0)なら誤分類としてカウントする
                # Trueは1, Falseは0として扱うので、update_weight_value != 0.0が真なら1となる
                errors += int(update_weight_value != 0.0)

            # 反復回数ごとの誤差を格納する
            self.error_list.append(errors)

        return self

    def __net_input(self, training_data_matrix: "np.ndarray[[float]]") -> "np.ndarray[float]":
        """総入力(特徴ベクトル * 重みベクトル)を計算
        プライベートメソッド

        Args:
            training_data_matrix ([np.ndarray]): [shape = [サンプルの数, 特徴ベクトルの要素数]]

        Returns:
            np.ndarray[float]: [shape = [1], intで返さないのは後々ndarray同士で演算するため] 
        """
        # バイアスユニットは特徴ベクトルと積をとらない
        return np.dot(training_data_matrix, self.weight_vector[1:]) + self.weight_vector[0]

    def __predict(self, training_data_matrix: "np.ndarray[[float]]") -> "np.ndarray[int]":
        """予測クラスラベルを返す
        プライベートメソッド

        Args:
            training_data_matrix (np.ndarray[[float]]): [shape = [サンプルの数, 特徴ベクトルの要素数]]

        Returns:
            np.ndarray[int]: [shape = [1], intで返さないのは後々ndarray同士で演算するため]
        """
        return np.where(self.__net_input(training_data_matrix) > 0.0, 1, -1)
