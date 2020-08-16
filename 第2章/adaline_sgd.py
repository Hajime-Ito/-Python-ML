# -*- coding: utf-8 -*-
"""
adaline.py
@author Ito Hajime
"""
import numpy as np
from numpy.random import seed


class AdalineSGD():
    """Adaptive Linear Neuron 分類機"""

    def __init__(self, learning_rate: float = 0.01, epoch: int = 10, shuffle=True, random_state=None) -> None:
        """イニシャライザ

        Args:
            learning_rate (float, optional): [学習率]. Defaults to 0.01.
            epoch (int, optional): [学習回数]. Defaults to 50.
            random_state (int, optional): [ランダムシード値]. Defaults to 1.
        """
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.weight_vector_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, training_data_matrix: "np.ndarray[[float]]", class_label_vector: "np.ndarray[float]") -> object:
        """トレーニングデータに適合させる。(重み学習)

        Args:
            training_data_matrix (np.ndarray[[float]]): [トレーニングデータ行列, shape = [サンプルの数, 特徴ベクトルの要素数]]
            class_label_vector (np.ndarray[float]): [正解クラスラベルベクトル, shape = [サンプルの数]]

        Returns:
            object: [自分自身]
        """
        # 重みベクトルの生成
        self.__initialize_weights(training_data_matrix.shape[1])
        # コストを格納するリストの生成
        self.cost_list = []
        # トレーニングの回数分トレーニングデータを反復
        for _ in range(self.epoch):
            # 指定された場合はトレーニングデータをシャッフル
            if self.shuffle:
                training_data_matrix, class_label_vector = self.__shuffle(training_data_matrix, class_label_vector)
            # 各サンプルのコストを格納するリストの生成
            cost = []
            # 各サンプルに対する計算
            for training_data_vector, class_label in zip(training_data_matrix, class_label_vector):
                # 特徴量と目的変数を用いた重みの更新とコストの計算
                cost.append(self.__update_weights(training_data_vector, class_label))
            # サンプルの平均コストの計算
            avg_cost = sum(cost) / len(class_label_vector)
            # 平均コストを格納
            self.cost_list.append(avg_cost)

        return self

    def partial_fit(self, training_data_matrix: "np.ndarray[[float]]", class_label_vector: "np.ndarray[float]") -> object:
        """重みを再初期化することなくトレーニングデータに適合させる

        Args:
            training_data_matrix (np.ndarray[[float]]): [トレーニングデータ]
            class_label_vector (np.ndarray[float]): [正解クラスラベル]
        """
        # 初期化されていない場合は初期化を実行
        if not self.weight_vector_initialized:
            self.__initialize_weights(training_data_matrix.shape[1])
        # 目的変数(クラスラベルベクトル)の要素数が2以上の場合は
        # 各サンプルの特徴量と目的変数で重みを更新
        if class_label_vector.ravel().shape[0] > 1:
            for training_data_vector, class_label in zip(training_data_matrix, class_label_vector):
                self.__update_weights(training_data_vector, class_label)
            # 目的変数の要素数が1の場合は
            # サンプル全体の特徴量と目的変数で重みを更新
        else:
            self.__update_weights(training_data_matrix, class_label_vector)

        return self

    def __shuffle(self, training_data_matrix: "np.ndarray[[float]]", class_label_vector: "np.ndarray[float]") -> "np.ndarray[[float]], np.ndarray[float]":
        """トレーニングデータをシャッフル
        プライベートメソッド
        """
        r = self.rgen.permutation(len(class_label_vector))
        return training_data_matrix[r], class_label_vector[r]

    def __initialize_weights(self, m: int) -> None:
        """重みを小さな乱数に初期化

        Args:
            m (int): [特徴ベクトルの要素数]
        """
        self.rgen = np.random.RandomState(self.random_state)
        self.weight_vector = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.weight_vector_initialized = True

    def __update_weights(self, training_data_vector: "np.ndarray[float]", class_label: float) -> float:
        """ADALINEの学習規則を用いて重みを更新

        Args:
            training_data_vector (np.ndarray[float]): [特徴ベクトル]
            class_label (float): [正解ラベル]

        Returns:
            float: [コスト]
        """
        # 活性化関数の出力の計算
        output = self.__activation(self.__net_input(training_data_vector))
        # 誤差の計算
        error = (class_label - output)
        # 重みの更新
        self.weight_vector[1:] += self.learning_rate * training_data_vector.dot(error)
        # バイアスユニットの更新
        self.weight_vector[0] += self.learning_rate * error
        # コストの計算
        cost = 0.5 * error ** 2
        return cost

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
        return net_input

    def predict(self, training_data_matrix: "np.ndarray[[float]]") -> "np.ndarray[int]":
        """予測クラスラベルを返す

        Args:
            training_data_matrix (np.ndarray[[float]]): [shape = [サンプルの数, 特徴ベクトルの要素数]]

        Returns:
            np.ndarray[int]: [shape = [特徴ベクトルの要素数]]
        """
        return np.where(self.__net_input(training_data_matrix) > 0.0, 1, -1)
