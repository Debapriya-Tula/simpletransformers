#!/usr/bin/env python
# TextAugment: mixup
#
# Copyright (C) 2018-2020
# Authors: Joseph Sefara, Vukosi Marivate
#
# URL: <https://github.com/dsfsi/textaugment/>
# For license information, see LICENSE
import numpy as np
import random
import torch
from .cmi_loss import CMILoss


class MIXUP:
    """
    This class implements the mixup algorithm [1] for natural language processing.
    [1] Zhang, Hongyi, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. "mixup: Beyond empirical risk
    minimization." in International Conference on Learning Representations (2018).
    https://openreview.net/forum?id=r1Ddp1-Rb
    """

    @staticmethod
    def validate(**kwargs):
        """Validate input data"""

        if "data" in kwargs:
            if isinstance(kwargs["data"], list):
                kwargs["data"] = np.array(kwargs["data"])
            if not isinstance(kwargs["data"], torch.Tensor):
                raise TypeError(
                    "data must be numpy array or torch tensor. Found "
                    + str(type(kwargs["data"]))
                )
        if "labels" in kwargs:
            if isinstance(kwargs["labels"], (list, type(None))):
                kwargs["labels"] = np.array(kwargs["labels"])
            if not isinstance(kwargs["labels"], torch.Tensor):
                raise TypeError(
                    "labels must be numpy array. Found " + str(type(kwargs["labels"]))
                )
        if "batch_size" in kwargs:
            if not isinstance(kwargs["batch_size"], int):
                raise TypeError(
                    "batch_size must be a valid integer. Found "
                    + str(type(kwargs["batch_size"]))
                )
        if "shuffle" in kwargs:
            if not isinstance(kwargs["shuffle"], bool):
                raise TypeError(
                    "shuffle must be a boolean. Found " + str(type(kwargs["shuffle"]))
                )
        if "runs" in kwargs:
            if not isinstance(kwargs["runs"], int):
                raise TypeError(
                    "runs must be a valid integer. Found " + str(type(kwargs["runs"]))
                )

    def __init__(self, random_state=1, runs=1):
        self.random_state = random_state
        self.runs = runs
        if isinstance(self.random_state, int):
            random.seed(self.random_state)
            np.random.seed(self.random_state)
        else:
            raise TypeError("random_state must have type int")

    def mixup_data(self, x, y=None, alpha=0.4):
        """This method performs mixup. If runs = 1 it just does 1 mixup with whole batch, any n of runs
        creates many mixup matches.
        :type x: Numpy array
        :param x: Data array
        :type y: Numpy array
        :param y: (optional) labels
        :type alpha: float
        :param alpha: alpha
        :rtype: tuple
        :return: Returns mixed inputs, pairs of targets, and lambda
        """
        x = x.cpu()
        y = y.cpu()
        if self.runs is None:
            self.runs = 1
        output_x = []
        output_y = []
        batch_size = x.shape[0]

        lam_vector = np.random.beta(alpha, alpha, batch_size)
        index = np.random.permutation(batch_size)
        mixed_x = (x.T * lam_vector).T + (x[index, :].T * (1.0 - lam_vector)).T
        output_x.append(mixed_x)
        if y is None:
            return torch.cat(output_x, axis=0)
        # mixed_y = (y.T * lam_vector).T + (y[index].T * (1.0 - lam_vector)).T
        # output_y.append(mixed_y)
        # return torch.cat(output_x, axis=0), torch.cat(output_y, axis=0)
        y_a, y_b = y, y[index]
        return torch.cat(output_x, axis=0), [y_a, y_b, torch.from_numpy(lam_vector)]

    @staticmethod
    def mixup_criterion(
        criterion, pred, y_a, y_b, lam, processed_df=None, base_lang=None
    ):
        if isinstance(criterion, CMILoss):
            return (
                lam * criterion(pred, y_a, processed_df, base_lang)
                + (1 - lam) * criterion(pred, y_b, processed_df, base_lang)
            ).mean()
        else:
            return (lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)).mean()

    def flow(self, data, labels=None, batch_size=32, shuffle=True, runs=1):
        """This function implements the batch iterator and specifically calls mixup
        :param data: Input data. Numpy ndarray or list of lists.
        :param labels: Labels. Numpy ndarray or list of lists.
        :param batch_size: Int (default: 32).
        :param shuffle: Boolean (default: True).
        :param runs: Int (default: 1). Number of augmentations
        :rtype:   array or tuple
        :return:  array or tuple of arrays (X_data array, labels array)."""

        self.validate(
            data=data, labels=labels, batch_size=batch_size, shuffle=shuffle, runs=runs
        )

        self.runs = runs

        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1

        def data_generator():
            data_size = len(data)
            while True:
                # Shuffle the data at each epoch
                if shuffle:
                    shuffle_indices = np.random.permutation(np.arange(data_size))
                    shuffled_data = data[shuffle_indices]
                    if labels is not None:
                        shuffled_labels = labels[shuffle_indices]
                else:
                    shuffled_data = data
                    if labels is not None:
                        shuffled_labels = labels
                for batch_num in range(num_batches_per_epoch):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size)
                    X = shuffled_data[start_index:end_index]
                    if labels is None:
                        X = self.mixup_data(X, y=None)
                        yield X
                    else:
                        y = shuffled_labels[start_index:end_index]
                        X, y = self.mixup_data(X, y)
                        yield X, y

        return data_generator(), num_batches_per_epoch
