from sklearn.neighbors import KNeighborsClassifier
import random
import numpy as np
import copy


class KNeighbours():

    model = None
    final_k = 0
    data = None
    labels = None
    max_k = 0
    batches = 0

    def __init__(self, data, max_k, final_k=0, batches=10):
        self.data = data
        self.max_k = max_k
        self.final_k = final_k
        self.batches = batches

    def train_k(self):
        for k in range(self.max_k):
            data_folds = self.split_set_randomly()
            for test_fold in data_folds:
                training_data = data_folds.copy()
                training_data.remove(test_fold)
                

    def split_set_randomly(self):
        data = copy.copy(self.data)
        random.shuffle(data)
        data_folds = np.split(data, self.batches)
        return data_folds


