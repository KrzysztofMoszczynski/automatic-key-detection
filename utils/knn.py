from sklearn.neighbors import KNeighborsClassifier
import time
from ast import literal_eval
import numpy as np
from utils.functions import flatten_df_arr, split_set_randomly
from constants import MIN_K_IMPROVEMENT
import utils.validate_models as validate_models


class KNeighbours():
    model = None
    final_k = 0
    data = None
    labels = None
    max_k = 0
    batches = 0

    def __init__(self, data, max_k=10, final_k=0, batches=10):
        self.data = data
        self.max_k = max_k
        self.final_k = final_k
        self.batches = batches

    def train_k(self):
        start_time = time.time()
        data_folds = split_set_randomly(self.data, self.batches)
        best_k = 0
        best_accuracy = 0
        best_score = 0
        for k in range(1, self.max_k + 1):
            print(f'Testing k={k}')
            knn = KNeighborsClassifier(k)
            k_score = 0
            k_accuracy = 0
            for index, test_fold in enumerate(data_folds):
                training_data = data_folds.copy()
                training_data.pop(index)
                training_data = flatten_df_arr(training_data)
                training_features = training_data['pitches'].apply(literal_eval)
                training_labels = training_data['key']
                test_features = test_fold['pitches'].apply(literal_eval)
                test_labels = test_fold['key']
                knn.fit(training_features.tolist(), training_labels.tolist())
                predictions = knn.predict(test_features.tolist())
                score = validate_models.count_evaluation_score(predictions, test_labels.tolist())
                accuracy = knn.score(test_features.tolist(), test_labels.tolist())
                k_score = k_score + score
                k_accuracy = k_accuracy + accuracy
            k_score = k_score / len(data_folds)
            k_accuracy = k_accuracy / len(data_folds)
            print(f'Score for k={k} is {k_score}')
            print(f'Accuracy for k={k} is {k_accuracy}')
            if k_score - best_score > MIN_K_IMPROVEMENT:
                best_score = k_score
                best_k = k
                print(f'New best k was discovered. Score: {best_score}, best k number: {k}')
        end_time = time.time()
        print(f'Finding best hyperparameter k finished. Best k={best_k}. Process took {end_time - start_time} seconds.')

    def validate_knn(self, test_data):
        knn = KNeighborsClassifier(self.final_k)
        training_features = self.data['pitches'].apply(literal_eval)
        training_labels = self.data['key']
        test_features = test_data['pitches'].apply(literal_eval)
        test_labels = test_data['key']
        knn.fit(training_features.tolist(), training_labels.tolist())
        predictions = knn.predict(test_features.tolist())
        evaluation_score = validate_models.count_evaluation_score(predictions, test_labels.tolist())
        accuracy = knn.score(test_features.tolist(), test_labels.tolist())
        print(f'Accuracy: {accuracy}')

