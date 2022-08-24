import time
from utils import knn
from constants import BEST_K
from sklearn.model_selection import train_test_split
import torch
from utils.train import validate
import numpy as np


def test_knn(data):
    start_time = time.time()
    training_data, test_data = train_test_split(data)
    k_neighbours = knn.KNeighbours(training_data, final_k=BEST_K)
    k_neighbours.validate_knn(test_data)
    end_time = time.time()
    print(f'Klasyfikacja dla zbioru testowego zakończona w {round(end_time - start_time,2)} sekund')


def validate_random_song(data):
    start_time = time.time()
    training_data, test_sample = train_test_split(data.copy(), test_size=1)
    k_neighbours = knn.KNeighbours(training_data, final_k=BEST_K)
    k_neighbours.validate_knn(test_sample)
    end_time = time.time()
    print(f'Classification done in {end_time - start_time} seconds')


'''
Function counts a score for the classified songs basing on the following scoring:
Relation to Correct Key    |    Points
Same                       |    1.0
Perfect fifth              |    0.5
Relative major/minor       |    0.3
Parallel major/minor       |    0.2
Other                      |    0.0   
'''
def count_evaluation_score(predicted_labels, true_labels):
    score = 0.0
    for pred_label, true_label in zip(predicted_labels, true_labels):
        difference = abs(pred_label - true_label)
        if difference == 0:
            score = score + 1
        elif difference == 7 or difference == 5:
            score = score + 0.5
        elif difference == 21 or difference == 33:
            score = score + 0.3
        elif difference == 30:
            score = score + 0.2
    evaluation_score = score/len(true_labels)
    return evaluation_score



def predict_test_dl(model, data):
    y_pred = []
    y_true = []
    for i, (pitches, keys) in enumerate(data):
        pitches = pitches.cuda()
        x = model(pitches.float())
        value, pred = torch.max(x, 1)
        pred = pred.data.cpu()
        y_pred.extend(list(pred.numpy()))
        y_true.extend(list(keys.numpy()))
    accuracy = float(validate(model, data))
    return np.array(y_pred), np.array(y_true), accuracy


def predict_training_dl(model, data):
    y_pred = []
    y_true = []
    for i, (pitches, keys) in enumerate(data):
        pitches = pitches.cuda()
        x = model(pitches.float())
        value, pred = torch.max(x, 1)
        pred = pred.data.cpu()
        y_pred.extend(list(pred.numpy()))
        y_true.extend(list(keys.numpy()))
    accuracy = float(validate(model, data))
    return np.array(y_pred), np.array(y_true), accuracy


