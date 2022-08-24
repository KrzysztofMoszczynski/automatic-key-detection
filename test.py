import time

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from utils import prepare_data, validate_models, functions
from utils.song_dataset import SongsDataset
from constants import BATCH_SIZE, BEST_TRAIN_MODEL, MODELS_PATH, PATH_MATRIX, KEY_LABELS
import torch
import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np


device = None
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

model = BEST_TRAIN_MODEL.model.to(device)
model.load_state_dict(torch.load(MODELS_PATH + "/net_" + BEST_TRAIN_MODEL.name + ".pth"))

train_data, test_data = prepare_data.get_train_test_sets()
train_dataset = SongsDataset(train_data)
train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataset = SongsDataset(test_data)
test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE)

test_start_time = time.time()
y_pred_test, y_true_test, test_accuracy = validate_models.predict_test_dl(model, test_dl)
y_pred_test_keys = functions.map_labels_to_keys(y_pred_test)
y_true_test_keys = functions.map_labels_to_keys(y_true_test)
evaluation_score = validate_models.count_evaluation_score(y_true_test_keys, y_pred_test_keys)
print("Dokładność dla zbioru testowego: ", round(test_accuracy, 2), "%")
print(f"Punktacja Mirex dla zbioru testowego: {round(evaluation_score * 100, 2)}%")
matrix = pd.DataFrame(confusion_matrix(y_true_test, y_pred_test, labels=np.arange(0, 24)))  # Matrix
test_end_time = time.time()
matrix.to_csv(PATH_MATRIX + "test_confusion_matrix_1.csv")
print(matrix)
print(f'Klasyfikacja zbioru testowego zakończona w {round(test_end_time - test_start_time, 2)} sekund')

train_start_time = time.time()
y_pred_train, y_true_train, train_accuracy = validate_models.predict_training_dl(model, train_dl)
y_pred_train_keys = functions.map_labels_to_keys(y_pred_train)
y_true_train_keys = functions.map_labels_to_keys(y_true_train)
evaluation_score = validate_models.count_evaluation_score(y_true_train_keys, y_pred_train_keys)
print("Dokładność dla zbioru treningowego: ", round(train_accuracy, 2), "%")
print(f"Punktacja Mirex dla zbioru treningowego: {round(evaluation_score * 100, 2)}%")
matrix = pd.DataFrame(confusion_matrix(y_true_train, y_pred_train, labels=np.arange(0, 24)))  # Matrix
train_end_time = time.time()
matrix.to_csv(PATH_MATRIX + "train_confusion_matrix_1.csv")
print(matrix)
print(f'Klasyfikacja zbioru treningowego zakończona w {round(train_end_time- train_start_time, 2)} sekund')