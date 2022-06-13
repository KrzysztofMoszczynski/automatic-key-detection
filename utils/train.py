import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from constants import CROSS_VAL_EPOCHS, BATCH_SIZE, CROSS_VAL_EPOCHS_NN
from utils.models import get_models_for_training, get_models
from utils.functions import split_set_randomly, flatten_df_arr, transform_to_tensor
import tensorflow as tf
from keras.optimizers import SGD, Adam
from keras.losses import sparse_categorical_crossentropy
from torch.utils.data import DataLoader
from utils.song_dataset import SongsDataset
from ast import literal_eval


def get_device():
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def validate(model, data):
    total = 0
    correct = 0
    for i, (pitches, keys) in enumerate(data):
        pitches = pitches.cuda()
        x = model(pitches.float())
        value, pred = torch.max(x, 1)
        pred = pred.data.cpu()
        total += x.size(0)
        correct += torch.sum(pred == keys)

    return correct * 100. / total


def cel(model, data, ce):  # cross-entropy
    results = []

    with(torch.set_grad_enabled(False)):
        for i, (pitches, keys) in enumerate(data):
            pitches = pitches.cuda()
            keys = keys.cuda()
            pred = model(pitches.float())
            results.append(ce(pred, keys))
    return sum(results) / len(results)


def train_the_best_model(data, folds, lr=0.001):
    device = get_device()
    models = get_models_for_training()
    data_folds = split_set_randomly(data, folds)
    model_accuracy = []
    for model in models:
        print("Training model: ", model.name)
        folds_accuracies = []
        model = model.model
        accuracies = []
        training_losses = []
        validation_losses = []
        for index, val_fold in enumerate(data_folds):
            cnn = model.to(device)
            ce = nn.CrossEntropyLoss()
            optimizer = optim.Adam(cnn.parameters(), lr=lr)
            accuracy = 0
            training_data = data_folds.copy()
            training_data.pop(index)
            train_data = flatten_df_arr(data_folds)
            train_dataset = SongsDataset(train_data)
            train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE)
            val_dataset = SongsDataset(val_fold)
            val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE)
            '''training_features = train_data['pitches'].apply(literal_eval)
            training_labels = train_data['key']
            train_selected = pd.DataFrame([training_features, training_labels]).transpose()
            train_selected = train_selected.values.T.tolist()
            test_features = val_fold['pitches'].apply(literal_eval)
            test_labels = val_fold['key']
            print(train_selected)
            train_dl = transform_to_tensor(train_selected)
            val_dl = transform_to_tensor(val_fold)'''
            for epoch in range(CROSS_VAL_EPOCHS):
                losses = []
                fold_accuracies = []
                fold_training_losses = []
                fold_validation_losses = []
                for i, (pitches, keys) in enumerate(train_dl):
                    pitches = pitches.to(device)
                    keys = keys.to(device)
                    optimizer.zero_grad()
                    pred = cnn(pitches.float())
                    loss = ce(pred, keys)
                    losses.append(loss)
                    loss.backward()
                    optimizer.step()
                accuracy = float(validate(cnn, val_dl))
                training_loss = float(sum(losses) / len(losses))
                validation_loss = float(cel(model, val_dl, ce))
                fold_training_losses.append(training_loss)
                fold_validation_losses.append(validation_loss)
                fold_accuracies.append(accuracy)
            fold_best_accuracy = max(accuracies)
            accuracies.append(fold_best_accuracy)



def train_best_model(data, folds, lr=0.01):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    models = get_models()
    data_folds = split_set_randomly(data, folds)
    model_accuracy = []
    for model in models:
        folds_accuracies = []
        for index, val_fold in enumerate(data_folds):
            accuracy = 0
            model.compile(optimizer=SGD(learning_rate=lr, clipnorm=1), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            training_data = data_folds.copy()
            training_data.pop(index)
            train_data = flatten_df_arr(training_data)
            training_features = train_data['pitches'].apply(literal_eval)
            training_labels = train_data['key']
            test_features = val_fold['pitches'].apply(literal_eval)
            test_labels = val_fold['key']
            model.fit(x=training_features.tolist(), y=training_labels.tolist(), batch_size=BATCH_SIZE, epochs=CROSS_VAL_EPOCHS_NN, verbose=2)
            exit(0)


def map_keys(x):
    if x >= 30:
        x = x - 17
    return x