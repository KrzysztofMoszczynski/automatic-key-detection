import copy

import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import os
from constants import BATCH_SIZE, CROSS_VAL_EPOCHS_NN, EPOCHS, MODELS_PATH
from utils.models import get_models_for_training
from utils.functions import split_set_randomly, flatten_df_arr
from torch.utils.data import DataLoader
from utils.song_dataset import SongsDataset
from sklearn.model_selection import train_test_split
import time


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


def find_the_best_model(data, folds, lr=0.001, train_best_model=False):
    device = get_device()
    models = get_models_for_training()
    data_folds = split_set_randomly(data, folds)
    accuracies = []
    best_model = None
    best_model_accuracy = 0
    for model_obj in models:
        start_time = time.time()
        print("Trening modelu: ", model_obj.name)
        model = model_obj.model
        model_accuracies = []
        for index, val_fold in enumerate(data_folds):
            print(f"Podzbiór {index + 1}/{folds}")
            cnn = model.to(device)
            ce = nn.CrossEntropyLoss()
            optimizer = optim.Adam(cnn.parameters(), lr=lr)
            fold_accuracies = []
            training_data = data_folds.copy()
            training_data.pop(index)
            train_data = flatten_df_arr(data_folds)
            train_dataset = SongsDataset(train_data)
            train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE)
            val_dataset = SongsDataset(val_fold)
            val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE)
            for epoch in range(CROSS_VAL_EPOCHS_NN):
                losses = []
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
            fold_best_accuracy = max(fold_accuracies)
            model_accuracies.append(fold_best_accuracy)
        model_accuracy = sum(model_accuracies) / len(model_accuracies)
        if model_accuracy > best_model_accuracy:
            best_model_accuracy = model_accuracy
            best_model = model_obj
        end_time = time.time()
        print(f"Trening modelu {model_obj.name} zakończony w {round(end_time-start_time, 2)} sekund. "
              f"Średnia dokładność dla tego modelu wyniosła {round(model_accuracy, 2)}%")
    if train_best_model:
        train_model(best_model, data, lr)


def train_model(model_obj, data, lr=0.001):
    device = get_device()
    train_data, val_data = train_test_split(data)
    accuracies = []
    model = model_obj.model
    model_name = model_obj.name
    training_losses = []
    validation_losses = []
    cnn = model.to(device)
    ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=lr)
    start_time = time.time()
    best_model = None
    best_accuracy = 0
    should_stop_in_3_epochs = False
    stop_counter = 0
    train_dataset = SongsDataset(train_data)
    train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_dataset = SongsDataset(val_data)
    val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    for epoch in range(EPOCHS):
        print(f"EPOKA {epoch + 1}/{EPOCHS}")
        losses = []
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
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        accuracies.append(accuracy)
        if accuracy > best_accuracy:
            print(f"Znaleziono najlepszy model z dokładnością wynoszącą {round(accuracy, 2)}%")
            best_accuracy = accuracy
            best_model = copy.deepcopy(cnn)
        if should_stop_in_3_epochs is True:
            if stop_counter >= 3:
                print(
                    f"Osiągnięto kryterium stopu. Najlepsza dokładność wyniosła {round(best_accuracy, 2)}%")
                break
            else:
                stop_counter = stop_counter + 1
        elif len(accuracies) > 2 and accuracy < best_accuracy and accuracy < accuracies[epoch-1]:
            should_stop_in_3_epochs = True
    end_time = time.time()
    print(f"Trening zakończony w {round(end_time-start_time, 2)} sekund.")
    file_path = os.path.join(MODELS_PATH, 'net_{}.pth'.format(model_name))
    torch.save(best_model.state_dict(), file_path)
    plt.plot(accuracies)
    plt.savefig('accuracies_{}.png'.format(model_name))
    plt.clf()
    plt.plot(training_losses, label="training")
    plt.plot(validation_losses, label="validation")
    plt.legend()
    plt.savefig('errors_{}.png'.format(model_name))

'''def train_best_model(data, folds, lr=0.01):
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
            exit(0)'''


def map_keys(x):
    if x >= 30:
        x = x - 17
    return x