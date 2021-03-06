from torch import nn
import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy


class Model:
    def __init__(self, name, model):
        self.name = name
        self.model = model


def get_models_for_training():
    model_0_h_layers = Model("model_0_h_layers", nn.Sequential(
        nn.Linear(12, 24)
    ))

    model_1_h_layer_10 = Model("model_1_h_layer_10", nn.Sequential(
        nn.Linear(12, 10),
        nn.ReLU(),
        nn.Linear(10, 24)
    ))

    model_1_h_layer_20 = Model("model_1_h_layer_20", nn.Sequential(
        nn.Linear(12, 20),
        nn.ReLU(),
        nn.Linear(20, 24)
    ))

    model_1_h_layer_40 = Model("model_1_h_layer_40", nn.Sequential(
        nn.Linear(12, 40),
        nn.ReLU(),
        nn.Linear(40, 24)
    ))

    model_1_h_layer_60 = Model("model_1_h_layer_60", nn.Sequential(
        nn.Linear(12, 60),
        nn.ReLU(),
        nn.Linear(60, 24)
    ))

    model_2_h_layers_20_30 = Model("model_2_h_layers_20_30", nn.Sequential(
        nn.Linear(12, 20),
        nn.ReLU(),
        nn.Linear(20, 30),
        nn.ReLU(),
        nn.Linear(30, 24)
    ))

    model_2_h_layers_30_60 = Model("model_2_h_layers_30_60", nn.Sequential(
        nn.Linear(12, 30),
        nn.ReLU(),
        nn.Linear(30, 60),
        nn.ReLU(),
        nn.Linear(60, 24)
    ))

    model_2_h_layers_60_30 = Model("model_2_h_layers_60_30", nn.Sequential(
        nn.Linear(12, 60),
        nn.ReLU(),
        nn.Linear(60, 30),
        nn.ReLU(),
        nn.Linear(30, 24)
    ))

    model_2_h_layers_50_50 = Model("model_2_h_layers_50_50", nn.Sequential(
        nn.Linear(12, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 24)
    ))

    model_3_h_layers_30_30_30 = Model("model_3_h_layers_30_30_30", nn.Sequential(
        nn.Linear(12, 30),
        nn.ReLU(),
        nn.Linear(30, 30),
        nn.ReLU(),
        nn.Linear(30, 30),
        nn.ReLU(),
        nn.Linear(30, 24)
    ))

    model_3_h_layers_30_40_30 = Model("model_3_h_layers_30_40_30", nn.Sequential(
        nn.Linear(12, 30),
        nn.ReLU(),
        nn.Linear(30, 40),
        nn.ReLU(),
        nn.Linear(40, 30),
        nn.ReLU(),
        nn.Linear(30, 24)
    ))

    return [model_0_h_layers, model_1_h_layer_10, model_1_h_layer_20, model_1_h_layer_40, model_1_h_layer_60,
            model_2_h_layers_20_30, model_2_h_layers_30_60, model_2_h_layers_60_30, model_2_h_layers_50_50,
            model_3_h_layers_30_30_30, model_3_h_layers_30_40_30]


def get_models():
    model_1_h_layer_10 = Sequential([
        Dense(units=10, input_shape=(12,), activation='relu'),
        Dense(units=24, activation='softmax')
    ])

    model_1_h_layer_20 = Sequential([
        Dense(units=20, input_shape=(12,), activation='relu'),
        Dense(units=24, activation='softmax')
    ])

    model_1_h_layer_40 = Sequential([
        Dense(units=40, input_shape=(12,), activation='relu'),
        Dense(units=24, activation='softmax')
    ])

    model_1_h_layer_60 = Sequential([
        Dense(units=60, input_shape=(12,), activation='relu'),
        Dense(units=24, activation='softmax')
    ])

    model_2_h_layers_20_30 = Sequential([
        Dense(units=20, input_shape=(12,), activation='relu'),
        Dense(units=30, activation='relu'),
        Dense(units=24, activation='softmax')
    ])

    model_2_h_layers_30_60 = Sequential([
        Dense(units=30, input_shape=(12,), activation='relu'),
        Dense(units=60, activation='relu'),
        Dense(units=24, activation='softmax')
    ])

    model_2_h_layers_60_40 = Sequential([
        Dense(units=60, input_shape=(12,), activation='relu'),
        Dense(units=40, activation='relu'),
        Dense(units=24, activation='softmax')
    ])

    model_2_h_layers_50_50 = Sequential([
        Dense(units=50, input_shape=(12,), activation='relu'),
        Dense(units=60, activation='relu'),
        Dense(units=24, activation='softmax')
    ])

    model_3_h_layers_20_30_30 = Sequential([
        Dense(units=20, input_shape=(12,), activation='relu'),
        Dense(units=30, activation='relu'),
        Dense(units=30, activation='relu'),
        Dense(units=24, activation='softmax'),
    ])

    return [model_1_h_layer_10, model_1_h_layer_20, model_1_h_layer_40, model_1_h_layer_60, model_2_h_layers_20_30, model_2_h_layers_30_60,
            model_2_h_layers_60_40, model_2_h_layers_50_50, model_3_h_layers_20_30_30]