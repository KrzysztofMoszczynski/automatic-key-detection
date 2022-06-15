import pandas as pd
import numpy as np
import torchvision
import torch
from constants import BATCH_SIZE


def flatten_arr(arr):
    return_arr = []
    for element in arr:
        return_arr += element
    return return_arr


def map_labels_to_keys(labels):
    keys = []
    labels_it = labels.copy()
    for label in labels_it:
        if label >= 12:
            label = label + 18
        keys.append(label)
    return keys


def flatten_df_arr(arr):
    return_df = pd.DataFrame(columns=arr[0].columns)
    for dataframe in arr:
        return_df = return_df.append(dataframe)
    return return_df


def split_set_randomly(data, batches):
    data = data.copy()
    data = data.sample(n=len(data.index)).reset_index(drop=True)
    data_folds = np.array_split(data, batches)
    return data_folds


def transform_to_tensor(data):
    tensor_data = torch.tensor(data)
    return torch.utils.data.DataLoader(tensor_data, batch_size=BATCH_SIZE)
