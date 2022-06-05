import pandas as pd


def flatten_arr(arr):
    return_arr = []
    for element in arr:
        return_arr += element
    return return_arr


def flatten_df_arr(arr):
    return_df = pd.DataFrame(columns=arr[0].columns)
    for dataframe in arr:
        return_df = return_df.append(dataframe)
    return return_df


