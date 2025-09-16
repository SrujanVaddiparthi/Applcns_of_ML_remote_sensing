from copy import deepcopy
import copy
import numpy as np


def load_data(data_path):
    data = np.load(data_path)
    data_copy = copy.deepcopy(data)
    return data_copy

# oh this for prob 2 cools
# def desc_stats(data):
#     ht , wd, bands = data.shape
#     for i in range(bands):
#         arr = data[i, :, :]
#         band = i + 1
#         min = np.min(arr)
#         max = np.max(arr)
#         mean = np.mean(arr)
#         std = np.std(arr)
#
#prob 1: handling of no data with nan replacement
def replace_with_nan(arr: np.ndarray, no_data_value = 0):
    nan_replaced = arr.astype(float, copy=True)
    nan_replaced[arr == no_data_value] = np.nan # dam i love the automcomplete here lol
    return nan_replaced

