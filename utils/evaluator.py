# encoding: utf-8
# author: zhaotianhong
# contact: zhaotianhong2016@email.szu.edu.cn


import os
import json
import datetime
import pandas as pd
from utils import loss
import numpy as np


def collect( y_pred, y_true, config):
    """
    Args:
        batch(dict): 输入数据，字典类型，包含两个Key:(y_true, y_pred):
            batch['y_true']: (num_samples/batch_size, timeslots, ..., feature_dim)
            batch['y_pred']: (num_samples/batch_size, timeslots, ..., feature_dim)
    """
    intermediate_result = {}
    len_timeslots = config.output_window
    for i in range(1, len_timeslots + 1):
        for metric in config.metrics:
            if metric + '@' + str(i) not in intermediate_result:
                intermediate_result[metric + '@' + str(i)] = []

    for i in range(1, len_timeslots + 1):
        for metric in config.metrics:
            if metric == 'masked_MAE':
                intermediate_result[metric + '@' + str(i)].append(
                    loss.masked_mae_torch(y_pred[:, i - 1], y_true[:, i - 1], 0).item())
            elif metric == 'masked_MSE':
                intermediate_result[metric + '@' + str(i)].append(
                    loss.masked_mse_torch(y_pred[:, i - 1], y_true[:, i - 1], 0).item())
            elif metric == 'masked_RMSE':
                intermediate_result[metric + '@' + str(i)].append(
                    loss.masked_rmse_torch(y_pred[:, i - 1], y_true[:, i - 1], 0).item())
            elif metric == 'masked_MAPE':
                intermediate_result[metric + '@' + str(i)].append(
                    loss.masked_mape_torch(y_pred[:, i - 1], y_true[:, i - 1], 0).item())
            elif metric == 'MAE':
                intermediate_result[metric + '@' + str(i)].append(
                    loss.masked_mae_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
            elif metric == 'MSE':
                intermediate_result[metric + '@' + str(i)].append(
                    loss.masked_mse_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
            elif metric == 'RMSE':
                intermediate_result[metric + '@' + str(i)].append(
                    loss.masked_rmse_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
            elif metric == 'MAPE':
                intermediate_result[metric + '@' + str(i)].append(
                    loss.masked_mape_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
            elif metric == 'R2':
                intermediate_result[metric + '@' + str(i)].append(
                    loss.r2_score_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
            elif metric == 'EVAR':
                intermediate_result[metric + '@' + str(i)].append(
                    loss.explained_variance_score_torch(y_pred[:, i - 1], y_true[:, i - 1]).item())
    return intermediate_result


def save_result(intermediate_result, config, info):
    """
    将评估结果保存到 save_path 文件夹下的 filename 文件中
    Args:
        save_path: 保存路径
        filename: 保存文件名
    """

    dataframe = {}
    for metric in config.metrics:
        dataframe[metric] = []
    for i in range(1, config.output_window + 1):
        for metric in config.metrics:
            dataframe[metric].append(np.mean(intermediate_result[metric+'@'+str(i)]))
    dataframe = pd.DataFrame(dataframe, index=range(1, config.output_window + 1))
    path = config.result_path.replace('.csv', info+'.csv')
    dataframe.to_csv(path, index=False)

    return dataframe

