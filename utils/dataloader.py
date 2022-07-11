# encoding: utf-8
# author: zhaotianhong
# contact: zhaotianhong2016@email.szu.edu.cn

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from utils.normalization import StandardScaler, NormalScaler, NoneScaler, \
    MinMax01Scaler, MinMax11Scaler, LogScaler
from utils import executor


def get_dataloader(reader, config):
    log_info = "Dataset loading"
    print(executor.get_local_time(), log_info)
    executor.logging(log_info, config)

    # 读取原始数据
    x_train_bus, y_train_bus, x_val_bus, y_val_bus, x_test_bus, y_test_bus = reader._get_value_data(
        reader.bus_path, "bus.npz")
    x_train_metro, y_train_metro, x_val_metro, y_val_metro, x_test_metro, y_test_metro = reader._get_value_data(
        reader.metro_path, "metro.npz")
    x_train_taxi, y_train_taxi, x_val_taxi, y_val_taxi, x_test_taxi, y_test_taxi = reader._get_value_data(
        reader.taxi_path, "taxi.npz")

    # 三种模式在一起归一化
    concat_x = np.concatenate((x_train_bus, x_train_metro, x_train_taxi), axis=0)
    scaler = reader._get_scalar(reader.scaler_type, concat_x, concat_x)
    log_info = 'scaler mean: %s, std: %s'%(scaler.mean, scaler.std)
    print(executor.get_local_time(), log_info)
    executor.logging(log_info, config)


    # 数据归一化
    bus_train_data, bus_eval_data, bus_test_data, scaler_bus = reader._scaler_data(
        x_train_bus, y_train_bus, x_val_bus, y_val_bus, x_test_bus, y_test_bus, scaler)
    metro_train_data, metro_eval_data, metro_test_data, scaler_metro = reader._scaler_data(
        x_train_metro, y_train_metro, x_val_metro, y_val_metro, x_test_metro, y_test_metro, scaler)
    taxi_train_data, taxi_eval_data, taxi_test_data, scaler_taxi = reader._scaler_data(
        x_train_taxi, y_train_taxi, x_val_taxi, y_val_taxi, x_test_taxi, y_test_taxi, scaler)

    # 构建训练，验证，测试数据集
    train_dataset = Bus_Metro_Taxi_Dataset(config, bus_train_data, metro_train_data, taxi_train_data)
    val_dataset = Bus_Metro_Taxi_Dataset(config, bus_eval_data, metro_eval_data, taxi_eval_data)
    test_dataset = Bus_Metro_Taxi_Dataset(config, bus_test_data, metro_test_data, taxi_test_data)

    config.scaler = scaler
    config.scaler_bus = scaler_bus
    config.scaler_metro = scaler_metro
    config.scaler_taxi = scaler_taxi

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    log_info = "Dataset loading completed"
    print(executor.get_local_time(), log_info)
    executor.logging(log_info, config)
    return train_loader, val_loader, test_loader



class Bus_Metro_Taxi_Dataset(Dataset):
    def __init__(self, config, bus_data, metro_data, taxi_data):
        self.od_graph = config.od_graph
        self.sa_graph = config.sa_graph

        self.bus_data = bus_data
        self.metro_data = metro_data
        self.taxi_data = taxi_data

    def __len__(self):
        return len(self.bus_data)

    def __getitem__(self, idx):
        bus = self.bus_data[idx][0]
        metro = self.metro_data[idx][0]
        taxi = self.taxi_data[idx][0]

        target_bus = self.bus_data[idx][1]
        target_metro = self.metro_data[idx][1]
        target_taxi = self.taxi_data[idx][1]

        return bus, metro, taxi, target_bus, target_metro, target_taxi


class Read_Data():
    def __init__(self, config):
        self.input_window = config.input_window
        self.output_channel = config.output_channel
        self.cache_folder = os.path.join("dataset/cache", "His_%s_Pre_%s" % (config.input_window, config.output_channel))
        self.scaler_type = "standard"  # normal
        self.output_dim = 1
        self.train_rate = config.train_rate
        self.eval_rate = config.eval_rate
        self.od_rel_path = config.od_rel_path
        self.sa_rel_path = config.sa_rel_path
        self.geo_path = config.geo_path
        self.bus_path = config.bus_path
        self.metro_path = config.metro_path
        self.taxi_path = config.taxi_path

        self.geo_to_ind = {}
        self.geo_ids = self._load_geo()

        # self.od_graph = self._get_od_adjacency_matrix()
        # self.sa_graph = self._get_sa_adjacency_matrix()

        print(executor.get_local_time(), 'Loading adjacency matrix')

        config.od_graph = self._get_od_adjacency_matrix(config.od_rel_path)
        config.od_bus_graph = self._get_od_adjacency_matrix(config.od_bus_rel_path)
        config.od_metro_graph = self._get_od_adjacency_matrix(config.od_metro_rel_path)
        config.od_taxi_graph = self._get_od_adjacency_matrix(config.od_taxi_rel_path)

        config.sa_graph = self._get_sa_adjacency_matrix()
        print(executor.get_local_time(),'Loading adjacency matrix completed')

        # self.bus_values = self._get_value_data(self.bus_path, "bus.npz")
        # self.metro_values = self._get_value_data(self.metro_path, "metro.npz")
        # self.taxi_values = self._get_value_data(self.taxi_path, "taxi.npz")



    def _get_value_data(self, dyna_path, cache_name):
        if os.path.exists(self.cache_folder):  # 文件夹
            cache = os.path.join(self.cache_folder, cache_name)
        else:
            os.makedirs(self.cache_folder)
            cache = os.path.join(self.cache_folder, cache_name)

        if os.path.exists(cache):
            x_train, y_train, x_val, y_val, x_test, y_test = self._load_cache_train_val_test(cache)
            return x_train, y_train, x_val, y_val, x_test, y_test

        else:
            raw = self._load_dyna(dyna_path)
            x, y = self._generate_input_data(raw)
            cache = os.path.join(self.cache_folder, cache_name)
            x_train, y_train, x_val, y_val, x_test, y_test = self._split_train_val_test(x,y,cache)
            # train_data, eval_data, test_data, scaler = self._scaler_data(x_train, y_train, x_val, y_val, x_test, y_test, scaler)
            # return train_data, eval_data, test_data, scaler

            return x_train, y_train, x_val, y_val, x_test, y_test


    def _load_geo(self):
        geofile = pd.read_csv(self.geo_path)
        geo_ids = list(geofile['geo_id'])
        for index, idx in enumerate(geo_ids):
            self.geo_to_ind[idx] = index
        return geo_ids

    def _get_od_adjacency_matrix(self, path):
        relfile = pd.read_csv(path)
        max = relfile['link_weight'].max()
        distance_df = relfile[['origin_id', 'destination_id', 'link_weight']]
        adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        for row in distance_df.values:
            if row[2] > 0:
                adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = 1 # 1-np.exp(-(row[2]/max*50))
            else:
                continue
        return adj_mx

    def _get_sa_adjacency_matrix(self):
        relfile = pd.read_csv(self.sa_rel_path)
        distance_df = relfile[['origin_id', 'destination_id', 'link_weight']]
        adj_mx = np.zeros((len(self.geo_ids), len(self.geo_ids)), dtype=np.float32)
        for row in distance_df.values:
            if row[2] > 0:
                adj_mx[self.geo_to_ind[row[0]], self.geo_to_ind[row[1]]] = 1.0
            else:
                continue
        return adj_mx

    def _load_dyna(self, dyna_path):
        dynafile = pd.read_csv(dyna_path)
        self.timesolts = list(dynafile['time'][:int(dynafile.shape[0] / len(self.geo_ids))])
        self.idx_of_timesolts = dict()
        self.timesolts = list(map(lambda x: x.replace('T', ' ').replace('Z', ''), self.timesolts))
        self.timesolts = np.array(self.timesolts, dtype='datetime64[ns]')
        for idx, _ts in enumerate(self.timesolts):
            self.idx_of_timesolts[_ts] = idx
        df = pd.DataFrame(dynafile["values"])
        len_time = len(self.timesolts)
        data = []
        for i in range(0, df.shape[0], len_time):
            data.append(df[i:i + len_time].values)
        data = np.array(data, dtype=np.float)  # (len(self.geo_ids), len_time, feature_dim)
        data = data.swapaxes(0, 1)  # (len_time, len(self.geo_ids), feature_dim)
        return data

    def _generate_input_data(self, df):
        """
        根据全局参数`input_window`和`output_window`切分输入，产生模型需要的张量输入，
        即使用过去`input_window`长度的时间序列去预测未来`output_window`长度的时间序列

        Args:
            df(np.ndarray): 数据数组，shape: (len_time, ..., feature_dim)

        Returns:
            tuple: tuple contains:
                x(np.ndarray): 模型输入数据，(epoch_size, input_length, ..., feature_dim) \n
                y(np.ndarray): 模型输出数据，(epoch_size, output_length, ..., feature_dim)
        """
        num_samples = df.shape[0]
        # 预测用的过去时间窗口长度 取决于self.input_window
        x_offsets = np.sort(np.concatenate((np.arange(-self.input_window + 1, 1, 1),)))
        # 未来时间窗口长度 取决于self.output_window
        y_offsets = np.sort(np.array(self.output_channel))

        x, y = [], []
        min_t = abs(min(x_offsets))
        max_t = abs(num_samples - self.output_channel[-1]-1)
        for t in range(min_t, max_t):
            x_t = df[t + x_offsets, ...]
            y_t = df[t + y_offsets, ...]
            x.append(x_t)
            y.append(y_t)
        x = np.stack(x, axis=0)
        y = np.stack(y, axis=0)
        return x, y

    def _split_train_val_test(self, x, y, cache_file_name):
        """
        划分训练集、测试集、验证集，并缓存数据集

        Args:
            x(np.ndarray): 输入数据 (num_samples, input_length, ..., feature_dim)
            y(np.ndarray): 输出数据 (num_samples, input_length, ..., feature_dim)

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """
        test_rate = 1 - self.train_rate - self.eval_rate
        num_samples = x.shape[0]
        num_test = round(num_samples * test_rate)
        num_train = round(num_samples * self.train_rate)
        num_val = num_samples - num_test - num_train

        # train
        x_train, y_train = x[:num_train].astype(np.float32), y[:num_train].astype(np.float32)
        # val
        x_val, y_val = x[num_train: num_train + num_val].astype(np.float32), y[num_train: num_train + num_val].astype(np.float32)
        # test
        x_test, y_test = x[-num_test:].astype(np.float32), y[-num_test:].astype(np.float32)

        np.savez_compressed(
            cache_file_name,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            x_val=x_val,
            y_val=y_val,
        )

        return x_train, y_train, x_val, y_val, x_test, y_test

    def _load_cache_train_val_test(self, cache_file_name):
        """
        加载之前缓存好的训练集、测试集、验证集

        Returns:
            tuple: tuple contains:
                x_train: (num_samples, input_length, ..., feature_dim) \n
                y_train: (num_samples, input_length, ..., feature_dim) \n
                x_val: (num_samples, input_length, ..., feature_dim) \n
                y_val: (num_samples, input_length, ..., feature_dim) \n
                x_test: (num_samples, input_length, ..., feature_dim) \n
                y_test: (num_samples, input_length, ..., feature_dim)
        """

        cut_train = 2000000
        cut_test = [0,-1]

        # cut_test = [54+84, 54+84+24]  # 早高峰
        # cut_test = [54+204, 54+204+24]  # 晚高峰
        # cut_test = [54+168, 54+168+36]  # 平峰


        cat_data = np.load(cache_file_name)
        x_train = cat_data['x_train'][:cut_train].astype(np.float32)
        y_train = cat_data['y_train'][:cut_train].astype(np.float32)
        x_test = cat_data['x_test'][cut_test[0]:cut_test[1]].astype(np.float32)
        y_test = cat_data['y_test'][cut_test[0]:cut_test[1]].astype(np.float32)
        x_val = cat_data['x_val'].astype(np.float32)
        y_val = cat_data['y_val'].astype(np.float32)

        print('Train shape X %s - Train shape Y %s'%(x_train.shape,y_train.shape) )
        print('Val shape X %s - Val shape Y %s'%(x_val.shape,y_val.shape))
        print('Test shape X %s - Test shape Y %s'%(x_test.shape,y_test.shape))

        return x_train, y_train, x_val, y_val, x_test, y_test

    def _scaler_data(self, x_train, y_train, x_val, y_val, x_test, y_test, scaler=None):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        # 加载数据集

        # 数据归一化

        self.feature_dim = x_train.shape[-1]

        if scaler:
            scaler = scaler
        else:
            scaler = self._get_scalar(self.scaler_type, x_train[..., :self.output_dim], y_train[..., :self.output_dim])

        x_train[..., :self.output_dim] = scaler.transform(x_train[..., :self.output_dim])
        y_train[..., :self.output_dim] = scaler.transform(y_train[..., :self.output_dim])
        x_val[..., :self.output_dim] = scaler.transform(x_val[..., :self.output_dim])
        y_val[..., :self.output_dim] = scaler.transform(y_val[..., :self.output_dim])
        x_test[..., :self.output_dim] = scaler.transform(x_test[..., :self.output_dim])
        y_test[..., :self.output_dim] = scaler.transform(y_test[..., :self.output_dim])

        # 把训练集的X和y聚合在一起成为list，测试集验证集同理
        # x_train/y_train: (num_samples, input_length, ..., feature_dim)
        # train_data(list): train_data[i]是一个元组，由x_train[i]和y_train[i]组成
        train_data = list(zip(x_train, y_train))
        eval_data = list(zip(x_val, y_val))
        test_data = list(zip(x_test, y_test))

        return train_data, eval_data, test_data, scaler

    def _get_scalar(self, scaler_type, x_train, y_train):
        """
        根据全局参数`scaler_type`选择数据归一化方法

        Args:
            x_train: 训练数据X
            y_train: 训练数据y

        Returns:
            Scaler: 归一化对象
        """
        if scaler_type == "normal":
            scaler = NormalScaler(maxx=max(x_train.max(), x_train.max()))

        elif scaler_type == "standard":
            scaler = StandardScaler(mean=x_train.mean(), std=x_train.std())

        elif scaler_type == "minmax01":
            scaler = MinMax01Scaler(
                maxx=max(x_train.max(), y_train.max()), minn=min(x_train.min(), y_train.min()))

        elif scaler_type == "minmax11":
            scaler = MinMax11Scaler(
                maxx=max(x_train.max(), y_train.max()), minn=min(x_train.min(), y_train.min()))

        elif scaler_type == "log":
            scaler = LogScaler()

        elif scaler_type == "none":
            scaler = NoneScaler()

        else:
            raise ValueError('Scaler type error!')
        return scaler
