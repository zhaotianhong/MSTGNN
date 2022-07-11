# encoding: utf-8
# author: zhaotianhong
# contact: zhaotianhong2016@email.szu.edu.cn


import argparse
import torch



def my_config(gpu_id):
    parser = argparse.ArgumentParser()

    # 数据集配置
    output_channels = [1,3,6]

    parser.add_argument('--model_name', type=str, default=None, help='模型名称')
    parser.add_argument('--input_window', type=int, default=12, help='输入历史观测步长')
    parser.add_argument('--output_window', type=int, default=len(output_channels), help='最大预测时间步长')
    parser.add_argument('--output_channel', type=int, default=output_channels, help='最大预测时间步长') #
    parser.add_argument('--output_dim', type=int, default=1, help='输出维度一个bus的通道')  #

    parser.add_argument('--train_rate', type=float, default=0.7, help='训练比例')
    parser.add_argument('--eval_rate', type=float, default=0.1, help='验证比例')
    parser.add_argument('--bus_path', type=str, default='dataset/raw/SZ_BUS.dyna', help='原始公交路径')
    parser.add_argument('--metro_path', type=str, default='dataset/raw/SZ_METRO.dyna', help='原始地铁路径')
    parser.add_argument('--taxi_path', type=str, default='dataset/raw/SZ_TAXI.dyna', help='原始出租车路径')
    parser.add_argument('--od_rel_path', type=str, default='dataset/raw/OD.rel', help='OD图路径')
    parser.add_argument('--od_bus_rel_path', type=str, default='dataset/raw/OD_BUS.rel', help='bus OD图路径')
    parser.add_argument('--od_metro_rel_path', type=str, default='dataset/raw/OD_METRO.rel', help='metro OD图路径')
    parser.add_argument('--od_taxi_rel_path', type=str, default='dataset/raw/OD_TAXI.rel', help='taxi OD图路径')

    parser.add_argument('--sa_rel_path', type=str, default='dataset/raw/SA.rel', help='空间邻路径')
    parser.add_argument('--geo_path', type=str, default='dataset/raw/SZ_SLU.geo', help='预测区域路径')
    parser.add_argument('--scaler',  default=None, help='标准化参数')
    parser.add_argument('--model_dir', default=None, help='模型路径')
    parser.add_argument('--logging_path', default=None, help='日志文件')
    parser.add_argument('--result_path', default=None, help='结果保存文件')
    parser.add_argument('--bus_dim', type=int, default=1, help='公交数据维度')

    # 模型配置
    device = torch.device("cuda:%s"%(gpu_id)) if torch.cuda.is_available() else torch.device("cpu")
    parser.add_argument('--device', default=device, help='是否用GPU')
    parser.add_argument('--in_channels', type=int, default=1, help='模型名称')
    parser.add_argument('--num_of_vertices', type=int, default=1386, help='模型名称')
    parser.add_argument('--num_of_timesteps', type=int, default=12, help='模型名称')
    parser.add_argument('--nb_chev_filter', type=int, default=64, help='通道？')
    parser.add_argument('--Ks', type=int, default=3, help='空间卷积核')
    parser.add_argument('--Kt', type=int, default=3, help='时间卷积核')

    # 训练配置
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--gamma', type=float, default=0.98, help='学习率下降')

    parser.add_argument('--epochs', type=int, default=300, help='训练次数')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout')

    # 其他设置
    parser.add_argument('--metrics', default= ['MAE', 'MSE', 'RMSE', 'MAPE', 'masked_MAE',
                        'masked_MSE', 'masked_RMSE', 'masked_MAPE', 'R2', 'EVAR'], help='评价标准')
    parser.add_argument('--result_name', default= "Test", help='结果保存文件名')

    args = parser.parse_args()
    return args
