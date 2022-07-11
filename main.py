# encoding: utf-8
# author: zhaotianhong
# contact: zhaotianhong2016@email.szu.edu.cn

import os
import torch
from utils import dataloader, executor
import config as Config
from model import model as My_model
import numpy as np
import random
import sys


seed = 20
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


gup_id = sys.argv[1]
train = sys.argv[2]
name = sys.argv[3]


config = Config.my_config(gup_id)
config.model_name = 'H_%s_P_%s_V_%s'%(config.input_window, config.output_channel, name)
executor.init_path(config)  # 初始化保存路径,配置缓存
if train:
    config.training = True
    executor.logging('*** Training ***', config)
else:
    config.training = False
    executor.logging('*** Evaluating ***', config)

reader = dataloader.Read_Data(config)  # 数据阅读器（先读取矩阵和地理数据用于建模）

model = My_model.Fusion_NA2(config)

# model = nn.DataParallel(model, device_ids=[4,7])
model.to(config.device)

log_info = "Modeling completed"
print(executor.get_local_time(), log_info)
executor.logging(log_info, config)
log_info, trainable_num = executor.get_parameter_number(model)
print(executor.get_local_time(), log_info)
executor.logging(log_info, config)
config.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,weight_decay =0.000)  # config.learning_rate
config.scheduler = torch.optim.lr_scheduler.StepLR(config.optimizer, step_size=50, gamma=0.8)
train_dataloader, eval_dataloader, test_dataloader = dataloader.get_dataloader(reader, config)

if config.training:
    executor.train(model, train_dataloader, test_dataloader, config)
    model.load_state_dict(torch.load(os.path.join(config.model_dir, 'epoch_%s.pkl' % ('best'))))
    executor.evaluate(model, test_dataloader, config, show=[])
else:
    model.load_state_dict(torch.load(os.path.join(config.model_dir, 'epoch_%s.pkl'%('best'))))
    executor.evaluate(model, test_dataloader, config, show=[10, 23, 24, 28])  # 10, 23, 24, 28




if __name__ == '__main__':
    print(executor.get_local_time(), "starting")
    gpu_id =0
    name = 'CTMSTG-0704-AUX-NA2-D30-'  #
    train = False
    run(gpu_id, name, train)
