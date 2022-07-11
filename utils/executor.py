import pandas as pd
from torch.autograd import Variable
import torch
import numpy as np
import datetime
from utils import evaluator, plot_result
from scipy.sparse.linalg import eigs
import os
from torch.utils.tensorboard import SummaryWriter


def train(model, train_dataloader, eval_dataloader, config):
    '''模型训练'''
    print(get_local_time(), "Train starting")
    best_epoch = 10000000
    stay = 0
    for epoch in range(config.epochs):
        if stay > 20:
            break
        loss_train = _train_epoch(model, train_dataloader, config)
        loss_val = _valid_epoch(model, eval_dataloader, config)

        if loss_val < best_epoch:
            stay = 0
            best_epoch = loss_val
            torch.save(model.state_dict(), os.path.join(config.model_dir, 'epoch_%s.pkl'%('best')))
        else:
            stay += 1

        loss_info = 'Epochs: %s/%s  Train loss: %.6f  Val loss: %.6f  Learning rate:%.6f'%\
                    (epoch, config.epochs, loss_train, loss_val, config.optimizer.state_dict()['param_groups'][0]['lr'])
        logging(loss_info, config)
        config.writer.add_scalar("Train loss", loss_train, epoch)
        config.writer.add_scalar("Val loss", loss_val, epoch)
        config.writer.add_scalar("Learning rate", config.optimizer.state_dict()['param_groups'][0]['lr'], epoch)

        print(get_local_time(), loss_info)
        config.scheduler.step()
        # print(get_local_time(), 'Model saved at %s'%(os.path.join(config.model_dir, 'epoch_%s.pkl'%(epoch))))
    print(get_local_time(), "Train finished !")


def evaluate(model, test_dataloader, config, show):
    """
    use model to test data
    Args:
        test_dataloader(torch.Dataloader): Dataloader
    """
    with torch.no_grad():
        model.eval()
        y_truths = []
        y_preds = []
        losses = []
        for batch in test_dataloader:
            bus = Variable(batch[0]).to(config.device)
            metro = Variable(batch[1]).to(config.device)
            taxi = Variable(batch[2]).to(config.device)

            y_bus_true = Variable(batch[3]).to(config.device)
            y_metro_true = Variable(batch[4]).to(config.device)
            y_taxi_true = Variable(batch[5]).to(config.device)

            y_true = torch.cat((y_bus_true, y_metro_true, y_taxi_true), dim=3)

            output, loss = model.predict(bus, metro, taxi)
            # if len(output) == 2:
            #     output = output[0]
            loss = model.calculate_loss(bus, metro, taxi, y_bus_true, y_metro_true, y_taxi_true)
            losses.append(loss.item())

            y_true = config.scaler.inverse_transform(y_true)
            y_pred = config.scaler.inverse_transform(output)

            y_truths.append(y_true.cpu().numpy())
            y_preds.append(y_pred.cpu().numpy())
        B, T, N, C = y_preds[0].shape
        y_preds = np.concatenate(y_preds, axis=0)
        y_truths = np.concatenate(y_truths, axis=0)
        for i in range(C):
            y_p = y_preds[:,:,:,i]
            y_t = y_truths[:,:,:,i]
            y_p = y_p.reshape(-1,T,N)
            y_t = y_t.reshape(-1,T,N)  # concatenate on batch

            result = evaluator.collect(torch.tensor(y_p), torch.tensor(y_t),config)
            test_result = evaluator.save_result(result, config, '__T__'+str(i))  # 保存评估结果
            if False: # 输出具体预测结果
                save_raw_result(y_p, y_t, config.result_path.replace('.csv',''), '__T__'+str(i))

        if len(show) > 0:
            plot_result.show_compare(y_preds[:,:,:,0], y_truths[:,:,:,0], time_step=1, blocks=show)
        mean_loss = np.mean(losses)
        print('loss:', mean_loss)
        print(test_result)



def _train_epoch(model, train_dataloader, config):
    '''训练一个epoch'''
    model.train()
    losses = []
    for batch in train_dataloader:
        config.optimizer.zero_grad()
        bus = Variable(batch[0]).to(config.device)
        metro = Variable(batch[1]).to(config.device)
        taxi = Variable(batch[2]).to(config.device)

        y_bus_true = Variable(batch[3]).to(config.device)
        y_metro_true = Variable(batch[4]).to(config.device)
        y_taxi_true = Variable(batch[5]).to(config.device)

        loss = model.calculate_loss(bus, metro, taxi, y_bus_true, y_metro_true, y_taxi_true)
        losses.append(loss.item())
        loss.backward()
        config.optimizer.step()
    mean_loss = np.mean(losses)
    return mean_loss



def _valid_epoch(model, eval_dataloader, config):
    """
    完成模型一个轮次的评估
    Returns:
        float: 评估数据的平均损失值
    """
    with torch.no_grad():
        model.eval()
        losses = []
        for batch in eval_dataloader:
            bus = Variable(batch[0]).to(config.device)
            metro = Variable(batch[1]).to(config.device)
            taxi = Variable(batch[2]).to(config.device)
            y_bus_true = Variable(batch[3]).to(config.device)
            y_metro_true = Variable(batch[4]).to(config.device)
            y_taxi_true = Variable(batch[5]).to(config.device)
            loss = model.calculate_loss(bus, metro, taxi, y_bus_true, y_metro_true, y_taxi_true)
            losses.append(loss.item())
        mean_loss = np.mean(losses)
        return mean_loss



def get_local_time():
    """
    获取时间
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y %H:%M:%S')
    return cur


def logging(log_info, config):
    '''记录日志信息'''
    with open(config.logging_path, 'a+') as fw:
        fw.write(get_local_time()+":"+ log_info + '\n')


def init_path(config):
    '''初始化模型，结果，日志文件夹'''

    config.model_dir = 'save/model/%s'%(config.model_name)
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)
    time_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    config.logging_path = 'save/logging/%s_%s.txt' % (config.model_name, time_name)
    config.result_path = 'save/result/%s_%s.csv' % (config.model_name, time_name)
    config.writer = SummaryWriter('save/logging/tensorboard/%s_%s' % (config.model_name, time_name))


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    return 'Total parameter: %s Trainable parameter: %s'%(total_num, trainable_num), trainable_num


def save_raw_result(predict, ture, path_base, info):
    pre_v = np.transpose(predict, (1,0,2))
    ture_v = np.transpose(ture, (1,0,2))

    for t in range(len(pre_v)):
        path = path_base + info + '__timestep_%s_predict.csv'%(t)
        data_df = pd.DataFrame(pre_v[t])
        data_df.to_csv(path)

    for t in range(len(ture_v)):
        path = path_base + info + '__timestep_%s_ture.csv'%(t)
        data_df = pd.DataFrame(ture_v[t])
        data_df.to_csv(path)