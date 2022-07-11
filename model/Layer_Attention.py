# encoding: utf-8
# author: zhaotianhong
# contact: zhaotianhong2016@email.szu.edu.cn

import torch
from torch import nn
import pandas as pd
import torch.nn.functional as F
import math
import datetime



class ScaleDotProductAttention3(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention3, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

class MultiHeadAttention3(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention3, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention3()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, x):
        # 1. dot product with weight matrices
        mask = None
        q, k, v = x, x, x
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        # attention = self.concat(attention)
        attention = torch.mean(attention, dim=1)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out, attention

    def split(self, tensor):
        """
        split tensor by number of head
        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)
        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, x):
        # 1. dot product with weight matrices
        mask = None
        q, k, v = x, x, x
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        # attention = self.concat(attention)
        attention = torch.mean(attention, dim=1)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out, attention

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor





class SumAttentionDot(nn.Module):
    '''
    多模式注意力机制
    '''
    def __init__(self, dim_out, num_view, num_of_vertices):
        super(SumAttentionDot, self).__init__()
        # in_channels 三种交通模式融合后的通道大小

        att_dim = dim_out*num_view

        self.softmax = nn.Softmax(dim=-1)
        self.relation = nn.Conv1d(att_dim, att_dim, 1, stride=1, bias=False)
        self.att_channel = nn.Conv1d(num_of_vertices, att_dim, 1, stride=1)

        self.outlayer = nn.Linear(att_dim, dim_out)
        self.dropout = nn.Dropout(0.3)
        self.relu = torch.nn.ReLU()


    def forward(self,bmt):
        if len(bmt) > 1:
            x_bmt = torch.cat(bmt, dim=-1)  # 如果使用96维度应该转置 B N C
        else:
            x_bmt = bmt[0]

        x_c = self.att_channel(x_bmt)
        x_r = self.relation(x_c)
        a_score = self.softmax(x_r)
        x = x_bmt @ a_score

        x = self.relu(x)
        x = self.dropout(x)
        x_out = self.outlayer(x)

        return x_out


class SumAttentionDot2(nn.Module):
    '''
    多模式注意力机制
    '''
    def __init__(self, dim_out):
        super(SumAttentionDot2, self).__init__()
        # in_channels 三种交通模式融合后的通道大小

        att_dim = dim_out*3
        num_of_vertices = 1386

        # self.att_s = MultiHeadAttention(att_dim, n_head)
        # self.att_c = MultiHeadAttention(num_of_vertices, n_head)

        self.softmax = nn.Softmax(dim=-1)
        self.relation = nn.Conv1d(att_dim, att_dim, 1, stride=1,bias=False)
        self.att_channel = nn.Conv1d(num_of_vertices, att_dim, 1, stride=1)

        self.outlayer = nn.Linear(att_dim, dim_out)
        self.dropout = nn.Dropout(0.3)
        self.relu = torch.nn.ReLU()



    def forward(self,bus, metro, taxi):
        '''
        c*c * (c*n * n*c)
        :param bus:
        :param metro:
        :param taxi:
        :return:
        '''

        all_x = torch.cat((bus, metro, taxi), dim=-1)  # 如果使用96维度应该转置 B N C

        x_c = self.att_channel(all_x)
        x_r = self.relation(x_c)
        a_score = self.softmax(x_r)
        x = all_x @ a_score

        # x_s = all_x + x

        x = self.relu(x)
        x = self.dropout(x)
        x_out = self.outlayer(x)

        # time_str = str(datetime.datetime.now())
        # mean_v = torch.mean(a_score, dim=0)
        # attention_v = mean_v.cpu().numpy()
        # data_df = pd.DataFrame(attention_v)
        # data_df.to_csv('attention_o_%s.csv'%(time_str))

        return x_out


class SumAttentionDot3(nn.Module):
    '''
    多模式注意力机制
    '''
    def __init__(self, dim_out):
        super(SumAttentionDot3, self).__init__()
        # in_channels 三种交通模式融合后的通道大小

        att_dim = dim_out*3
        num_of_vertices = 1386

        self.softmax = nn.Softmax(dim=-1)
        self.att_channel = nn.Conv1d(num_of_vertices, att_dim, 1, stride=1)

        self.outlayer = nn.Linear(att_dim, dim_out)
        self.dropout = nn.Dropout(0.3)
        self.relu = torch.nn.ReLU()



    def forward(self,bus, metro, taxi):

        all_x = torch.cat((bus, metro, taxi), dim=-1)  # 如果使用96维度应该转置 B N C

        a = self.att_channel(all_x)
        a_score = self.softmax(a)
        x = all_x @ a_score

        x_s = all_x + x
        x = self.relu(x_s)

        x_c = self.dropout(x)
        x_out = self.outlayer(x_c)

        # time_str = str(datetime.datetime.now())
        # mean_v = torch.mean(a_score, dim=0)
        # attention_v = mean_v.cpu().numpy()
        # data_df = pd.DataFrame(attention_v)
        # data_df.to_csv('attention_n_%s.csv'%(time_str))

        return x_out


class SumAttentionDot33(nn.Module):
    '''
    多模式注意力机制
    '''
    def __init__(self, dim_out):
        super(SumAttentionDot33, self).__init__()
        # in_channels 三种交通模式融合后的通道大小

        att_dim = dim_out*3
        num_of_vertices = 1386

        self.softmax_c = nn.Softmax(dim=-1)
        self.softmax_s = nn.Softmax(dim=-1)

        self.spatial_relation = nn.Conv1d(num_of_vertices, num_of_vertices, 1, stride=1, bias=False)
        self.spatial_c = nn.Linear(att_dim, num_of_vertices)

        self.channel_relation = nn.Conv1d(att_dim, att_dim, 1, stride=1, bias=False)
        self.channel_c = nn.Conv1d(num_of_vertices, att_dim, 1, stride=1)

        self.active_a = nn.Sigmoid()

        self.outlayer = nn.Linear(att_dim, dim_out)
        self.dropout = nn.Dropout(0.3)
        self.relu = torch.nn.ReLU()



    def forward(self,bus, metro, taxi):

        all_x = torch.cat((bus, metro, taxi), dim=-1)  # 如果使用96维度应该转置 B N C

        x_s = self.active_a(self.spatial_c(all_x))
        a_s = self.softmax_s(self.spatial_relation(x_s))
        x_s_a = a_s @ all_x

        x_c = self.active_a(self.channel_c(all_x))
        a_c = self.softmax_c(self.channel_relation(x_c))
        x_c_a = all_x @ a_c

        x = self.relu(x_s_a + x_c_a + all_x)

        x = self.dropout(x)
        x_out = self.outlayer(x)

        return x_out


class SumAttentionDot5(nn.Module):
    '''
    多模式注意力机制
    '''
    def __init__(self, dim_out):
        super(SumAttentionDot5, self).__init__()
        # in_channels 三种交通模式融合后的通道大小

        att_dim = dim_out*3
        num_of_vertices = 1386

        self.softmax_c = nn.Softmax(dim=-1)

        self.spatial = nn.Conv1d(num_of_vertices, num_of_vertices, 1, stride=1, bias=False)
        self.channel = nn.Linear(att_dim, att_dim, bias=False)

        self.active_a = nn.Sigmoid()

        self.outlayer = nn.Linear(att_dim, dim_out)
        self.dropout = nn.Dropout(0.3)
        self.relu = torch.nn.ReLU()



    def forward(self,bus, metro, taxi):

        all_x = torch.cat((bus, metro, taxi), dim=-1)  # 如果使用96维度应该转置 B N C

        x_s = self.spatial(all_x)
        x_c = self.channel(x_s)
        a_s = self.softmax_c(x_c)

        x = all_x * a_s

        x = self.relu(all_x + x)
        x = self.dropout(x)
        x_out = self.outlayer(x)

        return x_out



class SumAttentionDot4(nn.Module):
    '''
    多模式注意力机制
    '''
    def __init__(self, dim_out):
        super(SumAttentionDot4, self).__init__()
        # in_channels 三种交通模式融合后的通道大小

        att_dim = dim_out*3
        num_of_vertices = 1386

        # self.att_s = MultiHeadAttention(att_dim, n_head)
        # self.att_c = MultiHeadAttention(num_of_vertices, n_head)

        self.softmax = nn.Softmax(dim=-1)
        self.relation = nn.Conv1d(att_dim, att_dim, 1, stride=1,bias=False)
        self.att_channel = nn.Conv1d(num_of_vertices, att_dim, 1, stride=1)
        self.active_a = nn.Sigmoid()

        self.outlayer = nn.Linear(att_dim, dim_out)
        self.dropout = nn.Dropout(0.3)
        self.relu = torch.nn.ReLU()



    def forward(self,bus, metro, taxi):
        '''
        c*c * (c*n * n*c)
        :param bus:
        :param metro:
        :param taxi:
        :return:
        '''

        all_x = torch.cat((bus, metro, taxi), dim=-1)  # 如果使用96维度应该转置 B N C

        x_c = self.active_a(self.att_channel(all_x))
        x_r = self.relation(x_c)

        a_score = self.softmax(x_r)
        x = all_x @ a_score

        x = self.relu(x)
        x = self.dropout(x)
        x_out = self.outlayer(x)


        return x_out


# class SumAttentionDot3(nn.Module):
#     '''
#     多模式注意力机制
#     '''
#     def __init__(self, dim_out):
#         super(SumAttentionDot3, self).__init__()
#         # in_channels 三种交通模式融合后的通道大小
#         self.softmax = nn.Softmax(dim=-1)
#         self.relu = torch.nn.ReLU()
#         n_head = 6
#         att_dim = dim_out*3
#         num_of_vertices = 1386
#
#         self.att_layer = nn.Conv1d(num_of_vertices, num_of_vertices, 1, stride=1)
#         self.softmax = nn.Softmax(dim=-1)
#
#         # self.att = MultiHeadAttention3(att_dim, n_head)
#         self.outlayer = nn.Linear(att_dim, dim_out)
#
#
#     def forward(self,bus, metro, taxi):
#
#         all_x = torch.cat((bus, metro, taxi), dim=-1)  # 如果使用96维度应该转置
#
#         x_w = self.att_layer(all_x)
#         a = self.softmax(x_w)
#         x_a = all_x * a
#         x_act = self.relu(x_a)
#         x_out = self.outlayer(x_act)
#
#         #
#         # mean_v = torch.mean(a, dim=0)
#         # attention_v = mean_v.cpu().numpy()
#         # data_df = pd.DataFrame(attention_v)
#         # data_df.to_csv('attention_noon.csv')
#
#         return x_out


#
# class SumAttentionDot4(nn.Module):
#     '''
#     多模式注意力机制
#     '''
#
#     def __init__(self, dim_out):
#         super(SumAttentionDot4, self).__init__()
#         # in_channels 三种交通模式融合后的通道大小
#
#         n_head = 6
#         att_dim = dim_out * 3
#         num_of_vertices = 1386
#
#         # self.att_s = MultiHeadAttention(att_dim, n_head)
#         self.att_c = MultiHeadAttention(num_of_vertices, n_head)
#         self.outlayer = nn.Linear(att_dim, dim_out)
#         self.dropout = nn.Dropout(0.3)
#         self.relu = torch.nn.ReLU()
#
#
#     def forward(self, bus, metro, taxi):
#         all_x = torch.cat((bus, metro, taxi), dim=-1)  # 如果使用96维度应该转置
#         #
#         # x, a_s = self.att_s(all_x)
#         # x_s = all_x + x
#
#         x, a_c = self.att_c(all_x.permute(0, 2, 1))
#         x_c = all_x.permute(0, 2, 1) + x
#         x_c = self.relu(x_c)
#         x_c = self.dropout(x_c)
#         x_out = self.outlayer(x_c.permute(0, 2, 1))
#
#         return x_out
#
#
# class SumAttentionDot3(nn.Module):
#     '''
#     多模式注意力机制
#     '''
#
#     def __init__(self, dim_out):
#         super(SumAttentionDot3, self).__init__()
#         # in_channels 三种交通模式融合后的通道大小
#
#         n_head = 6
#         att_dim = dim_out * 3
#         num_of_vertices = 1386
#
#         # self.att_s = MultiHeadAttention(att_dim, n_head)
#         self.att_c = MultiHeadAttention(num_of_vertices, n_head)
#         self.outlayer = nn.Linear(att_dim, dim_out)
#         self.dropout = nn.Dropout(0.3)
#         self.relu = torch.nn.ReLU()
#
#
#     def forward(self, bus, metro, taxi):
#         all_x = torch.cat((bus, metro, taxi), dim=-1)  # 如果使用96维度应该转置
#
#         # x, a_s = self.att_s(all_x)
#         # x_s = all_x + x
#
#         x, a_c = self.att_c(all_x.permute(0, 2, 1))
#         x_c = all_x.permute(0, 2, 1) + x
#         x_c = self.relu(x_c)
#         x_c = self.dropout(x_c)
#         x_out = self.outlayer(x_c.permute(0, 2, 1))
#
#         return x_out
#
#
#
