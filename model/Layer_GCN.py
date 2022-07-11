# encoding: utf-8
# author: zhaotianhong
# contact: zhaotianhong2016@email.szu.edu.cn

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def calculate_scaled_laplacian(adj):
    """
    L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    L' = 2L/lambda - I
    Args:
        adj: adj_matrix: weighted adjacency matrix of G.

    Returns:
        np.ndarray: L'
    """
    n = adj.shape[0]
    d = np.sum(adj, axis=1)   # D
    lap = np.diag(d) - adj    # L=D-A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                lap[i, j] /= np.sqrt(d[i] * d[j])
    lap[np.isinf(lap)] = 0
    lap[np.isnan(lap)] = 0
    lam = np.linalg.eigvals(lap).max().real
    L = 2 * lap / lam - np.eye(n)
    return L


def calculate_cheb_poly(lap, ks):
    """
    k-order Chebyshev polynomials : T0(L)~Tk(L)
    T0(L)=I/1 T1(L)=L Tk(L)=2LTk-1(L)-Tk-2(L)

    Args:
        lap: scaled laplacian matrix
        ks: k-order

    Returns:
        np.ndarray: T0(L)~Tk(L)
    """
    n = lap.shape[0]
    lap_list = [np.eye(n), lap[:]]
    for i in range(2, ks):
        lap_list.append(np.matmul(2 * lap, lap_list[-1]) - lap_list[-2])
    if ks == 0:
        raise ValueError('Ks must bigger than 0!')
    if ks == 1:
        return np.asarray(lap_list[0:1])  # 1*n*n
    else:
        return np.asarray(lap_list)       # Ks*n*n





class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)  # filter=(1,1)

    def forward(self, x):  # x: (batch_size, feature_dim(c_in), input_length, num_nodes)
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x  # return: (batch_size, c_out, input_length-1+1, num_nodes-1+1)



class GraphConvolution(nn.Module):
    def __init__(self, config, lk, in_channels=16, out_channels=16): #
        super(GraphConvolution, self).__init__()

        # ks, c_in, c_out, lk, device
        self.Lk = lk
        self.theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels, config.Ks).to(config.device))  # kernel: C_in*C_out*ks
        self.b = nn.Parameter(torch.FloatTensor(1, out_channels, 1, 1).to(config.device))
        self.align = Align(in_channels, out_channels)
        self.reset_parameters()
        # self.attention = Layer_CMA.MultiHeadAttention(in_channels, 6)


    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        # Lk: (Ks, num_nodes, num_nodes)
        # x:   (B, T_in, N_nodes, F_channel)
        # x_c: (batch_size, c_in, input_length, Ks, num_nodes)
        # theta: (c_in, c_out, Ks)
        # x_gc: (batch_size, c_out, input_length, num_nodes)
        B,C,T ,N= x.shape
        x = x.permute(0, 1, 3, 2)  # (batch_size, c_in, input_length, num_nodes)
        x_c = torch.einsum("knm,bitm->bitkn", self.Lk, x)  # delete num_nodes(n)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b  # delete Ks(k) c_in(i)
        x_in = self.align(x)  # (batch_size, c_out, input_length, num_nodes)
        x_out = torch.relu(x_gc + x_in)

        # x_in = x_out.view(B,-1,N)
        # x_att = self.attention(x_in, 'self.training')  # 如果使用96维度应该转置


        x_out = x_out.permute(0, 1, 3, 2)
        return x_out # residual connection



class GraphConvolutionAtten(nn.Module):
    def __init__(self, config, lk, in_channels=16, out_channels=16): #
        super(GraphConvolutionAtten, self).__init__()

        # ks, c_in, c_out, lk, device
        self.Lk = lk
        self.theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels, config.Ks).to(config.device))  # kernel: C_in*C_out*ks
        self.b = nn.Parameter(torch.FloatTensor(1, out_channels, 1, 1).to(config.device))

        self.align = Align(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        # Lk: (Ks, num_nodes, num_nodes)
        # x:   (B, T_in, N_nodes, F_channel)
        # x_c: (batch_size, c_in, input_length, Ks, num_nodes)
        # theta: (c_in, c_out, Ks)
        # x_gc: (batch_size, c_out, input_length, num_nodes)
        x = x.permute(0, 1, 3, 2)  # (batch_size, c_in, input_length, num_nodes)
        x_c = torch.einsum("knm,bitm->bitkn", self.Lk, x)  # delete num_nodes(n)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b  # delete Ks(k) c_in(i)
        x_in = self.align(x)  # (batch_size, c_out, input_length, num_nodes)
        x_out = torch.relu(x_gc + x_in)

        # self.attention()
        x_out = x_out.permute(0, 1, 3, 2)


        return x_out # residual connection
