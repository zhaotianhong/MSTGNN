# encoding: utf-8
# author: zhaotianhong
# contact: zhaotianhong2016@email.szu.edu.cn



import torch
import torch.nn as nn
from model import Layer_TGCN, Layer_Attention,Layer_GCN


class AuxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)

    def forward(self, bus, metro, taxi):
        '''
        三个方差尽量小
        :param bus:  B, N, C
        :param metro:
        :param taxi:
        :return:
        '''

        sum_bus = torch.mean(bus, dim=-1)  # 通道方向求和
        sum_metro = torch.mean(metro, dim=-1)  # 通道方向求和
        sum_taxi = torch.mean(taxi, dim=-1)  # 通道方向求和

        bm_s = 1 - self.cos_sim(sum_bus, sum_metro)
        bt_s = 1 - self.cos_sim(sum_bus, sum_taxi)

        sum_s = torch.mean(bm_s) + torch.mean(bt_s)

        return sum_s



class MSTGNN(nn.Module):
    def __init__(self, config):
        super(MSTGNN, self).__init__()

        emd_dim = 16
        hide = 32
        out = 128
        self.out_dim = 1
        device = config.device

        # 计算卷积核
        laplacian_mx = Layer_GCN.calculate_scaled_laplacian(config.sa_graph)
        sa_Lk = torch.FloatTensor(Layer_GCN.calculate_cheb_poly(laplacian_mx, config.Ks)).to(config.device)

        laplacian_mx = Layer_GCN.calculate_scaled_laplacian(config.od_bus_graph)
        od_bus_Lk = torch.FloatTensor(Layer_GCN.calculate_cheb_poly(laplacian_mx, config.Ks)).to(config.device)

        laplacian_mx = Layer_GCN.calculate_scaled_laplacian(config.od_metro_graph)
        od_metro_Lk = torch.FloatTensor(Layer_GCN.calculate_cheb_poly(laplacian_mx, config.Ks)).to(config.device)

        laplacian_mx = Layer_GCN.calculate_scaled_laplacian(config.od_taxi_graph)
        od_taxi_Lk = torch.FloatTensor(Layer_GCN.calculate_cheb_poly(laplacian_mx, config.Ks)).to(config.device)

        self.output_window = config.output_window
        self.training = config.training
        self.loss = nn.MSELoss()

        self.bus_sa = Layer_TGCN.TGCN_M(config, sa_Lk, emd_dim, hide, out)
        self.metro_sa = Layer_TGCN.TGCN_M(config, sa_Lk, emd_dim, hide, out)
        self.taxi_sa = Layer_TGCN.TGCN_M(config, sa_Lk, emd_dim, hide, out)

        self.bus_od = Layer_TGCN.TGCN_M(config, od_bus_Lk, emd_dim, hide, out)
        self.metro_od = Layer_TGCN.TGCN_M(config, od_metro_Lk, emd_dim, hide, out)
        self.taxi_od = Layer_TGCN.TGCN_M(config, od_taxi_Lk, emd_dim, hide, out)

        self.attention_global = Layer_Attention.SumAttentionDot2(out)  # MultiHeadAttention  config.num_of_vertices
        self.attention_local = Layer_Attention.SumAttentionDot2(out)  # Multi_Head_Attention

        self.weight2 = nn.init.xavier_uniform_(
            nn.Parameter(torch.FloatTensor(1, config.num_of_vertices, out, ).to(device)))
        self.weight1 = nn.init.xavier_uniform_(
            nn.Parameter(torch.FloatTensor(1, config.num_of_vertices, out, ).to(device)))

        self.out_fc = nn.Linear(out, config.output_window * self.out_dim)
        # self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.aux_loss = AuxLoss()

    def forward(self, bus, metro, taxi):
        """
        Args:、、
            bus: (B, T_in, N_nodes, F_channel)
            metro: (B, T_in, N_nodes, F_channel)
            taxi: (B, T_in, N_nodes, F_channel)
        Returns:
            torch.tensor: out: (B, T_in, N_nodes, F_channel) 和输入保持一致
        """

        bus_sa_x = self.bus_sa(bus)
        metro_sa_x = self.metro_sa(metro)
        taxi_sa_x = self.taxi_sa(taxi)

        bus_od_x = self.bus_od(bus)
        metro_od_x = self.metro_od(metro)
        taxi_od_x = self.taxi_od(taxi)  # B, C, N

        B, N, C = bus_sa_x.shape

        local_x_att = self.attention_local(bus_sa_x, metro_sa_x, taxi_sa_x)  # 如果使用96维度应该转置 B, C, N
        global_x_att = self.attention_global(bus_od_x, metro_od_x, taxi_od_x)

        x = local_x_att * self.weight1 + global_x_att * self.weight2


        loss_aux = self.aux_loss(bus_sa_x, metro_sa_x, taxi_sa_x) + self.aux_loss(bus_od_x, metro_od_x, taxi_od_x)

        out_x = self.out_fc(x).view(B, N, self.output_window, self.out_dim)
        out_x1 = out_x.permute(0, 2, 1, 3)

        return out_x1, loss_aux  # [B, out_windows, nodes, fea]

    def calculate_loss(self, bus, metro, taxi, y_bus, y_metro, y_taxi):
        y_predicted, loss_aux = self.predict(bus, metro, taxi)
        if self.out_dim == 1:
            y_true = y_bus
        else:
            y_true = torch.cat((y_bus, y_metro, y_taxi), dim=3)
        loss_mae = self.loss(y_predicted, y_true)
        loss = loss_mae + loss_aux
        # print('loss, mae, cos  ', loss, loss_mae, loss_aux)
        return loss

    def predict(self, bus, metro, taxi):
        # 多步预测
        y_preds, loss_aux = self.forward(bus, metro, taxi)
        return y_preds, loss_aux

