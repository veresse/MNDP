# 导入PyTorch相关库
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameter import hyperparameter

# 导入超参数类
hp = hyperparameter()


# 定义图注意力网络模型
class GAT(nn.Module):

    def __init__(self, dropout=0.2):
        super(GAT, self).__init__()

        # 初始化隐藏层维度、原子特征维度、dropout率
        self.hid_dim = hp.hid_dim
        self.atom_dim = hp.conv
        self.dropout = dropout

        # 定义dropout层
        self.do = nn.Dropout(dropout)

        # 定义图注意力机制层
        self.W_gnn = nn.ModuleList([
            nn.Linear(self.atom_dim, self.atom_dim),
            nn.Linear(self.atom_dim, self.atom_dim),
            nn.Linear(self.atom_dim, self.atom_dim)
        ])

        # 定义图注意力机制转换层
        self.W_gnn_trans = nn.Linear(self.atom_dim, self.hid_dim)

        # 定义化合物注意力参数列表
        self.compound_attn = nn.ParameterList([
            nn.Parameter(torch.randn(size=(2 * self.atom_dim, 1))) for _ in range(len(self.W_gnn))
        ])

    def forward(self, input, adj):
        # 前向传播过程
        for i in range(len(self.W_gnn)):
            h = torch.relu(self.W_gnn[i](input))

            size = h.size()[0]
            N = h.size()[1]

            # 计算注意力权重
            a_input = torch.cat([h.repeat(1, 1, N).view(size, N * N, -1), h.repeat(1, N, 1)], dim=2).view(size, N, -1,
                                                                                                          2 * self.atom_dim)
            e = F.leaky_relu(torch.matmul(a_input, self.compound_attn[i]).squeeze(3))

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=2)
            attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime = torch.matmul(attention, h)

            input = input + h_prime

        # 原子特征转换
        input = self.do(F.relu(self.W_gnn_trans(input)))
        return input
