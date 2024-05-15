# 导入PyTorch相关库
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义自注意力机制模块
class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        # 初始化隐藏层维度、注意力头数和dropout率
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        # 确保隐藏层维度可以被注意力头数整除
        assert hid_dim % n_heads == 0

        # 定义权重矩阵
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        # 定义dropout层
        self.do = nn.Dropout(dropout)

        # 设置缩放因子
        if torch.cuda.is_available():
            self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).cuda()
        else:
            self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))

    def forward(self, query, key, value, mask=None):
        # 确定批大小
        if len(query.shape) > len(key.shape):
            bsz = query.shape[0]
        else:
            bsz = key.shape[0]

        # 计算Q、K、V
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # 计算能量
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        Q, K = Q.cpu(), K.cpu()
        del Q, K

        # 对能量进行mask
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # 计算注意力得分
        return self.fc(torch.matmul(self.do(F.softmax(energy, dim=-1)), V).permute(0, 2, 1, 3).contiguous().view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads)))

# 定义位置前馈神经网络模块
class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        # 初始化隐藏层维度、前馈神经网络维度和dropout率
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        # 定义全连接层
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)

        # 定义dropout层
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播过程
        return self.fc_2(self.do(F.relu(self.fc_1(x.permute(0, 2, 1))))).permute(0, 2, 1)

# 定义解码器层模块
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()

        # 初始化层归一化、自注意力机制、编码器-解码器注意力机制、位置前馈神经网络和dropout率
        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout)
        self.ea = SelfAttention(hid_dim, n_heads, dropout)
        self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # 解码器层前向传播过程，在这里进行了SMILES和氨基酸的交互
        trg1 = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg1 = self.ln(trg1 + self.do(self.ea(trg1, src, src, src_mask)))
        trg1 = self.ln(trg1 + self.do(self.pf(trg1)))
        src1 = self.ln(src + self.do(self.sa(src, src, src, src_mask)))
        src1 = self.ln(src1 + self.do(self.ea(src1, trg, trg, trg_mask)))
        src1 = self.ln(src1 + self.do(self.pf(src1)))
        # trg,src= trg.cpu(),src.cpu()
        del trg, src, trg_mask, src_mask
        return trg1, src1

# 定义解码器模块
class Decoder(nn.Module):
    def __init__(self, hid_dim, n_layers, n_heads, pf_dim, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.dropout = dropout
        # 定义解码器层列表
        self.layers = nn.ModuleList(
            [DecoderLayer(hid_dim, n_heads, pf_dim, dropout)
             for _ in range(n_layers)])

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        src = src.to(torch.float32)

        # 逐层进行解码器操作
        for layer in self.layers:
            trg, src = layer(trg, src, trg_mask, src_mask)
        del trg_mask, src_mask
        return trg, src
