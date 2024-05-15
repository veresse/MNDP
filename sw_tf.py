import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

patch_size = 4
d_model = 32  # Embedding Size
model_dim_C = 32  # 最开始的patch embedding的大小
# max_num_token = 16
window_size = 4  # 带窗transformer的窗大小
# num_head = 4
num_head = 2
merge_size = 2  # 2 * 2的patch组合成1个patch

# 定义多头自注意力机制模块
class MultiHeadSelfAttention(nn.Module):

    def __init__(self, model_dim, num_head):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_head = num_head
        # 用于投影输入特征的线性层，将输入维度映射为3倍的模型维度
        self.proj_linear_layer = nn.Linear(model_dim, 3 * model_dim)  # 线性层
        # 最终的线性层，将输出维度映射回模型维度
        self.final_linear_layer = nn.Linear(model_dim, model_dim)  # 最终线性层

    def forward(self, input, additive_mask=None):
        bs, seqlen, model_dim = input.shape
        num_head = self.num_head
        head_dim = model_dim // num_head

        # 通过投影层分离查询、键和值
        proj_output = self.proj_linear_layer(input)  # 投影层
        q, k, v = proj_output.chunk(3, dim=-1)  # 切分成三部分：查询、键、值

        # 重塑并转置以进行多头操作
        q = q.reshape(bs, seqlen, num_head, head_dim).transpose(1, 2).reshape(bs * num_head, seqlen, head_dim)
        k = k.reshape(bs, seqlen, num_head, head_dim).transpose(1, 2).reshape(bs * num_head, seqlen, head_dim)
        v = v.reshape(bs, seqlen, num_head, head_dim).transpose(1, 2).reshape(bs * num_head, seqlen, head_dim)

        # 计算注意力权重
        attn_prob = F.softmax(torch.bmm(q, k.transpose(-2, -1)) / math.sqrt(head_dim), dim=-1)

        # 输出计算结果
        output = torch.bmm(attn_prob, v)
        output = output.reshape(bs, num_head, seqlen, head_dim).transpose(1, 2)
        output = output.reshape(bs, seqlen, model_dim)
        output = self.final_linear_layer(output)
        return attn_prob, output


# 窗口多头自注意力机制
def window_multi_head_self_attention(patch_embedding, mhsa, window_size=4, num_head=num_head):
    num_patch_in_window = window_size * window_size
    bs, num_patch, patch_depth = patch_embedding.shape
    image_height = image_width = int(math.sqrt(num_patch))

    # 调整补丁嵌入形状以适应窗口操作
    patch_embedding = patch_embedding.transpose(-1, -2)
    patch = patch_embedding.reshape(bs, patch_depth, image_height, image_width)
    window = F.unfold(patch, kernel_size=(window_size, window_size), stride=(window_size, window_size)).transpose(-1,
                                                                                                                  -2)

    bs, num_window, patch_depth_times_num_patch_in_window = window.shape
    window = window.reshape(bs * num_window, patch_depth, num_patch_in_window).transpose(-1, -2)

    # 应用多头自注意力机制
    attn_prob, output = mhsa(window)

    output = output.reshape(bs, num_window, num_patch_in_window, patch_depth)
    return output


# 将窗口输出转换为图像
def window2image(msa_output):
    bs, num_window, num_patch_in_window, patch_depth = msa_output.shape
    window_size = int(math.sqrt(num_patch_in_window))
    image_height = int(math.sqrt(num_window)) * window_size
    image_width = image_height

    # 重塑输出以形成图像
    msa_output = msa_output.reshape(bs, int(math.sqrt(num_window)),
                                    int(math.sqrt(num_window)),
                                    window_size,
                                    window_size,
                                    patch_depth)
    msa_output = msa_output.transpose(2, 3)
    image = msa_output.reshape(bs, image_height * image_width, patch_depth)

    image = image.transpose(-1, -2).reshape(bs, patch_depth, image_height, image_width)

    return image

#移位窗口模块
def shift_window(w_msa_output, window_size, shift_size, generate_mask=False):
    bs, num_window, num_patch_in_window, patch_depth = w_msa_output.shape

    w_msa_output = window2image(w_msa_output)
    bs, patch_depth, image_height, image_width = w_msa_output.shape

    rolled_w_msa_output = torch.roll(w_msa_output, shifts=(shift_size, shift_size), dims=(2, 3))

    shifted_w_msa_input = rolled_w_msa_output.reshape(bs, patch_depth,
                                                      int(math.sqrt(num_window)),
                                                      window_size,
                                                      int(math.sqrt(num_window)),
                                                      window_size
                                                     )

    shifted_w_msa_input = shifted_w_msa_input.transpose(3, 4)
    shifted_w_msa_input = shifted_w_msa_input.reshape(bs, patch_depth, num_window*num_patch_in_window)
    shifted_w_msa_input = shifted_w_msa_input.transpose(-1, -2) # [bs, num_window*num_patch_in_window, patch_depth]
    shifted_window = shifted_w_msa_input.reshape(bs, num_window, num_patch_in_window, patch_depth)

    if generate_mask:
        additive_mask = build_mask_for_shifted_wmsa(bs, image_height, image_width, window_size)
    else:
        additive_mask = None

    return shifted_window, additive_mask



def build_mask_for_shifted_wmsa(batch_size, image_height, image_width, window_size):
    index_matrix = torch.zeros(image_height, image_width)

    for i in range(image_height):
        for j in range(image_width):
            row_times = (i+window_size//2) // window_size
            col_times = (j+window_size//2) // window_size
            index_matrix[i, j] = row_times*(image_height//window_size) + col_times + 1
    rolled_index_matrix = torch.roll(index_matrix, shifts=(-window_size//2, -window_size//2), dims=(0, 1))
    rolled_index_matrix = rolled_index_matrix.unsqueeze(0).unsqueeze(0) #[bs, ch, h, w]

    c = F.unfold(rolled_index_matrix, kernel_size=(window_size, window_size),
                 stride=(window_size, window_size)).transpose(-1, -2)

    c = c.tile(batch_size, 1, 1)

    bs, num_window, num_patch_in_window = c.shape

    c1 = c.unsqueeze(-1)
    c2 = (c1 - c1.transpose(-1, -2)) == 0
    valid_matrix = c2.to(torch.float32)
    additive_mask = (1-valid_matrix)*(-1e-9)

    additive_mask = additive_mask.reshape(bs*num_window, num_patch_in_window, num_patch_in_window)

    return additive_mask


# 对移位的窗口应用多头自注意力机制
def shift_window_multi_head_self_attention(w_msa_output, mhsa, window_size=4, num_head=num_head):
    bs, num_window, num_patch_in_window, patch_depth = w_msa_output.shape

    # 移位窗口以便适应下一层的输入
    shifted_w_msa_input, additive_mask = shift_window(w_msa_output, window_size,
                                                      shift_size=-window_size // 2,
                                                      generate_mask=True)

    shifted_w_msa_input = shifted_w_msa_input.reshape(bs * num_window, num_patch_in_window, patch_depth)

    # 应用多头自注意力机制
    attn_prob, output = mhsa(shifted_w_msa_input, additive_mask=additive_mask)

    output = output.reshape(bs, num_window, num_patch_in_window, patch_depth)

    # 移位输出以便适应下一层的输入
    output, _ = shift_window(output, window_size, shift_size=window_size // 2, generate_mask=False)
    return output


# 定义patch合并模块
class PatchMerging(nn.Module):

    def __init__(self, model_dim, merge_size, output_depth_scale=0.5):
        super(PatchMerging, self).__init__()
        self.merge_size = merge_size
        # 线性层，将合并后的补丁映射到更小的维度
        self.proj_layer = nn.Linear(
            model_dim * merge_size * merge_size,
            int(model_dim * merge_size * merge_size * output_depth_scale)
        )

    def forward(self, input):
        bs, num_window, num_patch_in_window, patch_depth = input.shape
        window_size = int(math.sqrt(num_patch_in_window))

        input = window2image(input)

        # 将窗口中的补丁合并
        merged_window = F.unfold(input, kernel_size=(self.merge_size, self.merge_size),
                                 stride=(self.merge_size, self.merge_size)).transpose(-1, -2)
        # 通过线性层进行投影
        merged_window = self.proj_layer(merged_window)
        return merged_window


# 定义Swin Transformer块
class SwinTransformerBlock(nn.Module):

    def __init__(self, model_dim, window_size, num_head, act_layer=nn.GELU, drop=0.):
        super(SwinTransformerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(model_dim)  # 层归一化
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.layer_norm3 = nn.LayerNorm(model_dim)
        self.layer_norm4 = nn.LayerNorm(model_dim)
        self.wsma_mlp1 = nn.Linear(model_dim, 4 * model_dim)  # 线性层1
        self.wsma_mlp2 = nn.Linear(4 * model_dim, model_dim)  # 线性层2
        self.act = act_layer()  # 激活函数
        self.swsma_mlp1 = nn.Linear(model_dim, 4 * model_dim)  # 线性层3
        self.swsma_mlp2 = nn.Linear(4 * model_dim, model_dim)  # 线性层4
        self.drop = nn.Dropout(drop)  # Dropout

        self.mhsa1 = MultiHeadSelfAttention(model_dim, num_head)  # 多头自注意力机制1
        self.mhsa2 = MultiHeadSelfAttention(model_dim, num_head)  # 多头自注意力机制2

    def forward(self, input):
        bs, num_patch, patch_depth = input.shape
        input = input.float()

        input1 = self.layer_norm1(input)
        # 应用窗口多头自注意力机制
        w_msa_output = window_multi_head_self_attention(input1, self.mhsa1, window_size=4, num_head=num_head)
        bs, num_window, num_patch_in_window, patch_depth = w_msa_output.shape
        w_msa_output = input + w_msa_output.reshape(bs, num_patch, patch_depth)
        output1 = self.act(self.wsma_mlp1(self.layer_norm2(w_msa_output)))
        output1 = self.act(self.wsma_mlp2(output1))
        output1 += w_msa_output

        input2 = self.layer_norm3(output1)
        input2 = input2.reshape(bs, num_window, num_patch_in_window, patch_depth)
        # 对移位的窗口应用多头自注意力机制
        sw_msa_output = shift_window_multi_head_self_attention(input2, self.mhsa2, window_size=4, num_head=num_head)
        sw_msa_output = output1 + sw_msa_output.reshape(bs, num_patch, patch_depth)
        output2 = self.act(self.swsma_mlp1(self.layer_norm4(sw_msa_output)))
        output2 = self.act(self.swsma_mlp2(output2))
        output2 += sw_msa_output

        output2 = output2.reshape(bs, num_window, num_patch_in_window, patch_depth)

        return output2


# 定义Swin Transformer模型
class SwinTransformerModel(nn.Module):

    def __init__(self, patch_size=4, model_dim_C=64, num_classes=10,
                 window_size=4, num_head=num_head, merge_size=2):
        super(SwinTransformerModel, self).__init__()
        patch_depth = patch_size * patch_size * 3
        self.patch_size = patch_size
        self.model_dim_C = model_dim_C
        self.num_classes = num_classes

        self.patch_embedding_weight = nn.Parameter(torch.randn(patch_depth, model_dim_C))

        self.block1 = SwinTransformerBlock(model_dim_C, window_size, num_head)
        self.block2 = SwinTransformerBlock(model_dim_C * 2, window_size, num_head)
        self.block3 = SwinTransformerBlock(model_dim_C * 4, window_size, num_head)
        self.block4 = SwinTransformerBlock(model_dim_C * 8, window_size, num_head)

        self.patch_merging1 = PatchMerging(model_dim_C, merge_size)
        self.patch_merging2 = PatchMerging(model_dim_C * 2, merge_size)
        self.patch_merging3 = PatchMerging(model_dim_C * 4, merge_size)

        self.final_layer = nn.Linear(model_dim_C * 64, num_classes)

    def forward(self, image):
        patch_embedding = image

        # 第一个Swin Transformer块
        sw_msa_output = self.block1(patch_embedding)

        # 第一个patch合并
        merged_patch1 = self.patch_merging1(sw_msa_output)
        sw_msa_output_1 = self.block2(merged_patch1)

        # 第二个patch合并
        merged_patch2 = self.patch_merging2(sw_msa_output_1)
        sw_msa_output_2 = self.block3(merged_patch2)

        bs, num_window, num_patch_in_window, patch_depth = sw_msa_output_2.shape
        # 将输出重新整形以适应最终线性层的输入
        sw_msa_output_3 = sw_msa_output_2.reshape(bs, -1, patch_depth)

        return sw_msa_output_3
