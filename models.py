import torch
from torch import nn


class MyResBlock(nn.Module):
    def __init__(self, device, channel_num, down_sample=None):
        super().__init__()
        self.conv2d1 = nn.Conv2d(in_channels=channel_num, out_channels=channel_num, kernel_size=(3, 3), device=device, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(num_features=channel_num, device=device)
        self.relu1 = nn.ReLU()

        self.conv2d2 = nn.Conv2d(in_channels=channel_num, out_channels=channel_num, kernel_size=(3, 3), device=device, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(num_features=channel_num, device=device)
        self.down_sample = down_sample  # 图像缩小即为下采样，具体而言是 一块像素中取最大/平均; 与之相对，上采样就是放大，主要是插值与反卷积
        self.relu2 = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv2d1(x)
        out = self.batchNorm1(out)
        out = self.relu1(out)
        out = self.conv2d2(out)
        out = self.batchNorm2(out)

        if self.down_sample:
            residual = self.down_sample(x)
        out += residual
        out = self.relu2(out)

        return out


class MyResNet(nn.Module):
    def __init__(self, hyp_params):
        super(MyResNet, self).__init__()

        device = torch.device("cuda") if hyp_params.use_cuda else torch.device("cpu")
        block_num = hyp_params.block_num
        block_channel_num = hyp_params.block_channel_num
        linear1_out_channel_num = hyp_params.linear_channel_num

        self.conv2d1 = nn.Conv2d(in_channels=1, out_channels=block_channel_num, kernel_size=(3, 3), device=device, padding=1)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.res_blocks = nn.Sequential(*(block_num * [MyResBlock(device, block_channel_num)]))
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten(start_dim=1)  # 维度从0开始
        self.linear1 = nn.Linear(in_features=block_channel_num * 49, out_features=linear1_out_channel_num, device=device)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=linear1_out_channel_num, out_features=hyp_params.label_num, device=device)

    def forward(self, x):
        out = self.conv2d1(x)
        out = self.relu1(out)
        out = self.max_pool1(out)
        out = self.res_blocks(out)
        out = self.max_pool2(out)
        out = self.flatten(out)
        out = self.linear1(out)
        out = self.relu2(out)
        out = self.linear2(out)

        return out
