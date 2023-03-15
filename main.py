import argparse
import random

import torch
from torch.utils.data import DataLoader
from dataloader import MyDataset
from eval import evaluate_metric
from train import train_model
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

parser = argparse.ArgumentParser(description='HIT Digital Media Lab2 1191000606 陈一帆')

# Fixed
parser.add_argument('--model', type=str, default='MyResNet', help='the model to be chosen')

# Task
parser.add_argument('--name', type=str, default='mnist', help='name of the trial (default: "mnist")')
parser.add_argument('--dataset_extend', action='store_false', help='whether to use extended dataset  (default: true)')

# Architecture
parser.add_argument('--block_num', type=int, default=20, help='number of residual blocks in the residual neural network (default: 20)')
parser.add_argument('--block_channel_num', type=int, default=3, help='number of input and output channels for residual blocks (default: 3)')
parser.add_argument('--linear_channel_num', type=int, default=49, help='number of output channels for the first linear layer in the residual neural network (default: 49)')

# Training
parser.add_argument('--optim', type=str, default='Adam', help='optimizer to use (default: Adam)')
parser.add_argument('--criterion', type=str, default='CrossEntropyLoss', help='criterion to use (default: CrossEntropyLoss)')  # 评价准则,不同的任务/数据集下选不同的criterion,eg.L1Loss
parser.add_argument('--use_cuda', action='store_false', help='choose to use cuda (default: true)')
parser.add_argument('--log_interval', type=int, default=60, help='after how many batches log the result (default: 60)')

# Tuning
parser.add_argument('--learning_rate', type=float, default=1e-3, help='initial learning rate (default: 1e-3)')
parser.add_argument('--batch_size', type=int, default=200, help='batch size (default: 200)')
parser.add_argument('--epoch_num', type=int, default=15, help='number of epochs (default: 15)')
parser.add_argument('--when', type=int, default=3, help='after how many epochs begin to decay learning rate if loss doesn\'t reduce (default: 3)')
parser.add_argument('--clip', type=float, default=0.8, help='gradient clip value (default: 0.8)')
parser.add_argument('--valid_radio', type=float, default=0.05, help='the radio of validation item when splitting the train dataset (default: 0.05)')

args = parser.parse_args()
hyp_params = args

torch.set_default_tensor_type("torch.FloatTensor")
if torch.cuda.is_available():
    if hyp_params.use_cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        print("using a cuda device")
    else:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
else:
    if hyp_params.use_cuda:
        print("WARNING: cuda device is not available. Using cpu device instead")
        hyp_params.use_cuda = False
    else:
        print("using cpu device")

if hyp_params.dataset_extend:
    hyp_params.label_num = 16
else:
    hyp_params.label_num = 10

train_dataset = MyDataset("train", hyp_params.label_num)
# train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [len(train_dataset) * (1 - hyp_params.valid_radio), len(train_dataset) * hyp_params.valid_radio], generator=torch.Generator(torch.device("cpu")))
hyp_params.valid_num = round(len(train_dataset) * hyp_params.valid_radio)
hyp_params.train_num = len(train_dataset) - hyp_params.valid_num
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [hyp_params.train_num, hyp_params.valid_num])
test_dataset = MyDataset("test", hyp_params.label_num)
hyp_params.test_num = len(test_dataset)

# shuffle这里如果是True的话,那么取出来的tensor在内存中了,Dataset设置了显存也没用。两种方法,一种是设置为false,另一种是修改torch里面torch\utils\data\sampler.py中generate函数(具体位置报错里面可以找到)，参数里面加上device='cuda'
train_dataloader = DataLoader(dataset=train_dataset, batch_size=hyp_params.batch_size, shuffle=True)
valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=hyp_params.batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=hyp_params.batch_size, shuffle=True)

if __name__ == "__main__":
    model = train_model(hyp_params, train_dataloader, valid_dataloader)
    # model = torch.load(f"./pre_trained_models/{hyp_params.name}.pt")
    MAE, MSE, CE, accuracy, truths, results, error_samples_list = evaluate_metric(hyp_params, model, test_dataloader)

    print("-" * 50)
    print(f"Cross Entropy Loss: {CE}")
    print(f"Mean Absolute Loss: {MAE}")
    print(f"Mean Square Loss: {MSE}")
    print(f"Accuracy: {accuracy}")

    random_samples_index = [random.randint(0, hyp_params.test_num - 1) for i in range(10)]
    random_samples_list = [test_dataset[i][0] for i in random_samples_index]
    samples_list = error_samples_list
    # samples_list = random_samples_list

    samples_result_one_hot = model(torch.stack(samples_list, 0))
    samples_result = torch.max(samples_result_one_hot, 1)[1].tolist()
    extend_characters = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "(", ")", "div", "mul", "plus", "sub"]

    samples_result = [extend_characters[samples_result[i]] for i in range(len(samples_result))]

    writer = SummaryWriter("log")

    for i in range(len(samples_result)):
        writer.add_image(tag=f"第{i + 1}个图片，鉴定为：" + samples_result[i], img_tensor=samples_list[i][0], dataformats="HW")
