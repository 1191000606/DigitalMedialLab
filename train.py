import re
import time
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import models
from eval import evaluate_loss
from torch.utils.data import DataLoader

from graph import init_graph, font_legend


def train_model(hyp_params, train_loader: DataLoader, valid_loader: DataLoader):
    model = getattr(models, hyp_params.model)(hyp_params)
    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.learning_rate)
    criterion = getattr(nn, hyp_params.criterion)()
    # patience = 10,先训练10次，不管有没有结果，10次之后，如果结果不变了，就让学习率乘以factor。verbose = True，更新的时候会打印输出提示信息的。完整版就看规约
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)

    batch_loss = []
    epoch_train_loss = []
    epoch_valid_loss = []
    epoch_valid_accuracy = []

    best_valid = 1e8
    for epoch in range(1, hyp_params.epoch_num + 1):
        start = time.time()
        train_loss, batch_loss_list = train(hyp_params, epoch, model, train_loader, optimizer, criterion)
        valid_loss, accuracy = evaluate_loss(hyp_params, model, valid_loader, criterion)

        end = time.time()
        duration = end - start
        # schedule采用ReduceLROnPlateau类型，超级实用的学习率调整策略，他会监测传入的指标，当这个指标不再下降或提高（取决于mode=min还是max）时，进行学习率更新，否则保持学习率不变。
        scheduler.step(valid_loss)

        print("-" * 50)
        print('Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f} | Valid Loss {:5.4f} | Valid Accuracy {:5.4f}'.format(epoch, duration, train_loss, valid_loss, accuracy))
        print("-" * 50)

        batch_loss += batch_loss_list
        epoch_train_loss.append(train_loss)
        epoch_valid_loss.append(valid_loss)
        epoch_valid_accuracy.append(accuracy)

        if valid_loss < best_valid:
            print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            torch.save(model, f'pre_trained_models/{hyp_params.name}.pt')
            best_valid = valid_loss

    draw_loss_graph(hyp_params, batch_loss, epoch_train_loss, epoch_valid_loss, epoch_valid_accuracy)
    return model


def train(hyp_params, epoch, model, train_loader, optimizer, criterion):
    batch_loss_list = []  # 一个batch的损失，用来绘图
    epoch_loss = 0  # 一个epoch的损失
    model.train()
    batch_num = hyp_params.train_num // hyp_params.batch_size
    proc_loss, proc_size = 0, 0  # 一次输出，eg 30个batch的损失
    start_time = time.time()
    for batch_index, (images_tensor, labels_tensor) in enumerate(train_loader):
        model.zero_grad()

        if hyp_params.use_cuda:
            with torch.cuda.device(0):
                images_tensor, labels_tensor = images_tensor.cuda(), labels_tensor.cuda()

        batch_size = images_tensor.size(0)

        net = nn.DataParallel(model) if batch_size > 10 else model
        output = net(images_tensor)
        loss = criterion(output, labels_tensor)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
        optimizer.step()

        batch_loss = loss.item() * batch_size
        batch_loss_list.append(loss.item())
        proc_loss += batch_loss
        proc_size += batch_size
        epoch_loss += batch_loss

        if batch_index % hyp_params.log_interval == 0 and batch_index > 0:
            avg_loss = proc_loss / proc_size
            elapsed_time = time.time() - start_time
            print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                  format(epoch, batch_index, batch_num, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
            proc_loss, proc_size = 0, 0

    return epoch_loss / hyp_params.train_num, batch_loss_list


def draw_loss_graph(hyp_params, batch_loss, epoch_train_loss, epoch_valid_loss, epoch_valid_accuracy):
    batch_x = [i + 1 for i in range(len(batch_loss))]
    init_graph("batch", f"{hyp_params.criterion}", f"loss graph ({len(batch_loss) // len(epoch_train_loss)} batch * {len(epoch_train_loss)} epoch)", dpi=1000)
    plt.plot(batch_x, batch_loss, linewidth=1)
    plt.savefig("./loss_graph/" + re.sub('[ :.]', '-', str(datetime.now())) + "-batch.jpg")
    plt.show()

    epoch_x = [i + 1 for i in range(len(epoch_train_loss))]
    init_graph("epoch", f"{hyp_params.criterion} And Accuracy", f"loss, accuracy graph ({len(epoch_train_loss)} epoch)", dpi=1000)
    plt.plot(epoch_x, epoch_train_loss, linewidth=1, label="train_loss")
    plt.plot(epoch_x, epoch_valid_loss, linewidth=1, label="valid_loss")
    plt.plot(epoch_x, epoch_valid_accuracy, linewidth=1, label="valid_accuracy")
    plt.legend(prop=font_legend, loc='upper right', frameon=False)
    plt.savefig("./loss_graph/" + re.sub('[ :.]', '-', str(datetime.now())) + "-epoch.jpg")
    plt.show()
