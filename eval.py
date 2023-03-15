import torch
from torch import nn
from sklearn.metrics import precision_score


def evaluate_loss(hyp_params, model, valid_loader, criterion):
    model.eval()
    total_loss = 0.0
    results = []
    truths = []

    with torch.no_grad():
        for batch_index, (images_tensor, labels_tensor) in enumerate(valid_loader):
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    images_tensor, labels_tensor = images_tensor.cuda(), labels_tensor.cuda()

            batch_size = images_tensor.size(0)

            net = nn.DataParallel(model) if batch_size > 10 else model
            output = net(images_tensor)

            results += torch.max(output, 1)[1].tolist()
            truths += torch.max(labels_tensor, 1)[1].tolist()
            total_loss += criterion(output, labels_tensor).item() * batch_size

    avg_loss = total_loss / hyp_params.test_num
    accuracy = precision_score(truths, results, average="micro")

    return avg_loss, accuracy


def evaluate_metric(hyp_params, model, test_loader):
    model.eval()
    MAE_criterion = torch.nn.L1Loss()
    MSE_criterion = torch.nn.MSELoss()
    CE_criterion = torch.nn.CrossEntropyLoss()

    results = []
    truths = []
    error_samples_list = []
    MAE = 0.0
    MSE = 0.0
    CE = 0.0

    with torch.no_grad():
        for batch_index, (images_tensor, labels_tensor) in enumerate(test_loader):
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    images_tensor, labels_tensor = images_tensor.cuda(), labels_tensor.cuda()

            batch_size = images_tensor.size(0)

            net = nn.DataParallel(model) if batch_size > 10 else model
            output = net(images_tensor)

            MAE += MAE_criterion(output, labels_tensor).item() * batch_size
            MSE += MSE_criterion(output, labels_tensor).item() * batch_size
            CE += CE_criterion(output, labels_tensor).item() * batch_size

            result = torch.max(output, 1)[1].tolist()
            truth = torch.max(labels_tensor, 1)[1].tolist()

            error_samples_list += [images_tensor[i] for i in range(len(result)) if result[i] != truth[i]]
            results += result
            truths += truth

    MAE /= hyp_params.test_num
    MSE /= hyp_params.test_num
    CE /= hyp_params.test_num
    accuracy = precision_score(truths, results, average="micro")

    return MAE, MSE, CE, accuracy, truths, results, error_samples_list
