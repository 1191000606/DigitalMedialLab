from struct import unpack

import numpy as np
import torch
from torch import int64
from torch.utils.data.dataset import Dataset
import cv2


def load_data(item_num, label_num, images_file, label_file):
    image_row_num = 28
    image_column_num = 28
    assert (2051, item_num, image_row_num, image_column_num) == unpack(">4i", images_file.read(16))
    byte_num = item_num * image_row_num * image_column_num
    image_bytes = unpack(">" + str(byte_num) + "B", images_file.read(byte_num))
    images_file.close()

    images_tensor = torch.Tensor(image_bytes)
    images_tensor = torch.reshape(images_tensor, (item_num, image_row_num, image_column_num))
    images_tensor = torch.unsqueeze(images_tensor, 1).detach()

    assert (2049, item_num) == unpack(">2i", label_file.read(8))
    raw_labels_tensor = torch.tensor(unpack(">" + str(item_num) + "B", label_file.read(item_num)), dtype=int64).detach()
    label_file.close()
    onehot_labels_tensor = torch.zeros(item_num, label_num).scatter_(dim=1, index=torch.unsqueeze(raw_labels_tensor, 1), src=torch.FloatTensor([[1]] * item_num))

    return images_tensor, onehot_labels_tensor


def load_extend_data(split="train"):
    extend_characters = ["(", ")", "div", "mul", "plus", "sub"]
    extend_images = []
    extend_labels = []

    if split == "train":
        item_num_per_label = [5500, 5500, 3434, 4102, 4059, 5500]
        start_index = [1, 1, 1, 1, 1, 1]
    else:
        item_num_per_label = [1000, 1000, 800, 1000, 1000, 1000]
        start_index = [5501, 5501, 3435, 4103, 4060, 5501]
    label_num = 16

    for i in range(6):
        path = "./data/cfs/" + extend_characters[i] + "/"
        for j in range(item_num_per_label[i]):
            extend_images.append(torch.Tensor(cv2.imread(path + str(j + start_index[i]) + ".jpg", flags=0)))
        extend_labels += [i + 10] * item_num_per_label[i]
    images_tensor = torch.stack(extend_images, 0)
    images_tensor = torch.unsqueeze(images_tensor, 1).detach()
    labels_tensor = torch.tensor(extend_labels, dtype=int64).detach()
    onehot_labels_tensor = torch.zeros(sum(item_num_per_label), label_num).scatter_(dim=1, index=torch.unsqueeze(labels_tensor, 1), src=torch.FloatTensor([[1]] * sum(item_num_per_label)))

    return images_tensor, onehot_labels_tensor


class MyDataset(Dataset):
    def __init__(self, split="train", label_num=16):
        if split == "train":
            images_file = open("./data/train-images.idx3-ubyte", "rb")
            label_file = open("./data/train-labels.idx1-ubyte", "rb")
            item_num = 60000
        else:
            images_file = open("./data/t10k-images.idx3-ubyte", "rb")
            label_file = open("./data/t10k-labels.idx1-ubyte", "rb")
            item_num = 10000

        self.images_tensor, self.labels_tensor = load_data(item_num, label_num, images_file, label_file)

        if label_num == 16:
            extend_images_tensor, extend_labels_tensor = load_extend_data(split)
            self.images_tensor = torch.cat((self.images_tensor, extend_images_tensor), 0)
            self.labels_tensor = torch.cat((self.labels_tensor, extend_labels_tensor), 0)

    def __len__(self):
        return len(self.labels_tensor)

    def __getitem__(self, index):
        return self.images_tensor[index], self.labels_tensor[index]
