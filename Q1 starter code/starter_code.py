"""

QUESTION 1

Some helpful code for getting started.


"""

import torch
import torchvision
import torchvision.transforms as transforms
from imagenet10 import ImageNet10

import pandas as pd
import os

from config import *

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from model import *

# Gathers the meta data for the images
paths, classes = [], []
for i, dir_ in enumerate(CLASS_LABELS):
    for entry in os.scandir(ROOT_DIR + dir_):
        if (entry.is_file()):
            paths.append(entry.path)
            classes.append(i)

data = {'path': paths, 'class': classes}

data_df = pd.DataFrame(data, columns=['path', 'class'])
data_df = data_df.sample(frac=1).reset_index(drop=True)  # Shuffles the data

# See what the dataframe now contains
print("Found", len(data_df), "images.")
# If you want to see the image meta data
print(data_df.head())

# Split the data into train and test sets and instantiate our new ImageNet10 objects.
train_split = 0.80  # Defines the ratio of train/valid data.

# valid_size = 1.0 - train_size
train_size = int(len(data_df) * train_split)

data_transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

dataset_train = ImageNet10(
    df=data_df[:train_size],
    transform=data_transform,
)

dataset_valid = ImageNet10(
    df=data_df[train_size:].reset_index(drop=True),
    transform=data_transform,
)

# Data loaders for use during training
train_loader = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=64,
                                           shuffle=True,
                                           num_workers=2)

valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                           batch_size=128,
                                           shuffle=True,
                                           num_workers=2)

# See what you've loaded
print("len(dataset_train)", len(dataset_train))
print("len(dataset_valid)", len(dataset_valid))

print("len(train_loader)", len(train_loader))
print("len(valid_loader)", len(valid_loader))


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def q1_1_fc():
    # prepare tensorboard
    writer = SummaryWriter(comment="-q1.1-FC")
    output_path = writer.get_logdir()
    net = FCModel(3 * 128 * 128, len(classes))

    # define loss and optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    # get one batch of training data
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64,
        shuffle=
        False,  # make the first batch of data consistent between executions
        num_workers=2)
    train_data, train_label = next(iter(train_loader))

    # use gpu if available
    if torch.cuda.is_available():
        print("using gpu")
        net = net.cuda()
        criterion = criterion.cuda()
        train_data = train_data.cuda()
        train_label = train_label.cuda()

    # training code
    epoch_num = 40
    train_loss = []
    val_loss = []
    for epoch_idx in range(epoch_num):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(train_data)
        loss = criterion(outputs, train_label)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.cpu().detach().numpy().item())
        writer.add_scalar("Loss/train", loss.cpu().detach(), epoch_idx)

        # validation
        with torch.no_grad():
            tmp = 0
            for i, (val_data, val_label) in enumerate(valid_loader):
                val_data = val_data.to(train_data.device)
                val_label = val_label.to(train_label.device)
                outputs = net(val_data)
                loss = criterion(outputs, val_label)
                tmp += loss.cpu().detach()
            writer.add_scalar("Loss/val", tmp / len(valid_loader), epoch_idx)
            val_loss.append(tmp.numpy().item() / len(valid_loader))
        print(
            f"Train Epoch({epoch_idx + 1} / {epoch_num}), train_loss: {train_loss[-1]}, val_loss: {val_loss[-1]}"
        )

    print("save loss")
    torch.save(val_loss, os.path.join(output_path, "val_loss"))
    torch.save(train_loss, os.path.join(output_path, "train_loss"))
    writer.close()


def q1_1_cnn():
    # prepare tensorboard
    writer = SummaryWriter(comment="-q1.1-CNN")
    output_path = writer.get_logdir()
    net = CNNModel(3, len(classes))

    # define loss and optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    # get one batch of training data
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=64,
        shuffle=
        False,  # make the first batch of data consistent between executions
        num_workers=2)
    train_data, train_label = next(iter(train_loader))

    # use gpu if available
    if torch.cuda.is_available():
        print("using gpu")
        net = net.cuda()
        criterion = criterion.cuda()
        train_data = train_data.cuda()
        train_label = train_label.cuda()

    # training code
    epoch_num = 40
    train_loss = []
    val_loss = []
    for epoch_idx in range(epoch_num):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(train_data)
        loss = criterion(outputs, train_label)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.cpu().detach().numpy().item())
        writer.add_scalar("Loss/train", loss.cpu().detach(), epoch_idx)

        # validation
        with torch.no_grad():
            tmp = 0
            for i, (val_data, val_label) in enumerate(valid_loader):
                val_data = val_data.to(train_data.device)
                val_label = val_label.to(train_label.device)
                outputs = net(val_data)
                loss = criterion(outputs, val_label)
                tmp += loss.cpu().detach()
            writer.add_scalar("Loss/val", tmp / len(valid_loader), epoch_idx)
            val_loss.append(tmp.numpy().item() / len(valid_loader))
        print(
            f"Train Epoch({epoch_idx + 1} / {epoch_num}), train_loss: {train_loss[-1]}, val_loss: {val_loss[-1]}"
        )

    print("save loss")
    torch.save(val_loss, os.path.join(output_path, "cnn_val_loss"))
    torch.save(train_loss, os.path.join(output_path, "cnn_train_loss"))
    writer.close()


def q1_2():
    # prepare tensorboard
    writer = SummaryWriter(comment="-q1.2-CNN")
    output_path = writer.get_logdir()
    net = CNNModel(3, len(classes))

    # define loss and optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), amsgrad=True, weight_decay=1e-5)

    device = torch.device('cpu')
    # use gpu if available
    if torch.cuda.is_available():
        print("using gpu")
        net = net.cuda()
        criterion = criterion.cuda()
        device = torch.device('cuda')

    # training code
    epoch_num = 60
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    step = 0
    for epoch_idx in range(epoch_num):
        tmp = 0
        train_label_gt = []
        train_label_pred = []
        for i, (train_data, train_label) in enumerate(train_loader):
            train_data = train_data.to(device)
            train_label = train_label.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(train_data)
            loss = criterion(outputs, train_label)
            loss.backward()
            optimizer.step()

            train_label_gt.append(train_label.cpu())
            train_label_pred.append(outputs.cpu().detach())
            tmp += loss.cpu().detach()
            writer.add_scalar("Loss/train_per_batch",
                              loss.cpu().detach(), step)
            step += 1
        train_loss.append(tmp.numpy().item() / len(train_loader))
        train_acc.append(
            accuracy(torch.cat(train_label_pred),
                     torch.cat(train_label_gt))[0])
        writer.add_scalar("acc/train", train_acc[-1], epoch_idx)
        writer.add_scalar("Loss/train", tmp / len(train_loader), epoch_idx)

        # validation
        with torch.no_grad():
            tmp = 0
            val_label_gt = []
            val_label_pred = []
            for i, (val_data, val_label) in enumerate(valid_loader):
                val_data = val_data.to(device)
                val_label = val_label.to(device)
                outputs = net(val_data)
                loss = criterion(outputs, val_label)
                tmp += loss.cpu().detach()
                val_label_gt.append(val_label.cpu())
                val_label_pred.append(outputs.cpu().detach())
            val_acc.append(
                accuracy(torch.cat(val_label_pred),
                         torch.cat(val_label_gt))[0])
            writer.add_scalar("acc/val", val_acc[-1], epoch_idx)
            writer.add_scalar("Loss/val", tmp / len(valid_loader), epoch_idx)
            val_loss.append(tmp.numpy().item() / len(valid_loader))
        print(
            f"Train Epoch({epoch_idx + 1} / {epoch_num}), train_loss: {train_loss[-1]}, val_loss: {val_loss[-1]}, train_acc: {train_acc[-1]:.3f}%, val_acc: {val_acc[-1]:.3f}%"
        )

    print("save loss")
    torch.save(val_loss, os.path.join(output_path, "cnn_val_loss"))
    torch.save(train_loss, os.path.join(output_path, "cnn_train_loss"))
    print("save acc")
    torch.save(val_acc, os.path.join(output_path, "cnn_val_acc"))
    torch.save(train_acc, os.path.join(output_path, "cnn_train_acc"))
    writer.close()


# q1_1_fc()
# q1_1_cnn()
q1_2()
