import torch
import numpy as np
from torch.utils import data
from torch import nn

def DataIter(batch_size, features, labels, is_train=True):
    dataset = data.TensorDataset(features, labels)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def TryGpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    else:
        return torch.device('cpu')

def train_regression_model(net, train_iter, test_iter, loss, epochs, trainer):
    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        net.train()
        loss_sum = 0
        total = 0
        for X, y in train_iter:
            y_prob = net(X)
            l = loss(y_prob, y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            with torch.no_grad():
                loss_sum += l.item() * y.size(0)
                total += y.size(0)
        train_loss.append(loss_sum/total)
        print(f'epoch:{epoch+1}\ntrain loss:{loss_sum/total:5f}')

        net.eval()
        loss_sum = 0
        total = 0
        with torch.no_grad():
            for X, y in test_iter:
                y_prob = net(X)
                l = loss(y_prob, y)
                loss_sum += l.item() * y.size(0)
                total += y.size(0)
            test_loss.append(loss_sum/total)
            print(f'test  loss:{loss_sum/total:5f}')
    return net, train_loss, test_loss

def train_class_model(net, train_iter, test_iter, loss, epochs, trainer):
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    for epoch in range(epochs):
        net.train()
        loss_sum = 0
        crr = 0
        total = 0
        for X, y in train_iter:
            y_prob = net(X)
            l = loss(y_prob, y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            with torch.no_grad():
                loss_sum += l.item() * y.size(0)
                crr += (y_prob.argmax(dim=1) == y).sum().item()
                total += y.size(0)
        train_loss.append(loss_sum/total)
        train_acc.append(crr/total)
        print(f'epoch:{epoch+1}\ntrain loss:{loss_sum/total:5f}\ttrain acc:{crr/total:.5f}')

        net.eval()
        loss_sum = 0
        crr = 0
        total = 0
        with torch.no_grad():
            for X, y in test_iter:
                y_prob = net(X)
                l = loss(y_prob, y)
                loss_sum += l.item() * y.size(0)
                crr += (y_prob.argmax(dim=1) == y).sum().item()
                total += y.size(0)
            test_loss.append(loss_sum/total)
            test_acc.append(crr/total)
            print(f'test  loss:{loss_sum/total:5f}\ttest  acc:{crr/total:.5f}')
    return net, train_loss, test_loss, train_acc, test_acc

