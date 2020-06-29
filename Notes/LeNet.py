###############################################################################
# PyTorch implementation of LeNet (1998) CNN
###############################################################################
import time, os, sys, random, datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms

print(torch.version.__version__)

###############################################################################
use_cuda = torch.cuda.is_available()
#use_cuda = True
#use_cuda = False
###############################################################################
def cuda(arr):
    if use_cuda:
        return arr.cuda()
    return arr

###############################################################################
# tunable parameters

#learning_rate = 0.0001
learning_rate = 0.0003  # default
#learning_rate = 0.001
#learning_rate = 0.003

global nEpochs
nEpochs = 2

dropout = 0.0
#dropout = 0.1   # default
#dropout = 0.3
#dropout = 0.5

L2_lambda = 0.0
#L2_lambda = 0.001
#L2_lambda = 0.002

log_interval = 100
#log_interval = 10

###############################################################################
def resize(x, kernel_size, dilation, stride, padding):
    x = int(1 + (x + 2*padding - dilation * (kernel_size - 1) - 1)/stride)
    return x

###############################################################################
class LeNet(nn.Module):
###############################################################################
    def __init__(self, specs, dropout=0.0):
        super(LeNet, self).__init__()

        H, W, C0, C1, C2, kernel_size, F1, F2, nDigits, padding = specs
        pooling = 2
        stride = 1
        dilation = 1

        #self.pool = pool = nn.AvgPool2d(pooling)
        self.pool = pool = nn.MaxPool2d(pooling)

        self.conv1 = nn.Conv2d(1, C1, kernel_size, padding=padding)
        H = resize(H, kernel_size, dilation, stride, padding)
        W = resize(W, kernel_size, dilation, stride, padding)

        H = resize(H, pooling, dilation, pooling, 0)
        W = resize(W, pooling, dilation, pooling, 0)

        self.conv2 = nn.Conv2d(C1, C2, kernel_size, padding=padding)

        H = resize(H, kernel_size, dilation, stride, padding)
        W = resize(W, kernel_size, dilation, stride, padding)

        H = resize(H, pooling, dilation, pooling, 0)
        W = resize(W, pooling, dilation, pooling, 0)

        #print(H, W)
        size = H * W * C2

        self.linear0 = nn.Linear(size, F1)
        self.linear1 = nn.Linear(F1, F2)
        self.linear2 = nn.Linear(F2, 10)

        self.non_linear = nn.LeakyReLU(negative_slope=0.01)
        
        self.dropout = nn.Dropout(dropout)

        for p in self.parameters(): # optionally apply different randomization
            if p.dim() > 1:
                nn.init.kaiming_normal_(p)
                pass

    #################################################################
    def forward(self, prev):
        nBatch = len(prev)
        #print(prev.shape)
        prev = self.conv1(prev)
        prev = self.non_linear(prev)
        prev = self.dropout(prev)
        prev = self.pool(prev)

        prev = self.conv2(prev)
        prev = self.non_linear(prev)
        prev = self.dropout(prev)
        prev = self.pool(prev)

        prev = prev.view(nBatch, -1)
        #print(prev.shape)

        prev = self.linear0(prev)
        prev = self.non_linear(prev)
        prev = self.dropout(prev)

        prev = self.linear1(prev)
        prev = self.non_linear(prev)
        prev = self.dropout(prev)

        prev = self.linear2(prev)

        return prev

###############################################################################
def train_LeNet(model, train_loader, test_loader):
    if use_cuda:
        model = model.cuda()

    # define the loss functions
    criterion = nn.CrossEntropyLoss(reduction='sum')

    # choose an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)
    #optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)

    start = time.time()
    w_decay = 0.5
    for e in range(nEpochs):
        total_train_images = 0
        total_train_loss = 0
        train_images = 0
        train_loss = 0
        w_images = 0
        w_loss = 0

        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = cuda(data), cuda(target)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_images += len(data)
            train_loss += loss.data.item()

            if train_images > log_interval:
                total_train_images += train_images
                total_train_loss += train_loss
                if w_images == 0:
                    w_loss = train_loss
                    w_images = train_images
                log_message(None, "%3d %8d %8.3f %8.3f     %6.1f" % (e, total_train_images, train_loss/train_images, w_loss/w_images, (time.time()-start)))

                w_images = w_decay * w_images + train_images
                w_loss = w_decay * w_loss + train_loss
                train_images = 0
                train_loss = 0

                #break

        test_images = 0
        test_loss = 0
        nCorrect = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = cuda(data), cuda(target)
                output = model(data)
                loss = criterion(output, target)

                test_images += len(data)
                test_loss += loss.data.item()
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max value
                nCorrect += pred.eq(target.view_as(pred)).sum().item() # count correct items

        log_message(log_file, "%3d %8d %8.3f %8.3f %8.3f %8.1f     %6.1f" % (e, (e+1)*total_train_images, total_train_loss/total_train_images, w_loss/w_images, test_loss/test_images, 100*nCorrect/test_images, (time.time()-start)))

    return model

###############################################################################
###############################################################################
def log_message(outf, message):
    print(message)
    if not outf is None:
        outf.write(message)
        outf.write("\n")
        outf.flush()

###############################################################################
# start main
###############################################################################
if __name__ == "__main__":
    # process command line args
    parser = argparse.ArgumentParser(description='digit recognition via LeNet (1998) CNN')

    parser.add_argument('-log', default='./log/MNIST-temp.log', help='name of log file')
    parser.add_argument('-noisy', default=2, type=int, help='level of reporting')
    parser.add_argument('-path', default='./data/MNIST', help='path of MNIST data')
    parser.add_argument('-save', default='./model/MNIST/temp.pth', help='saved model file')
    parser.add_argument('-batch', default=64, type=int, help='batch size')
    parser.add_argument('-epochs', default=None, type=int, help='batch size')

    args = parser.parse_args()

    print("args =",args)
    global log_file
    os.makedirs(os.path.dirname(args.log), exist_ok=True) # ensure output directory exists
    log_file = open(args.log, "a")
    log_message(log_file, "\nstarting run: %s %s" % (datetime.date.today(), datetime.datetime.now()))

    H=28    # don't change
    W=28    # don't change
    C0=1    # don't change
    C1=6
    C2=16
    kernel_size=5
    F1 = 120
    F2 = 84
    nDigits=10    # don't change
    padding=0

    #C1 = 20
    #padding = 1

    specs = [H, W, C0, C1, C2, kernel_size, F1, F2, nDigits, padding]
    model = LeNet(specs, dropout=dropout)
    print(model)
    log_message(log_file, "specs = %s, dropout = %f L2 = %f, batch = %d" % (str(specs), dropout, L2_lambda, args.batch))

    data_path = args.path
    
    batch_size = args.batch
    test_batch_size = 1000

    if args.epochs:
        nEpochs = args.epochs

    # the following lines replace a lot of "boring" code
    mean = 0.1307
    std = 0.3081
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=True, download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=False, 
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    model = train_LeNet(model, train_loader, test_loader)

    os.makedirs(os.path.dirname(args.save), exist_ok=True) # ensure output directory exists
    torch.save(model, args.save)
