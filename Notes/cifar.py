###############################################################################
# PyTorch implementation of CIFAR-10 classifier
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
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

###############################################################################
# hyperparameters
###############################################################################
#learning_rate = 0.0001
learning_rate = 0.0003
#learning_rate = 0.001
#learning_rate = 0.003

global nEpochs
nEpochs = 2

dropout = 0.0
#dropout = 0.1
#dropout = 0.3
#dropout = 0.5

L2_lambda = 0.0
#L2_lambda = 0.001
#L2_lambda = 0.002

#mean = 0.5
#std = 0.5
mean = 0.0
std = 1.0

# log_interval has no effect on accuracy, only reporting
log_interval = 100
#log_interval = 10

###############################################################################
def resize(x, kernel_size, dilation, stride, padding):
    x = int(1 + (x + 2*padding - dilation * (kernel_size - 1) - 1)/stride)
    return x

###############################################################################
class CIFAR(nn.Module):
###############################################################################
    def __init__(self, specs, dropout=0.0):
        super(CIFAR, self).__init__()

        H, W, C0, C1, C2, kernel_size, F1, F2, nClasses, padding = specs
        pooling = 2
        stride = 1
        dilation = 1

        #self.pool = pool = nn.AvgPool2d(pooling)
        self.pool = pool = nn.MaxPool2d(pooling)

        self.conv1 = nn.Conv2d(C0, C1, kernel_size, padding=padding)
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
        F0 = H * W * C2

        self.linear0 = nn.Linear(F0, F1)
        self.linear1 = nn.Linear(F1, F2)
        self.linear2 = nn.Linear(F2, nClasses)

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
def train_CIFAR(model, train_loader, test_loader):
    if use_cuda:
        model = model.cuda()

    # define the loss function
    criterion = nn.CrossEntropyLoss(reduction='sum')

    # choose an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)
    #optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)

    start = time.time()
    w_decay = 0.8 # this smooths the reported loss -- it has no effect on performance
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
                w_loss   = w_decay * w_loss   + train_loss
                train_images = 0
                train_loss = 0

                #break # terminate epoch early - useful for debugging

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
def display_image(img):
    npimg = img * std + mean
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

###############################################################################
# start main
###############################################################################
if __name__ == "__main__":
    # process command line args
    parser = argparse.ArgumentParser(description='cifar-10 recognition starter code with LeNet (1998) CNN')

    parser.add_argument('-log', default='./log/CIFAR-temp.log', help='name of log file')
    parser.add_argument('-noisy', default=2, type=int, help='level of reporting')
    #parser.add_argument('-path', default='./data/CIFAR', help='path of CIFAR data')
    parser.add_argument('-path', required=True, help='path of CIFAR data')
    parser.add_argument('-save', default='./model/CIFAR/temp.pth', help='saved model file')
    parser.add_argument('-batch', default=64, type=int, help='batch size')
    parser.add_argument('-epochs', default=nEpochs, type=int, help='number of epochs')

    args = parser.parse_args()

    print("args =",args)
    global log_file
    os.makedirs(os.path.dirname(args.log), exist_ok=True) # ensure output directory exists
    log_file = open(args.log, "a")
    log_message(log_file, "\nstarting run: %s %s" % (datetime.date.today(), datetime.datetime.now()))

    H=32    # don't change
    W=32    # don't change
    C0=3    # don't change
    C1=6
    C2=16
    kernel_size=5
    F1 = 120
    F2 = 84
    nClasses=10    # don't change
    padding=0

    specs = [H, W, C0, C1, C2, kernel_size, F1, F2, nClasses, padding]
    model = CIFAR(specs, dropout=dropout)
    print(model)
    log_message(log_file, "specs = %s, dropout = %f L2 = %f, batch = %d" % (str(specs), dropout, L2_lambda, args.batch))

    data_path = args.path
    batch_size = args.batch
    test_batch_size = 1000 # test_batch_size has no effect on accuracy
    nEpochs = args.epochs

    # the following lines replace a lot of "boring" code
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_path, train=True, download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean, mean, mean), (std, std, std))])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(data_path, train=False, 
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean, mean, mean), (std, std, std))])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    # just for fun, let's display a random image
    batch = next(iter(train_loader))
    #print(batch)
    image = batch[0][0]
    label = batch[1][0]
    print(label, classes[label])
    display_image(image)
    
    model = train_CIFAR(model, train_loader, test_loader)

    os.makedirs(os.path.dirname(args.save), exist_ok=True) # ensure output directory exists
    torch.save(model, args.save)
