###############################################################################
# PyTorch examples - MSELoss (Mean Squared Error)
###############################################################################
import time
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
#from torch.autograd import Variable

###############################################################################
use_cuda = torch.cuda.is_available()
use_cuda = False

###############################################################################
# hyperparameters
###############################################################################
nLinear1 = 100
nLinear2 = 100

#learning_rate = 0.0001
learning_rate = 0.001
#learning_rate = 0.01
#learning_rate = 0.1

# regularization (dropout, L2, batch size) -- lecture 4
dropout = 0.0
#dropout = 0.1

L2_lambda = 0.0
#L2_lambda = 0.1

#batch_size = 100
batch_size = 10
#batch_size = 1

# control amount of training data
nTrainPoints = 10
#nTrainPoints = 19
#nTrainPoints = 1009
nTrainPoints = 5003

nDevPoints = 101

nTestPoints = 111

#DesiredTrainSamples = 5000
DesiredTrainSamples = 50000

DisplayInterval = 5000

###############################################################################
class NN(nn.Module):
###############################################################################
    def __init__(self, input_size, nLinear1, nLinear2, output_size, dropout=0.0):
        super(NN, self).__init__()
        self.linear0 = nn.Linear(input_size, nLinear1)
        self.linear1 = nn.Linear(nLinear1, nLinear2)

        self.out0 = nn.Linear(nLinear2, 1)    # output for func0
        self.out1 = nn.Linear(nLinear2, 1)    # output for func1

        # all applicable non-linear functions from PyTorch with default parameters
        # pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        #self.non_linear = nn.Sigmoid()
        #self.non_linear = nn.Tanh()
        #self.non_linear = nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=False, min_value=None, max_value=None)
        #self.non_linear = nn.Softsign()
        #self.non_linear = nn.LogSigmoid()

        #self.non_linear = nn.Tanhshrink()
        #self.non_linear = nn.Hardshrink(lambd=0.5)
        #self.non_linear = nn.Softshrink(lambd=0.5)

        #self.non_linear = nn.CELU(alpha=1.0, inplace=False)
        #self.non_linear = nn.ELU()
        #self.non_linear = nn.PReLU(num_parameters=1, init=0.25)
        #self.non_linear = nn.ReLU(inplace=False)
        #self.non_linear = nn.ReLU6(inplace=False)
        #self.non_linear = nn.RReLU(lower=0.125, upper=0.3333333333333333, inplace=False)
        #self.non_linear = nn.Softplus(beta=1, threshold=20)
        #self.non_linear = nn.SELU(inplace=False)
        self.non_linear = nn.LeakyReLU(negative_slope=0.01, inplace=False)

        self.dropout = nn.Dropout(dropout)

        for p in self.parameters(): # optionally apply different randomization
            if p.dim() > 1:
                # all random initializations from PyTorch with default parameters
                nn.init.constant_(p, val=1.0)   # this is a really bad idea -- don't use it
                nn.init.orthogonal_(p, gain=1)
                nn.init.sparse_(p, sparsity=0.9, std=0.01)

                nn.init.uniform_(p, a=0.0, b=1.0)   # this works poorly, see next
                nn.init.uniform_(p, a=-1.0, b=1.0)  # default is a=0.0, b=1.0
                nn.init.normal_(p, mean=0.0, std=1.0)

                nn.init.kaiming_uniform_(p, a=0, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.kaiming_normal_(p, a=0, mode='fan_in', nonlinearity='leaky_relu')

                nn.init.xavier_uniform_(p)
                nn.init.xavier_normal_(p)

    ###########################################################################
    def forward(self, inputs):
        nPoints = len(inputs)
        x = inputs.view(nPoints, -1)

        x = self.linear0(x)
        x = self.non_linear(x)
        x = self.dropout(x)

        x = self.linear1(x)
        x = self.non_linear(x)
        x = self.dropout(x)

        out0 = self.out0(x)  # func0
        out1 = self.out1(x)  # func1

        return out0, out1

###############################################################################
def train(model, optimizer, criteria, data, idx_list, update=True):
    model.zero_grad()

    nPoints = len(idx_list)
    inputs = []
    targets0 = []
    targets1 = []
    loss0 = 0
    loss1 = 0

    for i in idx_list:
        inputs.append(data[0][i])
        targets0.append(data[1][i])
        targets1.append(data[2][i])

    #inputs  = Variable(torch.Tensor(inputs))   # Variable is no longer needed
    inputs  = torch.Tensor(inputs)
    targets0 = torch.Tensor(targets0)
    targets1 = torch.Tensor(targets1)

    if use_cuda:
        inputs = inputs.cuda()
        targets0 = targets0.cuda()
        targets1 = targets1.cuda()

    output0, output1 = model(inputs)     # "model()" invokes "model.forward()"

    loss0 += criteria[0](output0.view(-1), targets0)
    loss1 += criteria[1](output1.view(-1), targets1)

    if update:
        loss = loss0 + loss1
        loss.backward()
        optimizer.step()

    return loss0.item(), loss1.item(), nPoints

###############################################################################
def train_model(model, train_data, dev_data):
    train_list  = [i for i in range(len(train_data[0]))]    # list of indexes of training data
    dev_list    = [i for i in range(len(dev_data[0]))]      # list of indexes of dev data
    
    criterion0 = nn.MSELoss(reduction='sum')
    criterion1 = nn.MSELoss(reduction='sum')
    criteria = [criterion0, criterion1]

    # choose an optimizer
    # all optimizers from PyTorch with default parameters, except learning_rate, and weight_decay
    ##optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08) # doesn't work with "dense gradients"
    ##optimizer = torch.optim.LBFGS(model.parameters(), lr=100*learning_rate, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, line_search_fn=None) # requires addtional "closure" step during training

    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1*learning_rate, momentum=0, dampening=0, weight_decay=L2_lambda, nesterov=False)
    #optimizer = torch.optim.ASGD(model.parameters(), lr=0.1*learning_rate, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=L2_lambda)

    #optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate, etas=(0.5, 1.2), step_sizes=(1e-06, 50))
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, eps=1e-08, weight_decay=L2_lambda, momentum=0, centered=False)

    #optimizer = torch.optim.Adadelta(model.parameters(), lr=100*learning_rate, rho=0.9, eps=1e-06, weight_decay=L2_lambda)
    #optimizer = torch.optim.Adagrad(model.parameters(), lr=10*learning_rate, lr_decay=0, weight_decay=L2_lambda, initial_accumulator_value=0)#, eps=1e-10)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=L2_lambda, amsgrad=False)
    #optimizer = torch.optim.Adamax(model.parameters(), lr=2*learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=L2_lambda)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=L2_lambda, amsgrad=False)

    ###############################################################################
    # train model 
    ###############################################################################
    start = time.time()

    nEpochs = ((DesiredTrainSamples - 1) // len(train_list)) + 1

    TrainLoss0 = 0
    TrainLoss1 = 0
    TrainPoints = 0
    TotalTrainSamples = 0
    nCurrentSamples = 0

    model.train()
    for epoch in range(nEpochs):
        shuffle(train_list)     # make it a habit to shuffle data, even if random
        #print(train_list)
        for i in range(0, len(train_list), batch_size):
            loss0, loss1, nPoints = train(model, optimizer, criteria, train_data, train_list[i:i+batch_size], update=True)
            #print(i, nTrainPairs, loss / nPoints)
            TrainLoss0 += loss0
            TrainLoss1 += loss1
            TrainPoints += nPoints
            TotalTrainSamples += nPoints
            nCurrentSamples += nPoints

            if (nCurrentSamples >= DisplayInterval) or (TotalTrainSamples >= DesiredTrainSamples):
                model.eval()
                DevLoss0, DevLoss1, DevPoints = train(model, optimizer, criteria, dev_data, dev_list, update=False)
                model.train()
                print("epoch = %5d %8d train = %8d\t%8.4f %8.4f\tdev = %8d\t%8.4f %8.4f" % (epoch, TotalTrainSamples, TrainPoints, TrainLoss0 / TrainPoints, TrainLoss1 / TrainPoints, DevPoints, DevLoss0 / DevPoints, DevLoss1 / DevPoints))

                TrainLoss0 = 0
                TrainLoss1 = 0
                TrainPoints = 0
                nCurrentSamples -= DisplayInterval

            if TotalTrainSamples >= DesiredTrainSamples:
                break

        if TotalTrainSamples >= DesiredTrainSamples:
            break

    torch.save(model, 'mse_02-temp.pth')

    return

###############################################################################
def func0(x):
    y = x*x
    return y

###############################################################################
def func1(x):
    #y = (x+1)*(x-2)
    y = 4*np.sin(x)
    return y

###############################################################################
def create_data(data_spec):
    x_lower, x_upper, nPoints = data_spec
    x = np.linspace(x_lower, x_upper, num=nPoints, endpoint=True).astype(np.float32)
    y0 = func0(x)
    y1 = func1(x)
    return [x, y0, y1]

###############################################################################
if __name__ == "__main__":
    start = time.time()

    ###############################################################################
    # create data
    ###############################################################################
    lower_bound = -5
    upper_bound = +5

    train_data  = create_data([lower_bound, upper_bound, nTrainPoints])
    dev_data    = create_data([lower_bound, upper_bound, nDevPoints])
    test_data   = create_data([lower_bound, upper_bound, nTestPoints])

    ###############################################################################
    # create model 
    ###############################################################################
    model = NN(1, nLinear1, nLinear2, 1, dropout)
    print(model)
    if use_cuda:
        model = model.cuda()

    train_model(model, train_data, dev_data)

    end = time.time()
    elapsed = end - start
    print("time = %.2f (sec)\n" % (elapsed))

    # let's see our neural network in action
    show_results = True
    #show_results = False
    if show_results:
        x, y0, y1 = test_data
        out0, out1 = model(torch.Tensor(x))
        #print(out0)
        out0 = out0.detach().cpu().numpy()[:,0] # remove output values from network, move to cpu, convert to numpy
        out1 = out1.detach().cpu().numpy()[:,0]
        #print(out0)
        #print(out0.shape)
        mse0 = (((y0 - out0)**2).sum()) / len(y0)
        print("mean squared error for test data =", mse0)

        nPlots = 4
        fig = plt.figure(figsize=(20,45))

        plt.subplot(nPlots,1,1)
        plt.plot(x, y0, 'b')
        plt.plot(x, out0, 'r')

        plt.subplot(nPlots,1,2)
        plt.plot(x, y0-y0, 'b')
        plt.plot(x, y0-out0, 'r')
        #plt.plot(train_data[0], train_data[1]-train_data[1], 'o')

        plt.subplot(nPlots,1,3)
        plt.plot(x, y1, 'b')
        plt.plot(x, out1, 'r')

        plt.subplot(nPlots,1,4)
        plt.plot(x, y1-y1, 'b')
        plt.plot(x, y1-out1, 'r')
        #plt.plot(train_data[0], train_data[2]-train_data[2], 'o')

        plt.show()
