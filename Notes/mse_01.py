###############################################################################
# PyTorch workshop examples - MSELoss (Mean Squared Error)
###############################################################################
import time
from random import shuffle

import torch
import torch.nn as nn
from torch.autograd import Variable

###################################################################################################
use_cuda = torch.cuda.is_available()
use_cuda = False

###############################################################################
# modifiable network parameters
###############################################################################
nLinear1 = 100

nLinear2 = 100

learning_rate = 0.001

###############################################################################
class NN(nn.Module):
###############################################################################
    def __init__(self, input_size, nLinear1, nLinear2, output_size):
        super(NN, self).__init__()
        self.linear0 = nn.Linear(input_size, nLinear1)
        self.linear1 = nn.Linear(nLinear1, nLinear2)
        self.linear2 = nn.Linear(nLinear2, output_size)

        #self.non_linear = nn.Sigmoid()
        #self.non_linear = nn.Tanh()
        #self.non_linear = nn.ReLU()
        #self.non_linear = nn.ELU()
        self.non_linear = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, inputs):
        nPoints = len(inputs)
        x = inputs.view(nPoints, -1)
        x = self.linear0(x)
        x = self.non_linear(x)
        x = self.linear1(x)
        x = self.non_linear(x)
        x = self.linear2(x)
        return x

###############################################################################
def train(model, optimizer, criterion, input_list, target_list, idx_list, update=True):
    model.zero_grad()

    nPoints = len(idx_list)
    loss = 0
    inputs = []
    targets = []

    for i in idx_list:
        inputs.append(input_list[i])
        targets.append(target_list[i])

    inputs  = Variable(torch.Tensor(inputs))
    targets = Variable(torch.Tensor(targets))

    if use_cuda:
        inputs = inputs.cuda()
        targets = targets.cuda()

    outputs = model(inputs)     # "model" invokes "model.forward"
    outputs = outputs.view(-1)  # convert shape of ouputs to match targets
    loss += criterion(outputs, targets)

    if update:
        loss.backward()
        optimizer.step()

    return loss.item(), nPoints

###############################################################################
def create_data(nItems):
    a = (torch.rand(nItems) - 0.5) * 10
    a2 = [(x*x) for x in a]
    a2 = torch.Tensor(a2)
    return a, a2

###############################################################################
if __name__ == "__main__":
    start = time.time()

    ###############################################################################
    # create data
    ###############################################################################
    train_input_list, train_target_list = create_data(5000)
    dev_input_list, dev_target_list     = create_data( 100)
    test_input_list, test_target_list   = create_data( 100)
    print(len(train_input_list), len(dev_input_list), len(test_input_list))

    nTrainPairs = len(train_input_list)
    train_list  = [i for i in range(nTrainPairs)]           # list of indexes of training data
    dev_list    = [i for i in range(len(dev_input_list))]   # list of indexes of dev data
    test_list   = [i for i in range(len(test_input_list))]  # list of indexes of test data
    
    ###############################################################################
    # create model 
    ###############################################################################
    model = NN(1, nLinear1, nLinear2, 1)
    if use_cuda:
        model = model.cuda()

    criterion = nn.MSELoss(reduction='sum')                    # mean squar error

    # choose an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    ###############################################################################
    # train model 
    ###############################################################################
    batch_size = 100
    batch_size = 10
    #batch_size = 1
    nEpochs = 10
    start = time.time()

    for epoch in range(nEpochs):
        shuffle(train_list)     # make it a habit to shuffle data, even if random
        #print(train_list)
        TrainLoss = 0
        TrainPoints = 0

        for i in range(0, nTrainPairs, batch_size):
            loss, nPoints = train(model, optimizer, criterion, train_input_list, train_target_list, train_list[i:i+batch_size])
            #print(i, nTrainPairs, loss / nPoints)
            TrainLoss += loss
            TrainPoints += nPoints

        update=False
        #update=True    # update=True is useful for debugging
        DevLoss, DevPoints = train(model, optimizer, criterion, dev_input_list, dev_target_list, dev_list, update=update)
        print("epoch = %2d train = %8d\t%.4f\tdev = %8d\t%.4f" % (epoch, TrainPoints, TrainLoss / TrainPoints, DevPoints, DevLoss / DevPoints))

    TestLoss, TestPoints = train(model, optimizer, criterion, test_input_list, test_target_list, test_list, update=update)
    print("\ntest  = %8d\t%.4f" % (TestPoints, TestLoss / TestPoints))

    test_outputs = model(test_input_list)
    total_error = 0
    for i in range(len(test_input_list)):
        x = test_input_list[i]
        x2 = x*x
        y = test_outputs.data[i]
        total_error += (x2 - y)**2

    print("total_error =", total_error / len(test_input_list))

    end = time.time()
    elapsed = end - start
    print("time = %.2f (sec)\n" % (elapsed))

    test_values = Variable(torch.Tensor([0,-1,2,-3,4,-5,6,-10]))
    if use_cuda:
        test_values = test_values.cuda()

    # let's see our neural network in action
    test_outputs = model(test_values)
    for i in range(len(test_values)):
        #print(test_values[i], test_values[i]*test_values[i], test_outputs[i])
        #x = test_values[i].data[0]
        x = test_values.data[i]
        x2 = x*x
        y = test_outputs.data[i]
        print("%8.2f %8.2f %10.4f %10.4f" % (x, x2, y, (x2-y)**2))
