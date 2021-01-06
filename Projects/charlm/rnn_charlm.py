###############################################################################
# rnn (recurrent neural net) character language model
###############################################################################
import time, os, sys, random, datetime
import argparse
import numpy as np
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn

###############################################################################
use_cuda = torch.cuda.is_available()
#use_cuda = True
#use_cuda = False
print(torch.version.__version__, use_cuda)
###############################################################################
def cuda(arr):
    if use_cuda:
        return arr.cuda()
    return arr

###############################################################################
global log_file
global CharDict
global CharList

CharDict = dict()
CharList = []

###############################################################################
# tunable hyper-parameters - default values
###############################################################################
# these values are used to create the model
###############################################################################
# character embedding
char_embed_size = 20

# recurrent layers
rnn_size = 200
#rnn_size = 400
rnn_nLayers = 2

RNN_layers = [rnn_size, rnn_nLayers]

# feed-forward layers
layer0 = 100

layer1 = 100

layer2 = 100

layer3 = 100

layer4 = 100

FFNN_layers = [layer0, layer1, layer2]
#FFNN_layers = [layer0, layer1, layer2, layer3]
#FFNN_layers = [layer0, layer1, layer2, layer3, layer4]

dropout = 0.0
dropout = 0.1
#dropout = 0.3
#dropout = 0.5

###############################################################################
# these values are used within the training code
###############################################################################
global learning_rate
global batch_size
global chunk_size
global nEpochs
global L2_lambda

#learning_rate = 0.0001
learning_rate = 0.0003
#learning_rate = 0.001
#learning_rate = 0.003

batch_size = 5
batch_size = 10
batch_size = 20

chunk_size = 50
chunk_size = 100
chunk_size = 200

#nEpochs = 1
nEpochs = 2
#nEpochs = 4
nEpochs = 10
#nEpochs = 20
#nEpochs = 100
#nEpochs = 200
#nEpochs = 500
#nEpochs = 1000

L2_lambda = 0.0
#L2_lambda = 0.001
#L2_lambda = 0.002

###############################################################################
class RNN(nn.Module):
###############################################################################
    def __init__(self, specs):
        super(RNN, self).__init__()

        nChars, embed_size, rnn_layers, ffnn_layers, dropout = specs
        self.CharEmbed = nn.Embedding(nChars, embed_size)

        rnn_size, rnn_nLayers = rnn_layers
        self.rnn = nn.GRU(embed_size, rnn_size, rnn_nLayers, dropout=dropout, batch_first=True)

        self.layers = nn.ModuleList([])
        prev_size = rnn_size
        for i, layer_size in enumerate(ffnn_layers):
            layer = nn.Linear(prev_size, layer_size)
            self.layers.append(layer)
            prev_size = layer_size

        self.out = nn.Linear(prev_size, nChars) # character - CrossEntropy

        self.non_linear = nn.LeakyReLU(negative_slope=0.01)
        
        self.dropout = nn.Dropout(dropout)

        for p in self.parameters(): # optionally apply different randomization
            if p.dim() > 1:
                #nn.init.xavier_uniform_(p)
                #nn.init.xavier_normal_(p)
                nn.init.kaiming_normal_(p)
                pass

    #################################################################
    def forward(self, seqs, hidden=None):
        # input is a list of sequences of torch longs already on cuda as needed
        nBatch = len(seqs)
        nChars = len(seqs[0])

        #print(seqs)
        seqs = torch.cat(seqs).view(nBatch, nChars)
        embed = self.CharEmbed(seqs)

        prev, hidden = self.rnn(embed, hidden)

        for layer in self.layers:
            prev = layer(prev)
            prev = self.non_linear(prev)
            prev = self.dropout(prev)

        out = self.out(prev) # chars
        
        #hidden = torch.transpose(hidden, 0, 1)
        return out, hidden

###############################################################################
def RNN_train(model, optimizer, criterion, chunks, update=True):
    model.zero_grad()
    loss = 0
    nFrames = 0

    out, hidden = model(chunks)
    #print(out.shape)

    skip = 5
    nBatch = len(chunks)   
    for i in range(nBatch):
        loss += criterion(out[i][skip-1:-1,:], chunks[i][skip:])
        nFrames += len(chunks[i]) - skip

    if update:
        if not loss is 0:
            loss.backward()
            optimizer.step()

    return loss.data.item(), nFrames

###############################################################################
def train_rnn_model(model, data_train, data_dev=None):
    if use_cuda:
        model = model.cuda()

    # define the loss functions
    criterion = nn.CrossEntropyLoss(reduction='sum')

    # choose an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)

    data_len = len(data_train) - chunk_size
    nLoops = 1 + len(data_train) // (batch_size * chunk_size)

    results = []

    start_time = time.time()
    for e in range(nEpochs):
        train_frames = 0
        train_loss = 0

        model.train()
        for i in range(nLoops):
            chunks = []
            for j in range(batch_size):
                start = np.random.randint(data_len)
                chunks.append(data_train[start:start+chunk_size])

            loss, nFrames = RNN_train(model, optimizer, criterion, chunks, update=True)

            train_frames += nFrames
            train_loss += loss
            print("%2d %6d %8.3f" % (e, (i+1)*batch_size*chunk_size, loss/nFrames))

        print(train_frames, train_loss, train_loss / train_frames)
        sample = generate(model, seed="The ", n=100)
        print(sample)

        dev_frames = 0
        dev_loss = 0
        if not data_dev is None:
            model.eval()
            chunks = []
            for start in range(0, len(data_dev)-chunk_size, chunk_size):
                chunks.append(data_dev[start:start+chunk_size])

            dev_loss, dev_frames = RNN_train(model, optimizer, criterion, chunks, update=False)

        if dev_frames == 0: dev_frames = 1
        log_message(log_file, "%3d %8d %8.3f\t%8d %8.3f\t%6.1f" % (e, train_frames, train_loss/train_frames, dev_frames, dev_loss/dev_frames, (time.time()-start_time)))

    torch.save(model, 'model/charlm-temp.pth')

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
# NOTE: will update this to store legal characters in a file, and read the file
# so to use this for other languages
###############################################################################
legal_chars = ['§', '\t', '\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '®', '°']
convert_chars = [['‑', '-'], ['–', '-'], ['—', '-'], ['‘', "'"], ['’', "'"], ['‚', ','], ['“', '"'], ['”', '"']]

###############################################################################
class Char:
    def __init__(self, char):
        self.char = char
        self.ndx = None
        self.type = None
        self.count = 0

###############################################################################
def create_char_dict():
    # you should read these chars from a file
    for c in legal_chars:
        if not c in CharDict:
            char = Char(c)
            char.ndx = len(CharList)
            char.type = 'legal'
            CharDict[c] = char
            CharList.append(char)

    for (f, t) in convert_chars:
        if not f in CharDict:
            char = Char(f)
            char.ndx = CharDict[t].ndx
            char.type = 'convert'
            CharDict[f] = char

###############################################################################
def add_illegal_chars(data):
    noisy = True
    noisy = False
    start = time.time()

    CharSet = sorted(set(data))
    for c in CharSet:
        if not c in CharDict:
            char = Char(c)
            char.ndx = 0
            char.type = 'bad'
            CharDict[c] = char
            if noisy:
                count = data.count(c)
                print(c, count)

    print("%8.1f = time to create char dict (%d, %d, %d)" % (time.time() - start, len(CharSet), len(CharDict), len(CharList)))

###############################################################################
# NOTE: this code is slow, need to change this at some point
###############################################################################
def convert_data(data, CharDict, quit_early=0):
    start = time.time()
    if quit_early > 0:
        data = data[:quit_early]
    ndx_data = torch.zeros(len(data), dtype=torch.long)
    for i, c in enumerate(data):
        #print(i, c)
        ndx_data[i] = CharDict[c].ndx

    elapsed = time.time() - start
    print("%8.1f = time to convert %d chars = %8.1f chars/sec" % (elapsed, len(data), len(data) / elapsed))

    return ndx_data

###############################################################################
def read_text_file(fn):
    fin = open(fn, 'r')
    data = fin.read()
    return data

###############################################################################
def generate(model, seed="The ", n=100):
    model.eval()
    ndx_data = cuda(convert_data(seed, CharDict))
    c, h = model([ndx_data])
    text = list(seed)
    for i in range(n):
        scores = c[0,-1]
        _, best = scores.max(0)
        best = best.data.item()
        text.append(CharList[best].char)
        c_in = cuda(torch.LongTensor([best]))
        c, h = model([c_in], h)

    return ''.join(text)

###############################################################################
# start main
###############################################################################
if __name__ == "__main__":
    # process command line args
    parser = argparse.ArgumentParser(description='create a character-based language model')

    # major function
    parser.add_argument('-model', default='None', help='the model to be loaded') # not implemented

    # inputs
    parser.add_argument('-train', default=None, required=True, help='training file')
    parser.add_argument('-test', default=None, help='test file')
    parser.add_argument('-dev', default=None, help='dev file')

    # hyper-parameters
    parser.add_argument('-lr', default=learning_rate, type=float, help='learning rate')
    parser.add_argument('-epochs', default=nEpochs, type=int, help='number of epochs')
    parser.add_argument('-batch', default=batch_size, type=int, help='batch size')
    parser.add_argument('-chunk', default=chunk_size, type=int, help='chunk size')
    parser.add_argument('-L2', default=L2_lambda, type=float, help='L2 regularization')
    parser.add_argument('-dropout', default=dropout, type=float, help='dropout')

    parser.add_argument('-embed', default=char_embed_size, type=int, help='char embed size')
    parser.add_argument('-rnn', default=str(RNN_layers), type=str, help='RNN layers; [rnn_size,rnn_nLayers]')
    parser.add_argument('-ffnn', default=str(FFNN_layers), type=str, help='FFNN; [size0,size1,...]')

    # other actions
    parser.add_argument('-log', default='./log/charlm-temp.log', help='name of log file')
    parser.add_argument('-noisy', default=2, type=int, help='level of reporting')

    args = parser.parse_args()
    print("args =",args)

    nEpochs = args.epochs
    learning_rate = args.lr
    dropout = args.dropout
    L2_lambda = args.L2
    batch_size = args.batch
    chunk_size = args.chunk

    char_embed_size = args.embed
    RNN_layers = eval(args.rnn)
    FFNN_layers = eval(args.ffnn)

    os.makedirs(os.path.dirname(args.log), exist_ok=True) # ensure output directory exists
    log_file = open(args.log, "a")
    log_message(log_file, "\nstarting run: %s %s" % (datetime.date.today(), datetime.datetime.now()))
    #log_message(log_file, "args = %s" % (str(args)))

    create_char_dict()

    data_train = read_text_file(args.train)
    add_illegal_chars(data_train)

    ndx_data_train = convert_data(data_train, CharDict)
    ndx_data_train = cuda(ndx_data_train)

    if not args.dev is None:
        data_dev = read_text_file(args.dev)
        add_illegal_chars(data_dev)
        ndx_data_dev = convert_data(data_dev, CharDict)
        ndx_data_dev = cuda(ndx_data_dev)

    else:
        data_dev = None
        ndx_data_dev = None

    nChars = len(CharList) # NOTE: CharList only contains 'legal' characters
    specs = [nChars, char_embed_size, RNN_layers, FFNN_layers, dropout]
    model = RNN(specs)
    log_message(log_file, "%s\t%f\t%f\t%d\t%d" % (str(specs), learning_rate, L2_lambda, batch_size, chunk_size))

    train_rnn_model(model, ndx_data_train, data_dev=ndx_data_dev)
    print(model)

