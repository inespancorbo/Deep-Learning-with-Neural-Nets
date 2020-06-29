###############################################################################
# ffnn (feed forward neural net) for F0 tracking and phone recognition
###############################################################################
import time, os, sys, random, datetime
import argparse
import soundfile as sf
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

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
SampleRate = 16000
frames_per_second = 200
pts_per_frame = SampleRate // frames_per_second

###############################################################################
# tunable parameters

# for each layer, specify "size, context, dilation"
layer0 = (500, 5, 1)    # b0 
layer0 = (500, 3, 1)

layer1 = (200, 0, 1)    # b0 
layer1 = (500, 1, 3)

layer2 = (100, 0, 1)    # b0 
layer2 = (500, 1, 6)

layer3 = (100, 1, 5)    # b0 
layer3 = (500, 1, 12)

layer4 = (500, 1, 24)

#FFNN_layers = [layer0, layer1, layer2]
#FFNN_layers = [layer0, layer1, layer2, layer3]
FFNN_layers = [layer0, layer1, layer2, layer3, layer4]

#learning_rate = 0.0001
learning_rate = 0.0003  # default
#learning_rate = 0.001
#learning_rate = 0.003

#nEpochs = 1
nEpochs = 2
nEpochs = 4
nEpochs = 10   # default
#nEpochs = 20
#nEpochs = 40
#nEpochs = 100
#nEpochs = 1000

#F0_scale = 20
#F0_scale = 50
F0_scale = 100  # best
#F0_scale = 200

F0_shift = 150
#F0_shift = 180
#F0_shift = 200
#F0_shift = 2000

#dropout = 0.0
#dropout = 0.1   # default
dropout = 0.3
#dropout = 0.5

#L2_lambda = 0.0
#L2_lambda = 0.001
L2_lambda = 0.002

#nperseg = 256
nperseg = 512
nSpecPts = (nperseg // 2) + 1

window_name = ("tukey", 0.25)
window_name = "hamming"
window = signal.get_window(window_name, nperseg)
debias = 0.97
debias = 0.98

###############################################################################
class FFNN(nn.Module):
###############################################################################
    def __init__(self, Cin, layers, nPhones, dropout=0.0):
        super(FFNN, self).__init__()

        self.layers = nn.ModuleList([])
        self.xxx = []
        self.padding = 0
        self.nPhones = nPhones

        for i, layer_specs in enumerate(layers):
            Cout, context, dilation = layer_specs
            kernel_size = 2*context + 1
            self.padding += context * dilation
            #layer = nn.Conv1d(Cin, Cout, kernel_size, stride=1, dilation=dilation, padding=padding)
            layer = nn.Linear(Cin * kernel_size, Cout)
            self.layers.append(layer)
            self.xxx.append((context, dilation))
            Cin = Cout

        self.out0 = nn.Linear(Cin, 1)       # f0 - F0Loss
        self.out1 = nn.Linear(Cin, 1)       # voicing - BCE
        self.out2 = nn.Linear(Cin, nPhones) # phones - CrossEntropy

        #self.non_linear = nn.Sigmoid()
        #self.non_linear = nn.Tanh()

        #self.non_linear = nn.Tanhshrink()
        #self.non_linear = nn.Hardshrink(lambd=0.5)
        #self.non_linear = nn.Softshrink(lambd=0.5)

        #self.non_linear = nn.ELU()
        #self.non_linear = nn.SELU()
        #self.non_linear = nn.ReLU()
        self.non_linear = nn.LeakyReLU(negative_slope=0.01)
        
        self.dropout = nn.Dropout(dropout)

        for p in self.parameters(): # optionally apply different randomization
            if p.dim() > 1:
                #nn.init.normal_(p)
                #nn.init.xavier_uniform_(p)
                #nn.init.xavier_normal_(p)
                nn.init.kaiming_normal_(p)
                pass

    #################################################################
    def forward(self, spec):
        nBatch = 1
        nFrames = len(spec)
        data = np.zeros([(2*self.padding+nFrames), nSpecPts], dtype=np.float32)
        data[:,:] = -15.0
        pad = self.padding
        data[pad:pad+nFrames, :] = spec
        data = cuda(torch.from_numpy(data))

        prev = data.view(-1, nBatch, nSpecPts)

        for i in range(len(self.layers)):
            context, dilation = self.xxx[i]
            l = len(prev) - 2*context*dilation
            blocks = []
            for b in range(2*context+1):
                start = b*dilation
                blocks.append(prev[start:start+l])

            prev = tuple(blocks)
            prev = torch.cat(prev, 2)   # 2 is axis

            layer = self.layers[i]
            prev = layer(prev)
            prev = self.non_linear(prev)
            prev = self.dropout(prev)

        out0 = self.out0(prev).view(-1)  # f0
        out1 = self.out1(prev).view(-1)  # voicing
        out2 = self.out2(prev).view(-1, self.nPhones)  # phones

        return out0, out1, out2

###############################################################################
def FFNN_train(model, optimizer, criteria, datafiles, idf_list, update=True):
    model.zero_grad()
    loss0 = 0
    loss1 = 0
    loss2 = 0
    #loss3 = 0
    nBatchFrames = 0
    nF0Frames = 0

    for idf in idf_list:
        df = datafiles[idf]
        f0_ = cuda(Variable(torch.from_numpy((df.f0 - F0_shift) / F0_scale)))
        voicing_ = cuda(Variable(torch.from_numpy(df.voicing)))
        mask = cuda(Variable(torch.from_numpy(df.F0Frames)))

        out0, out1, out2 = model(df.spec)

        loss0 += criteria[0](out0, f0_, mask)
        loss1 += criteria[1](out1, voicing_)
        if not df.phones is None:
            phones = cuda(Variable(torch.from_numpy(df.phones)))
            loss2 += criteria[2](out2, phones)
            #loss3 += criteria[2](out3, phones)

        nBatchFrames += df.nFrames
        nF0Frames += df.nF0Frames

    if loss0 != 0:
        loss0_v = loss0.data.item()
    else:
        loss0_v = 0
    if loss1 != 0:
        loss1_v = loss1.data.item()
    else:
        loss1_v = 0
    if loss2 != 0:
        loss2_v = loss2.data.item()
        #loss3_v = loss3.data.item()
    else:
        loss2_v = 0
        #loss3_v = 0

    if update:
        do_weights = True
        #do_weights = False
        if do_weights:
            """
            alpha = 0.5
            alpha = 1.0
            z0 = 1/(loss0_v + alpha)
            z1 = 1/(loss1_v + alpha)
            z2 = 1/(loss2_v + alpha)
            z3 = 1/(loss3_v + alpha)
            s = (z0 + z1 + z2) / 3.0
            w0 = z0 / s
            w1 = z1 / s
            w2 = z2 / s
            w3 = z3 / s
            """
            w0 = 100.0
            w1 = 1.0
            w2 = 1.0
            #w3 = w2
            loss0 *= w0
            loss1 *= w1
            loss2 *= w2
            #loss3 *= w3

        loss = 0
        loss += loss0
        loss += loss1
        loss += loss2
        #loss += loss3

        if not loss is 0:
            loss.backward()
            optimizer.step()

    if nF0Frames > 0:
        loss0 *= (nBatchFrames / nF0Frames)

    return [loss0_v, loss1_v, loss2_v], nBatchFrames
    #return [loss0_v, loss3_v, loss2_v], nBatchFrames

###############################################################################
def F0Loss(preds, targs, mask):
    preds = preds.view(-1)          # reshape to give a flat vector of length batch_size*seq_len
    targs = targs.view(-1)
    mask = mask.view(-1)

    errors = ((preds - targs)**2) * mask                # only voiced frames count towards error

    return torch.sum(errors)

###############################################################################
def train_f0_model(model, datafiles):
    if use_cuda:
        model = model.cuda()

    # define the loss functions
    criterion0 = F0Loss                                 # F0
    criterion1 = nn.BCEWithLogitsLoss(reduction='sum')  # voicing
    criterion2 = nn.CrossEntropyLoss(reduction='sum')   # phones
    criteria = [criterion0, criterion1, criterion2]

    # choose an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=L2_lambda)

    data_deck = [i for i in range(len(datafiles))]
    random.shuffle(data_deck)
    dev_rate = 0.10 # 10% for dev
    #dev_rate = 0.20 # 20% for dev
    dev_deck = data_deck[:int(dev_rate * len(data_deck))]
    train_deck = data_deck[len(dev_deck):]
    print(dev_deck)
    print(train_deck)

    batch_size = 1
    results = []

    start = time.time()
    for e in range(nEpochs):
        random.shuffle(train_deck)
        train_frames = 0
        train_loss0 = 0
        train_loss1 = 0
        train_loss2 = 0

        model.train()
        for i in range(0, len(train_deck), batch_size):
            loss, nBatchFrames = FFNN_train(model, optimizer, criteria, datafiles, train_deck[i:i+batch_size], update=True)

            nFrames = nBatchFrames
            train_frames += nFrames
            train_loss0 += loss[0]
            train_loss1 += loss[1]
            train_loss2 += loss[2]
            #print("%2d %6d %8.3f %8.3f %s" % (e, i, loss[0]/nFrames, loss[1]/nFrames, datafiles[train_deck[i]].name))
            print("%2d %6d %8.3f %8.3f %8.3f %s" % (e, i, loss[0]/nFrames, loss[1]/nFrames, loss[2]/nFrames, datafiles[train_deck[i]].name))
        
        dev_frames = 0
        dev_loss0 = 0
        dev_loss1 = 0
        dev_loss2 = 0

        model.eval()
        for idf in range(len(dev_deck)):
            loss, nBatchFrames = FFNN_train(model, optimizer, criteria, datafiles, dev_deck[idf:idf+1], update=False)

            nFrames = nBatchFrames
            dev_frames += nFrames
            dev_loss0 += loss[0]
            dev_loss1 += loss[1]
            dev_loss2 += loss[2]

        if dev_frames == 0: dev_frames = 1
        results.append(F0_scale * np.sqrt(dev_loss0/dev_frames))
        loss_window = 10
        loss_window =  5
        s = sorted(results[-loss_window:])
        score = sum(s[:3]) / min(3, len(s)) # average of best 3 from previous 5 epochs
        log_message(log_file, "%3d %8d %8.3f %8.3f %8.3f  %8d %8.3f %8.3f %8.3f    %6.1f\t%6.3f" % (e, train_frames, F0_scale * np.sqrt(train_loss0/train_frames), train_loss1/train_frames, train_loss2/train_frames, dev_frames, F0_scale * np.sqrt(dev_loss0/dev_frames), dev_loss1/dev_frames, dev_loss2/dev_frames, (time.time()-start), score))

    log_message(log_file, "%s\t%6.4f\t%f\t%f\t%f\t%f\t%f" % (str(FFNN_layers), learning_rate, F0_scale, F0_shift, dropout, batch_size, L2_lambda))

    torch.save(model, 'model/FFNN-temp.pth')

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
class DataFile:
    def __init__(self, name):
        self.name     = name
        self.nFrames  = None
        self.nF0Frames = None

        self.wav      = None
        self.spec     = None
        self.phones   = None
        self.f0       = None
        self.voicing  = None
        self.F0Frames = None

###############################################################################
def read_phone_list(ffn):
    fin = open(ffn, "r")
    phone_dict = dict()
    phone_list = []
    for line in fin:
        line = line.strip()
        if len(line) == 0 or line[0] == ';': continue
        if line[0] == '#': break

        values = line.strip().split()
        if len(values) != 1:
            print("ERROR: bad line in phone list:", line)
            continue

        ph = values[0]
        if ph in phone_dict:
            print("ERROR: duplicate phone:", ph)
            continue

        phone_dict[ph] = len(phone_dict)
        phone_list.append(ph)

    return phone_dict, phone_list

###############################################################################
def read_nali_file(ffn, nFrames, fps):
    if phone_dict is None:
        return None

    try:
        fin = open(ffn, "r")
    except:
        return None

    first = True
    phones = np.zeros(nFrames, dtype="long")
    prev_fr = 0
    prev_ph = 0

    for line in fin:
        line = line.strip()
        if (len(line) == 0) or (line[0] == ';'):
            continue
        if (line[0] == '#'):
            break

        values = line.strip().split()
        if first:
            t = float(values[0])
            first = False
        else:
            if line[0] == '*':
                continue
            else:
                t = float(values[0])
                fr = round(t * fps)
                phones[prev_fr:fr] = prev_ph

                prev_fr = fr
                prev_ph = phone_dict[values[1]]

    phones[prev_fr:nFrames] = prev_ph

    return phones

###############################################################################
def read_datafile(fn, path_wav, path_f0=None, ext=".wav", path_ph=None):
    attempt = ""
    try:
        attempt = "read_speech"
        ffn = path_wav + fn + ext
        data, sample_rate = sf.read(ffn) # sf.read returns data as float32 scaled to [-1.0 .. +1]
        nFrames = len(data) // pts_per_frame
        if sample_rate != SampleRate:
            raise ValueError("sample_rate (%d) != SampleRate (%d)" % (sample_rate, SampleRate))

        if path_f0 != None:
            attempt = "read_f0"
            ffn = path_f0 + fn + ".f0"
            f0_voicing = np.loadtxt(ffn, dtype="float32")
            f0 = f0_voicing[:,0].astype(np.float32)
            voicing = f0_voicing[:,1].astype(np.float32)
            F0Frames = (f0 > 0).astype(np.float32) * (voicing > 0.9).astype(np.float32)
            nF0Frames = int(F0Frames.sum())
            if nFrames != len(f0):
                raise ValueError("wav frames (%d) != f0 frames (%d)" % (nFrames, len(f0)))
        else:
            f0 = None
            voicing = None
            F0Frames = None
            nF0Frames = None

        do_phones = True
        #do_phones = False
        if path_ph != None and do_phones:
            attempt = "read_nali"
            ffn = path_ph + fn + ".nali"
            phones = read_nali_file(ffn, nFrames, 200)
        else:
            phones = None

        attempt = "create df"
        df = DataFile(fn)
        df.nFrames = nFrames
        df.wav = data.astype(np.float32)

        df.f0 = f0
        df.voicing = voicing
        df.F0Frames = F0Frames
        df.nF0Frames = nF0Frames
        df.phones = phones

        return df
    except:
        #raise
        print(fn, "error reading file components", attempt, ffn)
        return None

###############################################################################
def read_datafiles(fn_lst, path_wav, path_f0=None, path_ph=None, datafiles=None, noisy=2, ext=".wav"):
    if datafiles is None:
        datafiles = []

    fin = open(fn_lst, "r")
    nTotalFrames = 0
    nFiles  = 0
    nErrorFiles = 0

    for line in fin:
        line = line.strip()
        if len(line) == 0:
            continue
        if (line[0] == ';'):
            continue
        if (line[0] == '#'):
            break

        values = line.strip().split()
        df = read_datafile(values[0], path_wav, path_f0=path_f0, path_ph=path_ph, ext=ext)
        if df != None:
            datafiles.append(df)
            if df.nFrames:
                nTotalFrames += df.nFrames
            nFiles += 1

            if noisy > 1:
                print(nFiles, df.nFrames, df.nF0Frames, df.name)
        else:
            nErrorFiles += 1

    if noisy > 0:
        log_message(log_file, "%d files with %d frames --> %0.1f frames/files in %s" % (nFiles, nTotalFrames, nTotalFrames / nFiles, fn_lst))
    if nErrorFiles > 0:
        print(nErrorFiles, "files with errors")
    fin.close()
    return datafiles

###############################################################################
def get_phone_diff(ffn, ph_a, ph_b):
    if len(ph_a) != len(ph_b):
        print("len(ph_a) %d != len(ph_b) %d" % (len(ph_a), len(ph_b)))
        return

    os.makedirs(os.path.dirname(ffn), exist_ok=True) # ensure output directory exists
    out = open(ffn, "w")
    for i in range(len(ph_a)):
        out.write("%d\t%s\t%s" % (i, phone_list[ph_a[i]], phone_list[ph_b[i]]))
        out.write("\n")
        
###############################################################################
def get_f0_diff(ffn, f0_a, f0_b):
    if len(f0_a) != len(f0_b):
        print("len(f0_a) %d != len(f0_b) %d" % (len(f0_a), len(f0_b)))
        return

    os.makedirs(os.path.dirname(ffn), exist_ok=True) # ensure output directory exists
    out = open(ffn, "w")
    for i in range(len(f0_a)):
        out.write("%d\t%.1f\t%.1f" % (i, f0_a[i], f0_b[i]))
        out.write("\n")
        
###############################################################################
def get_model_outputs(df, model):
    if df.spec is None:
        get_spec_for_file(df)

    model.eval()
    f0_out, voicing_out, phone_out = model(df.spec)

    sm = nn.Sigmoid()
    voicing = sm(voicing_out).data.cpu().numpy()
    mask = (voicing > 0.05).astype(np.float32)

    f0 = (f0_out.data.cpu().numpy() * F0_scale + F0_shift) * mask

    phones = np.argmax(phone_out.data.cpu().numpy(), axis=1)
    #print(phones)
    #print(df.phones)

    df.f0_pred = f0
    df.voicing_pred = voicing
    df.ph_pred = phones

###############################################################################
def save_file_f0(df, path_out):
    ffn = path_out + df.name + ".f0"
    os.makedirs(os.path.dirname(ffn), exist_ok=True) # ensure output directory exists
    out = open(ffn, "w")

    for i in range(df.nFrames):
        out.write("%.1f\t%.1f\n" % (df.f0_pred[i], df.voicing_pred[i]))

###############################################################################
def save_file_ph(df, path_out):
    ffn = path_out + df.name + ".ph"
    os.makedirs(os.path.dirname(ffn), exist_ok=True) # ensure output directory exists
    out = open(ffn, "w")

    for i in range(df.nFrames):
        out.write("%s\n" % (phone_list[df.ph_pred[i]]))

###############################################################################
def get_spec_for_file(df):
    pad = (nperseg - pts_per_frame) // 2

    wav = np.zeros(pts_per_frame * df.nFrames + 2*pad, dtype="float32")
    wav[pad:len(df.wav)+pad] = signal.lfilter([1, -1], [1, -debias], df.wav)
    #wav[pad:len(df.wav)+pad] = df.wav # don't debias wav
    f, t, spc = signal.spectrogram(wav, window=window, noverlap=(nperseg-pts_per_frame), detrend=None)
    spc = np.moveaxis(spc, 0, 1)

    if len(spc) != df.nFrames: # this should never happen
        print("len(spc) %d != df.nFrames %d" % (len(spc), df.nFrames))
        exit()
    
    df.spec = spc

###############################################################################
def get_spec(datafiles):
    for df in datafiles:
        get_spec_for_file(df)

###############################################################################
# start main
###############################################################################
if __name__ == "__main__":
    # process command line args
    parser = argparse.ArgumentParser(description='get F0, voicing, and phones for a speech file.')

    # major function
    parser.add_argument('-model', default='None', help='the model to be loaded')
    parser.add_argument('-train', action='store_const', const=True, default=False, help='train a model')
    parser.add_argument('-phones', default='./data/phones.lst', help='list of phones for the classifier')

    # inputs
    parser.add_argument('-list', action='append', default=[], help='list of files to read')
    parser.add_argument('-file', action='append', default=[], help='name of file to read -- can use both -list & -file')

    parser.add_argument('-path', default=None, help='path of wav files')
    parser.add_argument('-path-f0', default=None, help='path of f0 files, default = -path')
    parser.add_argument('-path-ph', default=None, help='path of ph files, default = -path')

    # outputs
    parser.add_argument('-save-f0', default=None, help='path of output .f0 files, default = None')
    parser.add_argument('-save-ph', default=None, help='path of output .ph files, default = None')
    #parser.add_argument('-save-model', default='./model/FFNN-temp.pth', help='full name of saved model file')

    # other actions
    parser.add_argument('-log', default='./log/FFNN-temp.log', help='name of log file')
    parser.add_argument('-noisy', default=2, type=int, help='level of reporting')
    #-score-f0
    #-score-ph

    args = parser.parse_args()
    print("args =",args)
    global log_file
    os.makedirs(os.path.dirname(args.log), exist_ok=True) # ensure output directory exists
    log_file = open(args.log, "a")
    log_message(log_file, "\nstarting run: %s %s" % (datetime.date.today(), datetime.datetime.now()))
    #log_message(log_file, "args = %s" % (str(args)))

    if args.path is None:
        log_message(log_file, "ERROR: must specify path for wav files")
        exit()
    #if args.path_f0 is None: args.path_f0 = args.path
    #if args.path_ph is None: args.path_ph = args.path

    global phone_dict
    global phone_list
    phone_dict, phone_list = read_phone_list(args.phones)
    #print(phone_list)

    if args.model == 'None':
        if not args.train:
            print("ERROR: must specify -train if you are not loading a pre-trained -model")
            exit()
        model = FFNN(nSpecPts, FFNN_layers, len(phone_dict), dropout)
    else:
        if use_cuda:
            model = torch.load(args.model)
            model = model.cuda()
        else:
            model = torch.load(args.model, map_location='cpu')

        print("model loaded: %s" % (args.model))

    print(model)

    start = time.time()
    datafiles = []
    ext = '.wav'

    for fn in args.list:
        datafiles = read_datafiles(fn, args.path, path_f0=args.path_f0, path_ph=args.path_ph, datafiles=datafiles, ext=ext, noisy=args.noisy)

    for fn in args.file:
        df = read_datafile(fn, args.path, path_f0=args.path_f0, path_ph=args.path_ph, ext=ext)
        if df != None:
            datafiles.append(df)

    elapsed_read =  (time.time() - start)
    start = time.time()

    get_spec(datafiles)

    elapsed_spec =  (time.time() - start)
    log_message(log_file, "time to read files, get spec, #files = %.1f %.1f %d" % (elapsed_read, elapsed_spec, len(datafiles)))

    if args.train:
        #start = time.time()
        print("training")
        train_f0_model(model, datafiles)

    if True:
        start = time.time()
        nTotalFrames = 0

        for df in datafiles:
            get_model_outputs(df, model)
            nTotalFrames += df.nFrames

        elapsed =  (time.time() - start)
        log_message(log_file, "elapsed time for pitch tracking and phone recognition = %0.2f, %d files with %d frames --> x real time = %f" % (elapsed, len(datafiles), nTotalFrames, (nTotalFrames / frames_per_second) / elapsed)) 

        if not args.save_f0 is None:
            for df in datafiles:
                try:
                    save_file_f0(df, args.save_f0)
                except:
                    #raise
                    print("ERROR: could not save file .f0 for", df.name)

        if not args.save_ph is None:
            for df in datafiles:
                try:
                    save_file_ph(df, args.save_ph)
                except:
                    #raise
                    print("ERROR: could not save file .ph for", df.name)

    #get_phone_diff("./temp.dp", df.phones, df.phones_pred)
    #get_f0_diff("./temp.f0", df.f0, df.f0_pred)
