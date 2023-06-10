import os
import tqdm
import torch
import datetime
import argparse
import pickle as pkl
import pandas as pd
import numpy as np
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data
from torch_geometric.typing import OptPairTensor, Adj
from haversine import haversine_vector
from statistics import pstdev
from dates import dates_usa

def parse_args():
    parser = argparse.ArgumentParser()
    # mode = 2: Missing Data (for ABC mode), 3: Interpolation (for AC or C mode)
    parser.add_argument("--mode",type=int,default= [2,3][1])
    parser.add_argument("--ntraindays",type=int,default=3)
    parser.add_argument("--datadir",type=str,default=None)
    parser.add_argument("--num_sensors",type=int,default=20)
    parser.add_argument("--train_start_day",type=int,default=12)
    parser.add_argument("--date",type=str,default=None)
    parser.add_argument("--mfile",type=str,default=None)
    parser.add_argument("--runs",type=int,default=3)
    parser.add_argument("--fold",type=int,default=0)
    parser.add_argument("--set",type=int,default=0)
    parser.add_argument("--num_same_epochs",type=int,default=2)
    parser.add_argument("--tol",type=float,default=[1e-2,1][0])
    parser.add_argument("--canada",type=int,default=0)
    parser.add_argument("--spatiotemporal",type=int,default=1)
    parser.add_argument("--r2b",type=float,default=[0.5,0.75][0])
    args = parser.parse_args()
    return args
args = parse_args()

mode = args.mode
bUseTestDayInTrain = 1 if mode == 3 else 0
canada = args.canada
mytqdm = tqdm.tqdm
maxFolds = 5
pm = ['pm10','pm2_5'][1]
nGraphs = 1
suffix = 'tmp'
km = 1 / 2
grouptime = 30
edgetime = 120
trainloss_train_validation = 3  # 1: Val, 2: Train
trainCntLim = 3
trainLossLim = [100,150,200][0]
nTrainDays = args.ntraindays              # Num Train days in train graph
nTrainStartDay = args.train_start_day         # The first training day
nTestStartDay = 0           # The first test day, overridden by 'next'
nTotalDays = 91             # The last training day, Upto 90
skipdays = None             # [8,9,10]
num_epochs = 2000
extraValidation = True
latlongprecise = 4
metricprecise = 3
ll_min, ll_max, ll_off = [11.0, 28.48, 77.1], [45.0, 28.72, 77.33], [1, np.round(0.00902*km,latlongprecise), np.round(0.01017*km,latlongprecise)]

arr_sensorloc = np.array([(28.657, 77.227), (28.563, 77.187), (28.624, 77.287), (28.629, 77.241), (28.592, 77.227),
                 (28.672, 77.315), (28.531, 77.271), (28.551, 77.274), (28.568, 77.251), (28.674, 77.131),
                 (28.611, 77.238), (28.58, 77.234), (28.55, 77.216), (28.647, 77.316), (28.588, 77.222),
                (28.681, 77.303), (28.499, 77.265), (28.64, 77.146), (28.636, 77.201), (28.531, 77.19)])

date = ['all,next'][-1]
if canada == 1:
    date = ['2015-02-12', '2015-02-13', '2015-03-19', '2015-03-25', '2015-03-27', '2015-04-01', '2015-04-10', '2015-04-15',
            '2015-04-17', '2015-04-21', '2015-04-24', '2015-05-12',  '2015-05-14', '2015-05-20', '2015-05-22', '2015-09-10', '2015-09-23', '2015-11-23', 'next']
    date = ','.join(date)
elif canada == 2:
    date = ','.join(['canada_20{:02d}'.format(i) for i in range(6,17)]) + ',next'
elif canada == 3:
    date = ','.join(dates_usa)

if args.date is not None:
    date = args.date
metrics = []
metricsH = []
metricsF = args.mfile if args.mfile is not None else ['metrics_GS_{}{}_fold{}_m{}{}_td{}.csv'.format(['delhi','canada','canada','usa'][canada], canada if canada in [1,2] else '', args.fold, mode, '-st' if args.spatiotemporal else '', args.ntraindays)][0]
metricsCtr = 0

###############################################################################

class WeightedSAGEConv(MessagePassing):
    def __init__(self, in_channels: int,
                 out_channels: int, bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'add')
        super(WeightedSAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                normalize: Tensor) -> Tensor:

        x: OptPairTensor = (x, x)
        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, norm = normalize)
        out = self.lin_l(out)

        x_r = x[1]
        out += self.lin_r(x_r)

        return out

    def message(self, x_j: Tensor, norm) -> Tensor:
        return x_j*norm.view(-1,1)

#Net of 2 x 28 x 14 x 5 x 1
#input 2-> mean GraphSage 10-> mean GraphSage 6-> Linear Layer 3-> Output 1
#       ->  max GraphSage  4   max  GraphSage 2
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1Mean = WeightedSAGEConv(2,24)
        self.conv1Max = SAGEConv(2,8)
        self.conv1Max.aggr = 'max'
        self.conv2Mean = WeightedSAGEConv(32,12)
        self.conv2Max = SAGEConv(32,4)
        self.conv2Max.aggr = 'max'
        self.conv3 = nn.Linear(16, 8)
        self.conv4 = nn.Linear(8, 5)
        self.conv5 = nn.Linear(5, 1)

    def forward(self, x, edge_index, norm):
        y = F.relu(self.conv1Mean(x, edge_index, norm))
        z = F.relu(self.conv1Max(x, edge_index))
        x = torch.cat((y,z),1)

        y = F.relu(self.conv2Mean(x, edge_index, norm))
        z = F.relu(self.conv2Max(x, edge_index))
        x = torch.cat((y,z),1)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return torch.squeeze(x)

###############################################################################

def make_edges_all(X, offsets, sensorList=None, sensorMask=None, prev_offset = 0):
    spaceD = X[:, 1:3]
    timeD = X[:, 0]
    edges = np.empty((3,0))
    if sensorList is not None:
        sensorMask = np.zeros((offsets[-1]), dtype=bool)
        sensorMask[sensorList] = True

    for offset in offsets:
        print('Making edges for offset', prev_offset)
        size = offset - prev_offset
        for i in mytqdm(range(size)):
            if sensorList is not None and sensorMask[i + prev_offset]:
                continue
            dist = haversine_vector([spaceD[i+prev_offset]] * size, spaceD[prev_offset:offset])
            tm = (abs(timeD[i+prev_offset] - timeD[prev_offset:offset])/(grouptime-0.1)).astype(int)
            dist[i] = 100  # Ignoring self loop to i
            # Use KNN threshold
            maskTm = (tm <= (edgetime // grouptime))
            mask = maskTm & sensorMask[prev_offset:offset] if sensorList is not None else maskTm
            mask[i] = False  # Ignoring self loop to i
            if mask.sum() == 0:
                global nIsolatedNodes
                nIsolatedNodes += 1
                continue
            ww = 1.0/(1+dist[mask])
            jj = np.arange(len(mask))[mask]
            ii = np.full((len(ww)), i+prev_offset, dtype=int)
            ee = np.vstack((jj + prev_offset, ii, ww))
            edges = np.hstack((edges, ee))

        prev_offset = offset
        print('Edges so far:', edges.shape)

    return edges

def findNorm(tempEdges, y_size):
    # Find Normalization coefficients for edges in weighted graph sage
    weightSum = np.ones(y_size)
    ei = pd.DataFrame(tempEdges, columns = ['from', 'to', 'val'])
    ei = ei.groupby(['to']).agg('sum').reset_index()
    ei = ei[['from','to','val']]
    weightSum[ei.to.to_numpy().astype(int)] = ei.val.to_numpy()
    weightSum = torch.from_numpy(weightSum)
    wsum = weightSum[tempEdges[:,1]]
    if len(wsum) and wsum.min() == 0:
        print("Error in WeightedSageConv, as one of the nodes has no incoming edges to it.")
    normalize = (tempEdges[:,2]/wsum).type(torch.FloatTensor)
    return normalize

def make_graph(data_tuple, edges, meaner = False, ones_vector = None):
    X, y = data_tuple
    size = y.shape[0]
    if (not meaner) and (not ones_vector):
        Nodes = np.ones((size, 2))
        Nodes[:,0] = y.copy()
    else:
        Nodes = y.copy()

    graph = Data(x = torch.from_numpy(Nodes).type(torch.FloatTensor), \
                edge_index = torch.from_numpy(edges[:2,:]).type(torch.LongTensor),\
                norm = findNorm(edges.T, size),
                edge_attr = torch.from_numpy(edges[2, :]),
                train_mask = torch.from_numpy(np.arange(size)))
    return graph

def get_masks(data):
    sensorMask, otherMask = [], []
    j = 0
    for i, x in enumerate(data):
        if i == offsets[j]:
            j += 1
        k = "{}_{}_{}".format(int(x[0]), x[1], x[2]) if args.spatiotemporal else "{}_{}".format(x[1], x[2])
        if k in sensors_str[j]:
            sensorMask.append(i)
        else:
            otherMask.append(i)
    sensorMask = torch.as_tensor(sensorMask)
    otherMask = torch.as_tensor(otherMask)
    return sensorMask, otherMask

def get_masks_test(data, off=0):
    sensorMask, otherMask = [], []
    for i, x in enumerate(data):
        k = "{}_{}_{}".format(int(x[0]), x[1], x[2]) if args.spatiotemporal else "{}_{}".format(x[1], x[2])
        if k in sensors_str[-1]:
            sensorMask.append(i)
        else:
            otherMask.append(i)
    sensorMask = torch.as_tensor(sensorMask) + off
    otherMask = torch.as_tensor(otherMask) + off
    return sensorMask, otherMask

def xform_mode3(X, offsets, sensors_str, other_sensors_str):
    XX,OFF = [],[]
    idx = 0
    for i, x in enumerate(X):
        k = "{}_{}_{}".format(int(x[0]), x[1], x[2]) if args.spatiotemporal else "{}_{}".format(x[1], x[2])
        F = True
        if k in sensors_str[idx]:
            pass
        elif k in other_sensors_str[idx]:
            pass
        else:
            F = False
        if F:
            XX.append(x)
        if i+1 == offsets[idx]:
            OFF.append(len(XX))
            idx += 1
    return np.array(XX), OFF

def make_graphs(data_tuple, offsets, numGraphs:int = nGraphs, device = 'cpu'):
    X, y = data_tuple
    all_graphs = []
    size = len(y)
    graph_count = 0
    while graph_count < numGraphs:
        # find indices of the samples to be selected
        sensorMask, trainMask = get_masks(X)

        #removing edges coming out of validation nodes
        valEdges = make_edges_all(X, offsets, sensorList=sensorMask)
        torch_valWeights = torch.reshape(torch.tensor(valEdges[2,:], dtype = torch.float), (valEdges.shape[1],1))
        torch_valEdges = torch.tensor(valEdges[:2,:], dtype=torch.long)

        # Two features of nodes, 1) PM2.5 value, 2) Presence
        valNodes = np.ones((size, 2))
        valNodes[:, 0] = y.copy()
        valNodes[trainMask, :] = 0.0
        torch_valNodes = torch.tensor(valNodes, dtype = torch.float)

        # normalization calculation
        norm = findNorm(valEdges.T, size)
        sample = Data(x = torch_valNodes, edge_index = torch_valEdges, edge_attr = torch_valWeights, \
                      train_mask=trainMask, val_mask = torch.tensor([j for j in range(len(y)) if j not in trainMask], \
                      dtype=torch.long), norm = norm)
        sample.to(device)
        if sample.has_isolated_nodes():
            # raise 'Isolated Nodes'
            print('Isolated Nodes')
        if 1:
            all_graphs.append(sample)
            if graph_count % 10 == 0:
              print("Graphs Made:", graph_count+1)
            graph_count += 1
    return all_graphs

def prepareTestData(trainLen, trainEdges, finalData, wy):
    sensorMask, testMask = get_masks_test(finalData[trainLen:], trainLen)
    finalEdges = make_edges_all(finalData, [len(finalData)], sensorList=sensorMask, prev_offset=trainLen)

    assert sum([t.item() in finalEdges[0].astype(int) for t in testMask]) == 0
    finalEdges = np.concatenate((trainEdges, finalEdges), axis = 1)
    print('Total Test Graph Edges:', len(finalEdges[0]))

    final_data = finalData[:, wy]
    final_data = final_data.reshape(-1, 1)
    ones_vector = np.ones((final_data.shape[0], 1))
    final_data = np.hstack((final_data, ones_vector))
    final_data[testMask, :] = 0
    return final_data, finalEdges

def train(net, graph_list, opt, y):
    net.train()
    opt.zero_grad()
    loss = None
    for i in graph_list:
        graph = i
        output = net(graph.x.float(), graph.edge_index, graph.norm)
        output = torch.reshape(output,(-1,))
        if trainloss_train_validation == 3: # All
            indLoss = F.mse_loss(output.float(), y.float())
        elif trainloss_train_validation == 1: # Val
            indLoss = F.mse_loss(output[i.train_mask.type(torch.LongTensor)], \
                                y[i.train_mask.type(torch.LongTensor)])
        elif trainloss_train_validation == 2: # Train
            indLoss = F.mse_loss(output[i.val_mask], y[i.val_mask])

        loss = indLoss if loss is None else loss + indLoss
    loss.backward()
    opt.step()
    return loss/len(graph_list), output

def train_GraphSage(graph_list, graph, y, y_ms = (0, 1), num_epochs = 100, cuda = False):
    net = Net().float().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.05)

    epochs_iter = mytqdm(range(num_epochs), desc="Epoch")
    prev_loss = 1000000
    done = args.num_same_epochs
    for epoch in epochs_iter:
        loss, preds = train(net, graph_list, optimizer, y)
        if abs(prev_loss - loss) < args.tol:
            done -= 1
        else: done = args.num_same_epochs
        prev_loss = loss
        if epoch%10 == 9 or done==0:
            global metrics, metricsH
            train_rmse, train_preds, lossOut, lossOutVal = evaluate_GraphSage(net, graph, graph_list, y, y_ms)
            fitLoss = loss.item() ** 0.5
            validloss = lossOutVal.item() / len(graph_list)
            print('({})'.format(epoch+1), "RMSEs: Fitting: {:.2f}, Full: {:.2f}, Train: {:.2f}, Vald: {:.2f}".format(fitLoss, train_rmse.item(), lossOut.item()/len(graph_list), validloss))
            metrics = np.round([fitLoss, train_rmse.item(), lossOut.item() / len(graph_list),validloss, epoch],metricprecise).tolist()
            metricsH = ['Fitting', 'Full', 'Train', 'Validation', 'Epochs']
            if done==0:
                break
    return net, fitLoss, validloss

def evaluate_GraphSage(net, graph, graph_list, y, y_ms = (0, 1)):
    # y = graph.x[:, 0]
    net.eval()
    preds = net(graph.x.float(), graph.edge_index, graph.norm)
    rmse = torch.sqrt(F.mse_loss(preds, y))
    unnormalized_rmse = y_ms[1]*rmse

    lossOut, lossOutVal = None, None
    for i in range(len(graph_list)):
        ith_graph = graph_list[i]
        graph = ith_graph
        out = net(graph.x.float(), graph.edge_index, graph.norm)
        _lossOut = F.mse_loss(out[ith_graph.val_mask], y[ith_graph.val_mask])**0.5
        _lossOutVal = F.mse_loss(out[ith_graph.train_mask.type(torch.LongTensor)], \
                            y[ith_graph.train_mask.type(torch.LongTensor)]) ** 0.5
        lossOut = _lossOut if lossOut is None else lossOut + _lossOut
        lossOutVal = _lossOutVal if lossOutVal is None else lossOutVal + _lossOutVal

    return unnormalized_rmse, preds, lossOut, lossOutVal

def evaluate_graph_list(graph_list, data_tuple, net = None, meaner = False):
    X, y = data_tuple
    totalVal = 0
    totaltrain = 0
    torch_y = torch.from_numpy(y).to(device)

    for i in range(len(graph_list)):
        ith_graph = graph_list[i]
        graph = ith_graph
        out = net(graph.x.float(), graph.edge_index, graph.norm)
        out = torch.reshape(out, (-1,))
        lossOut = F.mse_loss(out, torch_y)**0.5
        lossOutVal = F.mse_loss(out[ith_graph.train_mask.type(torch.LongTensor)], \
                                torch_y[ith_graph.train_mask.type(torch.LongTensor)])**0.5
        totalVal += lossOutVal.item()
        totaltrain += lossOut.item()
        print("Iter:", i)
        print("On Train Set: ", lossOut.item()," || On Validation Set: ", lossOutVal.item())

    print("validation loss : {:.2f}".format(totalVal/len(graph_list)))
    print("training   loss : {:.2f}".format(totaltrain/len(graph_list)))

def to_dt(time_string):
    return pd.to_datetime(time_string).tz_localize('UTC').tz_convert('Asia/Kolkata')

def preprocess_data(date, data_dir = "./data/", hour = None):
    file = date + "_all.csv"
    print('Reading', file)
    df = pd.read_csv(data_dir + file, index_col = 0, parse_dates = ["dateTime"])
    # filter time
    df = df[(df.dateTime >= to_dt(date)) & (df.dateTime <= to_dt(date+ " 18:00:00"))].reset_index(drop = True)

    # day subset
    dfHour = df[['dateTime','lat','long',pm]]

    # hour subset
    if hour:
        dfHour["hour"] = dfHour.dateTime.dt.hour
        dfHour = dfHour[dfHour.hour.isin(hour)]
        dfHour = dfHour.drop("hour", axis = 1)

    # preprocessing time
    dfHour.dateTime = dfHour.dateTime.dt.round('{}min'.format(grouptime))
    dfHour.dateTime = pd.to_datetime(dfHour.dateTime)
    dfHour.dateTime = dfHour.dateTime.dt.hour*60+dfHour.dateTime.dt.minute
    dfHour.dateTime %= 1440
    dfHour = dfHour[(dfHour.dateTime>=300) & (dfHour.dateTime<=1350)]
    dfHour = dfHour.sort_values(by = ['dateTime','lat','long'])
    dfHour = dfHour.reset_index(drop = True)

    # Rounding @15min and spatially
    if km is None:
        dfHour.lat = round(round(5*dfHour.lat.astype(float), 2)/5.0, 3)
        dfHour.long= round(round(5*dfHour.long.astype(float), 2)/5.0, 3)
    else:
        def round_val(val, min, off):
            val1 = ((val - min) / off).astype(int)
            val2 = (val1 * off) + min
            return round(val2, latlongprecise)
        dfHour.lat = round_val(dfHour.lat.astype(float), ll_min[1], ll_off[1])
        dfHour.long = round_val(dfHour.long.astype(float), ll_min[2], ll_off[2])

    # meaning pm values
    dfHour = dfHour.groupby(['dateTime','lat','long']).mean().reset_index()
    dfHour.loc[:, pm] = dfHour.loc[:, pm].round(2)

    return dfHour.values

def find_best_sensor_loc(data, dates):
    loc = dict()
    for i,d in enumerate(data):
        if d[1] not in loc.keys():
            loc[d[1]] = dict()
        if d[2] not in loc[d[1]].keys():
            loc[d[1]][d[2]] = []
        loc[d[1]][d[2]].append(i)
    L = []
    for d1 in loc.keys():
        for d2 in loc[d1].keys():
            L.append([d1, d2, len(loc[d1][d2])])
    L = np.array(L)
    Lall = L.copy()
    pre = 'rand-sensors-'
    mid = 'usa' if canada==3 else 'canada-{}'.format('days' if canada==1 else 'year') if canada else 'delhi'
    post = '-st' if args.spatiotemporal else ''
    S = pkl.load(open('bin/'+pre+mid+post+'.bin','rb'))
    LL,LLstr = [], []
    for k in dates:
        LL.append(S[k][args.fold])
        if args.spatiotemporal:
            LLstr.append(["{}_{}_{}".format(int(s[0]), s[1], s[2]) for s in LL[-1]])
        else:
            LLstr.append(["{}_{}".format(s[0], s[1]) for s in LL[-1]])
    return LL, LLstr, Lall, S

#####################################################################################

def save(data, file):
    pkl.dump(data, open(file,"wb"))

#####################################################################################

def xform_day(day):
    if canada:
        return 'canada_20{:02d}'.format(day)
    arr = [0, 30, 61]
    w = 0 if day <= 30 else 1 if day <= 61 else 2
    mon = ['2020-11-', '2020-12-', '2021-01-'][w]
    date = mon + '{:02d}'.format(day - arr[w])
    return date

#####################################################################################

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
print('Cuda:', cuda, '| Device:', device)
print('Num Sensors:', args.num_sensors)

dates_in = date.split(',')
dates_train, dates_test = [], []

# Preprocess
wy = 3
all_data = dict()
zero_test = 0
nIsolatedNodes = 0

if 'all' in dates_in[:-1]:
    dates_train = []
    for day in range(nTrainStartDay, 1 + nTotalDays):
        if skipdays is None or day not in skipdays:
            dates_train.append(xform_day(day))
else:
    dates_train = dates_in[:-1]
bTestNextDay = False
if 'each' in dates_in[-1:]:
    dates_test = []
    for day in range(nTestStartDay, 1 + nTotalDays):
        if skipdays is None or day not in skipdays:
            dates_test.append(xform_day(day))
elif 'next' in dates_in[-1:]:
    bTestNextDay = True
else:
    dates_test = dates_in[-1:]
if canada == 3:
    bTestNextDay = True
    dates_train = dates_usa
print('GivenDates:', date, '\nTrainDate:', dates_train, '\nTestDate:', dates_test)

if 1:
    for d in dates_train:
        all_data[d] = preprocess_data(d, args.datadir)
else:
    file_loc = 'bin/all_data_{}.bin'.format(['delhi', 'canada', 'canada', 'usa'][canada])
    all_data = pkl.load(open(file_loc, 'rb'))

# Ignore the PM10 from processed data
for dt in dates_train:
    all_data[dt] = all_data[dt][:,:-1]

for dateIdx, d in enumerate(dates_train):

    tmStart = datetime.datetime.now()
    bestvalidloss, bestmodel = 0, -1

    dates = dates_train[dateIdx:dateIdx+nTrainDays+bUseTestDayInTrain]
    if len(dates) < nTrainDays + bUseTestDayInTrain:
        break
    if len(dates_train) <= dateIdx + nTrainDays:
        break

    print('\n--------------------------------------------')
    print('\tTrain Day:', *dates)
    print('--------------------------------------------')
    if bTestNextDay:
        dates_test = [dates_train[dateIdx+nTrainDays]]

    tmTrainPreStart = datetime.datetime.now()

    trainData = np.zeros((0, wy + 1))
    arrtestData, offsets = [], []
    for dt in dates_test:
        arrtestData.append((dt,all_data[dt]))
    for i,d in enumerate(dates):
        trainData = np.vstack((trainData, all_data[d]))
        offsets.append(len(trainData))

    sensors, sensors_str, allsensors, all_folds = find_best_sensor_loc(trainData, dates+dates_test)
    other_sensors_str = []
    if mode == 3:
        # Take some sensors to other_sensors
        for idx1 in range(len(offsets)):
            idxS, idxOS = [], []
            ns = len(sensors_str[idx1])
            idxOS = np.random.choice(np.arange(ns), size=int(ns*args.r2b), replace=False)
            idxS = [i for i in range(ns) if i not in idxOS]
            other_sensors_str.append(np.array(sensors_str[idx1])[idxOS].tolist())
            sensors_str[idx1] = np.array(sensors_str[idx1])[idxS].tolist()
        trainData, offsets = xform_mode3(trainData, offsets, sensors_str, other_sensors_str)

    x_train, y_train = trainData[:, :3], trainData[:, wy].flatten()
    data_tuple = (x_train, y_train)

    #####################################################################################

    graph_list = make_graphs(data_tuple, offsets, device = device)
    graph_eval = graph_list[0]

    tmTrainPre = (datetime.datetime.now() - tmTrainPreStart).total_seconds()

    for nruns in range(args.runs):

        ### training the model
        tmTrainStart = datetime.datetime.now()
        metrics_off = len(metrics)
        for trainCtr in range(trainCntLim):
            net, loss, validloss = train_GraphSage(graph_list, graph_eval, torch.tensor(data_tuple[1]).to(device), num_epochs = num_epochs, cuda=cuda)
            if bestmodel<0 or validloss < bestvalidloss:
                bestvalidloss = validloss
                bestmodel = nruns
            if loss < trainLossLim or trainCtr >= trainCntLim-1:
                break
            metrics = metrics[:metrics_off]
            metricsH = metricsH[:metrics_off]

        tmTrain = (datetime.datetime.now() - tmTrainStart).total_seconds()
        print("Train Time:", int(tmTrain*1000))

        metricsH.extend(['tmTrainPre','tmTrain'])
        metrics.extend([int(tmTrainPre*1000), int(tmTrain*1000)])

        ################## Test Data Loop ####################
        metric_offset,metrics_adjust = len(metrics), 0
        for testday,testData in arrtestData:

            if not bTestNextDay and testday in dates:
                metricsH.append(testday)
                metrics.append(0)
                metrics_adjust += 1
                continue

            print('\n----------- Test Day: {} -----------'.format(testday))

            tmTestStart = datetime.datetime.now()

            trainLen = 0
            finalData = testData
            trainEdges = np.zeros((3,0))

            print('Preparing final test data')
            final_data, final_edges = prepareTestData(trainLen, trainEdges, finalData, wy)
            print('Making final graph')
            graph_final = make_graph((None, final_data), final_edges, meaner = False, ones_vector = True)

            graph_final.to(device)
            net.eval()
            testIdx = [i for i in range(trainLen, final_data.shape[0]) if final_data[i,1] == 0]
            testIdx2 = np.array(testIdx) - trainLen

            tmTestPre = (datetime.datetime.now() - tmTestStart).total_seconds()

            testsensorIdx = [i for i in range(trainLen, final_data.shape[0]) if final_data[i,1] == 1]
            if len(testsensorIdx) <= 0:
                print('************ ZERO TEST SENSORS **************')
                zero_test += 1
                continue
            testsensorIdx2 = np.array(testsensorIdx) - trainLen
            testsensordata = testData[testsensorIdx2,3]
            testmean_std = testsensordata.mean(), pstdev(testsensordata)

            print('Evaluating final graph')
            evaluate_graph_list(graph_list, data_tuple, net, False)

            # evaluating the model on the test data
            tmInferStart = datetime.datetime.now()
            out = net(graph_final.x.float(), graph_final.edge_index, graph_final.norm)
            tmInfer = (datetime.datetime.now() - tmInferStart).total_seconds()

            outtest = out[testIdx].detach()
            tdt = torch.from_numpy(testData[testIdx2, 3]).to(device)
            testloss = torch.sqrt(F.mse_loss(outtest, tdt)).item()
            print("Model Test loss : {:.2f}".format(testloss))
            metrics.extend([np.round(testloss,metricprecise), int(tmTestPre*1000), int(tmInfer*1000)])
            metricsH.extend(['Testloss','tmTestPre','tmInfer'])

            if len(arrtestData) > 1:
                continue

            meanF = testData[:,3].mean()
            meanP = testData[testIdx2, 3].mean()
            print('Mean Predictor PM: Sensor {:.2f}, Full {:.2f}, Pred {:.2f}'.format(testmean_std[0],meanF,meanP))
            metrics.extend(np.round([testmean_std[0],meanF,meanP],metricprecise).tolist())
            metricsH.extend(['MeanPM: Sensor', 'Full', 'Pred'])

            # Predicted Loss
            out = tdt.clone()
            out[:] = meanP if 0 else testmean_std[0]
            Ploss = torch.sqrt(F.mse_loss(out, tdt)).item()

            # Sensor Loss
            tdtS = torch.from_numpy(testsensordata)
            out = tdtS.clone()
            out[:] = testmean_std[0]
            Sloss = torch.sqrt(F.mse_loss(out, tdtS)).item()

            # Full Loss
            tdt2 = torch.from_numpy(testData[:,3])
            out2 = tdt2.clone()
            out2[:] = meanF if 0 else testmean_std[0]
            Floss = torch.sqrt(F.mse_loss(out2, tdt2)).item()
            print("Mean Predictor Test Loss: Full {:.2f}, Pred {:.2f}, Sensor {:.2f}".format(Floss, Ploss, Sloss ) )
            metrics.extend(np.round([Floss, Ploss, Sloss],metricprecise).tolist())
            metricsH.extend(['MeanPredTestLoss_Full', 'Pred', 'Sensor'])

            print('Pred  Mean : {:.2f}, Std Dev: {:.2f}'.format(outtest.mean(), pstdev(outtest.cpu().tolist())) )
            print('TestP Mean : {:.2f}, Std Dev: {:.2f}'.format(testData[testIdx2,3].mean(), pstdev(testData[testIdx2,3])) )
            print('TestF Mean : {:.2f}, Std Dev: {:.2f}'.format(testData[:,3].mean(), pstdev(testData[:,3])) )
            metrics.extend(np.round([outtest.mean().item(), pstdev(outtest.cpu().tolist()), testData[testIdx2,3].mean(), pstdev(testData[testIdx2,3]), testData[:,3].mean(), pstdev(testData[:,3])],metricprecise).tolist())
            metricsH.extend(['PredMean', 'PredSD', 'TestPmean', 'TestPsd', 'TestFmean', 'TestFsd'])
            prev_o = 0
            for offset in offsets:
                pm = trainData[prev_o:offset, 3]
                print('Train Mean : {:.2f}, Std Dev: {:.2f}'.format(pm.mean(), pstdev(pm)))
                prev_o = offset

        print('\n------------------------------')
        if len(arrtestData) > 1:
            testlosses = metrics[metric_offset:]
            avgtestloss = sum(testlosses)/(len(testlosses)-metrics_adjust)
            maxtestloss = max(testlosses)
            print('TestLoss: Avg {:.2f}, Max {:.2f}'.format(avgtestloss, maxtestloss))
            metricsH.extend(['AvgTestLoss', 'MaxTestLoss'])
            metrics.extend(np.round([avgtestloss, maxtestloss],metricprecise).tolist())

        print('Num Edges: Train {} | Eval {} | Test {}'.format(len(graph_list[0].norm), len(graph_eval.norm), len(graph_final.norm)))

        metricsH.extend(['TrainEdges', 'EvalEdges', 'TestEdges', 'tmTotal', 'IsolatedNodes', 'BestModel'])
        metrics.extend([len(graph_list[0].norm), len(graph_eval.norm), len(graph_final.norm), int((datetime.datetime.now() - tmStart).total_seconds()*1000), nIsolatedNodes, bestmodel])

        if metricsF is not None: # and len(sys.argv)>1:
            if 0 and not os.path.exists(metricsF):
                f = open(metricsF, 'w')
                print('Dates', *metricsH, sep=',', file=f)
                f.close()
            f = open(metricsF, 'a')
            if metricsCtr == 0:
                print('Dates', *metricsH, sep=',', file=f)
            metricsCtr += 1
            print('|'.join(dates+[testday]), end=',', file=f)
            print(*metrics, sep=',', file=f)
            f.close()

        print('Total Time {:.2f}'.format( (datetime.datetime.now() - tmStart).total_seconds()) )

# EndLoop dates_train

if nIsolatedNodes:
    print('Graph has {} Isolated Nodes'.format(nIsolatedNodes))
if zero_test:
    print('********* {} ZERO TEST SENSORS **************'.format(zero_test))