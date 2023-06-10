import tqdm
import torch
import argparse
import math
import pickle as pkl
import pandas as pd
import numpy as np
from haversine import haversine, haversine_vector, Unit
from sklearn.model_selection import KFold

mytqdm = tqdm.notebook.tqdm if 0 else tqdm.tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_sensors",type=int,default=60)
    parser.add_argument("--train_start_day",type=int,default=12)
    parser.add_argument("--date",type=str,default=None)
    parser.add_argument("--mfile",type=str,default=None)
    parser.add_argument("--datadir",type=str,default=None)
    parser.add_argument("--canada",type=int,default=0)
    parser.add_argument("--usa",type=int,default=0)
    parser.add_argument("--spatiotemporal",type=int,default=1)
    args = parser.parse_args()
    return args
args = parse_args()

if args.usa:
    args.canada = 3

canada = args.canada
rand_sensor_locs = 1

fold = [3,5][rand_sensor_locs]

heatmap = 0
datadir = args.datadir
pm = ['pm10','pm2_5'][1]
pm2 = ['pm10','pm2_5'][0]
sensorTyp = ['random'][0]

km = 1
grouptime = 30

if canada == 1:
    nTotalDays = nTrainStartDay = 15
    args.num_sensors = 10
elif canada == 2:
    nTrainStartDay = 6
    nTotalDays = 16
elif canada == 3: # USA
    nTrainStartDay = 1
    nTotalDays = 10
else:
    nTrainStartDay = args.train_start_day
    nTotalDays = 91

nTestStartDay = nTrainStartDay           # The first test day, overridden by 'next'
skipdays = None
latlongprecise = 4
metricprecise = 3
if canada:
    ll_min, ll_max = [0, 43.135, -80.025], [0, 43.35, -79.638]
else:
    ll_min, ll_max = [11.0, 28.48, 77.1], [45.0, 28.72, 77.33]
ll_off = [1, np.round(0.00902*km,latlongprecise), np.round(0.01017*km,latlongprecise)]
g_lim = 0

date = 'all,next'
if args.date is not None:
    date = args.date

###############################################################################

def to_dt(time_string):
    return pd.to_datetime(time_string).tz_localize('UTC').tz_convert('Asia/Kolkata')

def preprocess_data(date, data_dir = "./data/", hour = None):
    file = date + "{}.csv".format('_all' if canada == 0 else '')
    print('Reading', file)
    df = pd.read_csv(data_dir + file, index_col = 0, parse_dates = ["dateTime"])
    # filter time
    if not canada:
        df = df[(df.dateTime >= to_dt(date)) & (df.dateTime <= to_dt(date+ " 18:00:00"))].reset_index(drop = True)

    # day subset
    dfHour = df[['dateTime','lat','long',pm,pm2]]

    # hour subset
    if hour:
        dfHour["hour"] = dfHour.dateTime.dt.hour
        dfHour = dfHour[dfHour.hour.isin(hour)]
        dfHour = dfHour.drop("hour", axis = 1)

    # preprocessing time
    dfHour.dateTime = dfHour.dateTime.dt.round('{}min'.format(grouptime))
    dfHour.dateTime = pd.to_datetime(dfHour.dateTime)
    if canada != 1:
        dfHour.dateTime = dfHour.dateTime.dt.hour*60+dfHour.dateTime.dt.minute
        dfHour.dateTime %= 1440
        dfHour = dfHour[(dfHour.dateTime>=300) & (dfHour.dateTime<=1350)]
    dfHour = dfHour.sort_values(by = ['dateTime','lat','long'])
    dfHour = dfHour.reset_index(drop = True)

    def round_val(val, min, off):
        val1 = ((val - min) / off).astype(int)
        val2 = (val1 * off) + min
        return round(val2, latlongprecise)
    dfHour.lat = round_val(dfHour.lat.astype(float), ll_min[1], ll_off[1])
    dfHour.long = round_val(dfHour.long.astype(float), ll_min[2], ll_off[2])

    # meaning pm values
    df = dfHour.groupby(['dateTime','lat','long']).mean().reset_index()
    df.loc[:, pm] = df.loc[:, pm].round(2)
    df.loc[:, pm2] = df.loc[:, pm2].round(2)

    return df.values

def find_best_sensor_loc(data, lim=0):
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
    if sensorTyp.startswith('rand'):
        L = L[L[:, 2] > lim]
        if len(L) > args.num_sensors:
            idx = np.random.choice(len(L), args.num_sensors)
            LL = L[idx, :2]
        else:
            LL = L[:, :2]
    else:
        LL = L[L[:, 2].argsort()][::-1]
        LL = LL[:args.num_sensors,:2]
    return LL, Lall

def num_sensor_loc(sensors, data):
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
    lim, cnt = 0, 0
    L = L[L[:, 2] > lim]
    for s in sensors:
        for LL in L:
            if s[0] == LL[0] and s[1] == LL[1]:
                cnt += 1
                break
    return cnt

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
print('Policy:', sensorTyp)

dates_in = date.split(',')
dates_train = []

# Preprocess
wy = 3
all_data = dict()

for day in range(min(nTrainStartDay,nTestStartDay), 1+nTotalDays):
    if skipdays is None or day not in skipdays:
        d = xform_day(day)
        dates_train.append(d)
        all_data[d] = preprocess_data(d, "raw_data/")

if canada == 1:
    print('For 2015 data')
    dates_train = ['2015-02-12', '2015-02-13', '2015-03-19', '2015-03-25', '2015-03-27', '2015-04-01', '2015-04-10', '2015-04-15',
     '2015-04-17', '2015-04-21', '2015-04-24', '2015-05-12',  '2015-05-14', '2015-05-20', '2015-05-22', '2015-09-10', '2015-09-23', '2015-11-23']

if args.usa:
    df = pd.read_csv('city_pollution_data.csv')
    df = df[['Date', 'Time', 'latitude', 'longitude', 'pm25_median', 'pm10_median']]
    df.columns = ['Date', 'Time', 'lat', 'long', 'pm2_5', 'pm10']
    ll_min, ll_max = [0, 21.3, -157.9], [0, 47.6, -71.0]
    not_nan = [not math.isnan(i) for i in df.pm2_5.values]
    df = df[not_nan]

    def round_val(val, min, off):
        val1 = ((val - min) / off).astype(int)
        val2 = (val1 * off) + min
        return round(val2, latlongprecise)
    df.lat = round_val(df.lat.astype(float), ll_min[1], ll_off[1])
    df.long = round_val(df.long.astype(float), ll_min[2], ll_off[2])

    all_data = dict()
    dates_train = df.Date.unique().tolist()
    for dt in dates_train:
        D = df[df.Date == dt]
        all_data[dt] = D[['Time', 'lat', 'long', 'pm2_5', 'pm10']].values
if 1:
    # Gather All Data, and have independent sensors
    testData = np.zeros((0, 5))
    gloc = dict()
    for dateIdx, dt in enumerate(dates_train):
        testData = np.concatenate((testData, all_data[dt]), axis=0)
        loc = dict()
        for i, d in enumerate(all_data[dt]):
            if d[1] not in loc.keys():
                loc[d[1]] = dict()
            if d[2] not in loc[d[1]].keys():
                loc[d[1]][d[2]] = 1
                if d[1] not in gloc.keys():
                    gloc[d[1]] = dict()
                if d[2] not in gloc[d[1]].keys():
                    gloc[d[1]][d[2]] = 0
                gloc[d[1]][d[2]] += 1
            else:
                loc[d[1]][d[2]] += 1

    print('Data Shape', testData.shape)
    sensors, allsensors = find_best_sensor_loc(testData, lim=g_lim)

def get_spatial_locs(X):
    dstr = ["{}_{}".format(x[1], x[2]) for x in X]
    L = list(set(dstr))
    return np.array(L)

S = dict()
seen, unseen = 0, 1
for dateIdx, dt in enumerate(dates_train):
    dt2 = dt
    if args.usa:
        dt2 = dt.split('/')
        dt2 = '{}-{:02d}-{:02d}'.format(dt2[2], int(dt2[0]), int(dt2[1]))
        print('\'{}\','.format(dt2))
    S[dt2] = []
    data = []

    kf = KFold(n_splits=fold, shuffle=True, random_state=0)
    X = all_data[dt]
    if args.spatiotemporal:
        for train, test in kf.split(X):
            data.append([X[train], X[test]])
            S[dt2].append(X[train][:, :3])
    else:
        L = get_spatial_locs(X)
        for trainL, testL in kf.split(L):
            trainLL = L[trainL]
            testLL = L[testL]
            train, test = [],[]
            for i, x in enumerate(X):
                k = "{}_{}".format(x[1], x[2])
                if k in trainLL:
                    train.append(i)
                elif k in testLL:
                    test.append(i)
                else:
                    raise 'Error'
            data.append([X[train],X[test]])
            S[dt2].append(X[train][:,1:3])

    for f in range(fold):
        if rand_sensor_locs == 0:
            data.append([[], []])
            for d in all_data[dt]:
                dd = '{}_{}'.format(d[1], d[2])
                w = seen if dd in SS[f] else unseen
                data[f][w].append(d)
        header = ['dateTime', 'lat', 'long', 'pm2_5', 'pm10']
        df = pd.DataFrame(data[f][seen])
        df.columns = header
        df.to_csv('{}/{}_f{}_train.csv'.format(datadir,dt2,f), header=header)
        df = pd.DataFrame(data[f][unseen])
        df.columns = header
        df.to_csv('{}/{}_f{}_test.csv'.format(datadir,dt2,f), header=header)

pre = 'rand-sensors-'
mid = 'usa' if args.usa else 'delhi' if not canada else 'canada-days' if canada == 1 else 'canada-year' if canada == 2 else ''
post = '-st' if args.spatiotemporal else ''
save(S, pre + mid + post + '.bin')
print('Done')
