import datetime
import argparse
import pandas as pd
import numpy as np
from polire import IDW
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from dates import dates_usa

T = [0,1][1]
mode_t = ['AB','AC','C','ABC'][1]
mode_p = ['D','CD'][0]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mfile",type=str,default=None)
    parser.add_argument("--fold",type=int,default=-1)
    parser.add_argument("--canada",type=int,default=-1)
    parser.add_argument("--algo",type=str,default=None)
    parser.add_argument("--train_days",type=int,default=3)
    parser.add_argument("--seed",type=int,default=10)
    parser.add_argument("--temporal_scaling",type=int,default=T)
    parser.add_argument("--spatiotemporal",type=int,default=1)
    parser.add_argument("--datadir",type=str,default=None)
    args = parser.parse_args()
    return args

args = parse_args()
maxFolds = 5
if args.spatiotemporal < 0 and mode_t == 'AB' and mode_p == 'CD':
    # Forecasting, single fold is enough
    maxFolds = 1
if args.fold < 0:
    args.fold = maxFolds

algos = ['xgb','idw','rf','err']
algo = algos[-1]
if args.algo is not None:
    if args.algo not in algos:
        raise 'Invalid algo'
    algo = args.algo
canada = args.canada
data_dir = args.datadir
np.random.seed(args.seed)

def get_suffixes(mode):
    suffixes = []
    if 'C' in mode or 'A' in mode:
        suffixes.append('train')
    if 'D' in mode or 'B' in mode:
        suffixes.append('test')
    return suffixes
train_suffix = get_suffixes(mode_t)

model = IDW(exponent=3) if algo=='idw' else RandomForestRegressor(n_estimators = 100, random_state = 0) if algo=='rf' else GradientBoostingRegressor() if algo=='xgb' else None
assert model is not None

def rename_cols(data):
    data.rename(
        columns={'dateTime': 'time', 'lat': 'latitude', 'long': 'longitude', 'pm2_5': 'PM25_Concentration',
                 'pm10': 'PM10_Concentration'}, inplace=True)

def return_data_0(train_input, test_input, with_scaling):
    train_output = np.array(train_input['pm2_5'])
    train_input = train_input[['dateTime','lat','long']]
    test_output = np.array(test_input['pm2_5'])
    test_input = test_input[['dateTime','lat','long']]
    if with_scaling:
        scaler = MinMaxScaler().fit(train_input)
        if args.temporal_scaling:
            data = scaler.transform(pd.concat((train_input, test_input)))
            test_input = data[len(train_input):]
            train_input = data[:len(train_input)]
        else:
            train_input = scaler.transform(train_input)
            test_input = scaler.transform(test_input)
    return train_input,train_output,test_input,test_output

def return_data_time(fold,data,with_scaling):
    train_input = None
    if 'A' in mode_t or 'B' in mode_t:
        for idx,dt in enumerate(data[:-1]):
            for suffix in train_suffix:
                input = pd.read_csv(data_dir+'/'+dt+'_f'+str(fold)+'_'+suffix+'.csv')
                if args.temporal_scaling:
                    input.dateTime += idx * 24 * 60
                train_input = pd.concat((train_input, input))
    if 'C' in mode_t:
        input = pd.read_csv(data_dir + '/' + data[-1] + '_f' + str(fold) + '_train.csv')
        if args.temporal_scaling:
            input.dateTime += (len(data)-1) * 24 * 60
        train_input = pd.concat((train_input, input))

    test_input = pd.read_csv(data_dir+'/'+data[-1]+'_f'+str(fold)+'_test.csv')
    if 'C' in mode_p:
        input = pd.read_csv(data_dir + '/' + data[-1] + '_f' + str(fold) + '_train.csv')
        test_input = pd.concat((input, test_input))
    if args.temporal_scaling:
        test_input.dateTime += (len(data)-1) * 24 * 60

    return return_data_0(train_input, test_input, with_scaling)

def train_model(model,train_input,train_output):
    model.fit(train_input, train_output)

def train_loss(model,train_input,train_output):
    train_input_1 = train_input
    train_pred = model.predict(train_input_1)
    err = mean_squared_error(train_pred, train_output, squared=False)
    return err

def infer_model(model,test_input):
    test_pred = model.predict(test_input)
    return test_pred

def eval_model(test_output, test_pred):
    err = mean_squared_error(test_output, test_pred, squared=False)
    return err

us_data = None
def return_data_forecast(fold, data):
    assert mode_t == 'AB' and mode_p == 'CD'

    data_input = None
    for idx,dt in enumerate(data):
        df = None
        for suffix in train_suffix:
            input = pd.read_csv(data_dir+'/'+dt+'_f'+str(fold)+'_'+suffix+'.csv')
            df = pd.concat((df, input))
        df = df.pivot_table(index=['lat','long'], columns='dateTime', aggfunc='mean')['pm2_5']
        data_input = pd.concat((data_input, df), axis=1)
    df = data_input.fillna(0)
    return df

def process(fold, date):
    tmStart = datetime.datetime.now()
    return_data = return_data_time
    train_input,train_output,test_input,test_output = return_data(fold=fold,data=date,with_scaling=True)
    tmPre = datetime.datetime.now()
    train_model(model,train_input,train_output)
    tmTrain = datetime.datetime.now()
    test_pred = infer_model(model,test_input)
    tmInfer = datetime.datetime.now()
    rmseTrain = train_loss(model,train_input,train_output)
    tmEvalTrain = datetime.datetime.now()

    tmEvalTrain = (tmEvalTrain - tmInfer).total_seconds()
    tmInfer = (tmInfer - tmTrain).total_seconds()
    tmTrain = (tmTrain - tmPre).total_seconds()
    tmPre = (tmPre - tmStart).total_seconds()

    tmEval = datetime.datetime.now()
    rmse = eval_model(test_output, test_pred)
    tmEval = (datetime.datetime.now() - tmEval).total_seconds()

    print(*date, "Fold: {}, RMSE: {}".format(fold, rmse))

    return '{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}'.format('|'.join(date),fold,rmseTrain,rmse,tmPre,tmTrain,tmEvalTrain,tmInfer,tmEval)


def xform_day(day):
    arr = [0, 30, 61]
    w = 0 if day <= 30 else 1 if day <= 61 else 2
    mon = ['2020-11-', '2020-12-', '2021-01-'][w]
    date = mon + '{:02d}'.format(day - arr[w])
    return date

folds = [i for i in range(maxFolds)] if args.fold >= maxFolds else [args.fold]
header = 'Date,Fold,TrainRMSE,TestRMSE,tmPre,tmTrain,tmTrnEval,tmInfer,tmEval'

def proc_file(canada=0):
    mfile = '{}_{}{}_{}{}_{}{}.metrics.csv'.format(algo, ['delhi','canada','canada','usa'][canada], canada if canada in [1,2] else '', mode_t, 'T' if args.temporal_scaling else '', mode_p, '_st' if args.spatiotemporal>0 else '_20' if args.spatiotemporal<0 else '') if args.mfile is None else args.mfile
    f = open(mfile, 'a')
    print(',,'.join([header for _ in range(len(folds))]), file=f)
    return f

def proc_canada(canada):
    f = proc_file(canada)
    if canada == 1:
        dates = ['2015-02-12', '2015-02-13', '2015-03-19', '2015-03-25', '2015-03-27', '2015-04-01', '2015-04-10', '2015-04-15',
         '2015-04-17', '2015-04-21', '2015-04-24', '2015-05-12',  '2015-05-14', '2015-05-20', '2015-05-22', '2015-09-10', '2015-09-23', '2015-11-23']
    else:
        dates = ['canada_20{:02d}'.format(i) for i in range(6,17)]
    for day in range(args.train_days, len(dates)):
        date = []
        for i in range(args.train_days,-1,-1):
            date.append(dates[day-i])
        metrics = []
        for fold in folds:
            metrics.append(process(fold, date))
        print(',,'.join(metrics), file=f)

def proc_delhi():
    f = proc_file()
    nTestStartDay = 15
    nTotalDays = 91
    for day in range(nTestStartDay, nTotalDays+1):
        date = []
        for i in range(args.train_days,-1,-1):
            date.append(xform_day(day-i))
        metrics = []
        for fold in folds:
            metrics.append(process(fold, date))
        print(',,'.join(metrics), file=f)

def proc_usa():
    f = proc_file(3)
    dates = dates_usa
    for day in range(args.train_days, len(dates)):
        date = []
        for i in range(args.train_days, -1, -1):
            date.append(dates[day - i])
        metrics = []
        for fold in folds:
            metrics.append(process(fold, date))
        print(',,'.join(metrics), file=f)

if canada < 0:
    proc_delhi()
    proc_canada(1)
    proc_canada(2)
    proc_usa()
elif canada == 0:
    proc_delhi()
elif canada == 1:
    proc_canada(canada)
else:
    proc_usa()
