import pandas as pd
from darts import TimeSeries
from darts.models import NBEATSModel
from fancyimpute import KNN
import numpy as np
import pandas as pd
import random
import datetime
import sys

usa = 0
mon_str = ['2020-11','2020-12','2021-01']
mons = ['Nov', 'Dec', 'Jan']
days = [0, 30, 61]

test_start = 15
grouptime = 30
km = 1
history = 3
test_portion = 1 if usa else 36

lat_range = {'min': 28.486547, 'max': 28.719998, 'diff': 0.00451}
long_range = {'min': 77.100136, 'max': 77.320763, 'diff': 0.005085}

def data_preparation(datafile):
    #Put the file location
    df = pd.read_csv(datafile)
    #type casting
    df.pm1_0 = df.pm1_0.astype(float)
    df.pm2_5 = df.pm2_5.astype(float)
    df.pm10 = df.pm10.astype(float)

    # Ensuring Delhi region and removing outliers from data
    df = df[(df.lat.astype(int) == 28) &(df.long.astype(int) == 77)]
    df = df[(df.pm1_0<=1500) & (df.pm2_5<=1500) & (df.pm10<=1500) & (df.pm1_0>=20) & (df.pm2_5>=30) & (df.pm10>=30)]

    # rounding @30min
    df.dateTime = pd.to_datetime(df.dateTime)
    df.dateTime = df.dateTime.dt.round('{}min'.format(grouptime))
    # use time as a feature as well
    df.dateTime = df.dateTime.dt.hour*60 + df.dateTime.dt.minute
    df.dateTime %= 1440

    def round_val(val, min, off):
        val1 = ((val - min) / off).astype(int)
        val2 = (val1 * off) + min
        return val2
    df.lat = round_val(df.lat.astype(float), lat_range['min'], lat_range['diff'])
    df.long = round_val(df.long.astype(float), long_range['min'], long_range['diff'])
    df['lat_grid'] = df.apply(
        lambda row: int(((row.lat - lat_range['min']) / lat_range['diff']) + 0.1),
        axis=1)
    df['long_grid'] = df.apply(
        lambda row: int(((row.long - long_range['min']) / long_range['diff']) + 0.1),
        axis=1)

    df = df.pivot_table(index=['lat_grid','long_grid'], columns='dateTime', aggfunc='mean')['pm2_5']
    return df

def data_preparation_usa(DF, datafile):
    #Put the file location
    df = DF[pd.DatetimeIndex(DF.dateTime)==datafile]
    #type casting
    df.pm2_5 = df.pm2_5.astype(float)
    df.pm10 = df.pm10.astype(float)

    # rounding @30min
    df.dateTime = pd.to_datetime(df.dateTime)
    df.dateTime = df.dateTime.dt.round('{}min'.format(grouptime))
    # use time as a feature as well
    df.dateTime = df.dateTime.dt.hour*60 + df.dateTime.dt.minute
    df.dateTime %= 1440

    def round_val(val, min, off):
        val1 = ((val - min) / off).astype(int)
        val2 = (val1 * off) + min
        return val2
    df.lat = round_val(df.lat.astype(float), lat_range['min'], lat_range['diff'])
    df.long = round_val(df.long.astype(float), long_range['min'], long_range['diff'])
    df['lat_grid'] = df.apply(
        lambda row: int(((row.lat - lat_range['min']) / lat_range['diff']) + 0.1),
        axis=1)
    df['long_grid'] = df.apply(
        lambda row: int(((row.long - long_range['min']) / long_range['diff']) + 0.1),
        axis=1)

    df = df.pivot_table(index=['lat_grid','long_grid'], columns='dateTime', aggfunc='mean')['pm2_5']
    return df

# Results of baseline -NBeats
from sklearn.metrics import mean_squared_error
from math import sqrt
test_rmse = []
test_days = []
df_shape = []
tm_data, tm_train, tm_infer, tm_eval = [],[],[],[]

def proc_delhi():
    for test_idx in range(test_start,30+31+30+1):
        start = test_idx - history
        df = None
        tmStart = datetime.datetime.now()
        for i in range(start, start + history + 1):
            mon_idx = 2 if i > days[2] else 1 if i > days[1] else 0
            datafile = 'raw_data/{}-{:02d}_all.csv'.format(mon_str[mon_idx], i - days[mon_idx])
            print('data', datafile)
            df1 = data_preparation(datafile)
            df = pd.concat([df, df1], axis=1) if df is not None else df1

        df = df.fillna(0)
        test_days.append('{} {}'.format(i - days[mon_idx], mons[mon_idx]))
        # df = df.set_index(['lat_grid','long_grid'])
        proc(df, tmStart, start)

def proc_usa():
    from dates import dates_usa

    DF = pd.read_csv('city_pollution_data.csv')
    DF = DF[['Date', 'Time', 'latitude', 'longitude', 'pm25_median', 'pm10_median']]
    DF.columns = ['dateTime', 'Time', 'lat', 'long', 'pm2_5', 'pm10']
    global lat_range, long_range
    lat_range = {'min': 21.3, 'max': 47.6, 'diff': 0.005}
    long_range = {'min': -157.9, 'max': -71.0, 'diff': 0.005}

    off = 0 if len(sys.argv) < 2 else int(sys.argv[1])
    for test_idx in range(history+off, len(dates_usa)):
        start = test_idx - history
        df = None
        tmStart = datetime.datetime.now()
        for i in range(start, start + history + 1):
            datafile = dates_usa[i]
            print('data', datafile)
            df1 = data_preparation_usa(DF, datafile)
            df = pd.concat([df, df1], axis=1)

        df = df.fillna(0)
        test_days.append(dates_usa[test_idx])
        # df = df.set_index(['lat_grid','long_grid'])

        proc(df, tmStart, start, ln=1)

def proc(df, tmStart, idx, ln=6):
    data = np.array(df)
    data[:,:-test_portion] = np.where(data[:,:-test_portion] == 0, np.nan, data[:,:-test_portion])

    tmData = datetime.datetime.now()

    random.seed(10)
    knn_imputer = KNN()
    data[:,:-test_portion] = knn_imputer.fit_transform(pd.DataFrame(data[:,:-test_portion]))
    model = NBEATSModel(input_chunk_length=ln, output_chunk_length=ln, random_state=df.shape[1]+1) #, pl_trainer_kwargs={"accelerator": "gpu","gpus": [1]})
    model.fit([TimeSeries.from_values(train_series[:-test_portion]) for train_series in data], epochs=2, verbose=True)

    tmTrain = datetime.datetime.now()

    pred_air = model.predict(series=[TimeSeries.from_values(time_series[:-test_portion]) for time_series in data], n=test_portion)
    predicted_series = model.predict(series=[TimeSeries.from_values(time_series[:-test_portion]) for time_series in data], n=test_portion)

    tmInfer = datetime.datetime.now()

    actual_test = data[:,-test_portion:]
    predicted_test = np.array([series.values().reshape(-1,) for series in predicted_series])
    rmse = sqrt(mean_squared_error(actual_test[actual_test>0], predicted_test[actual_test>0]))

    tmEval = datetime.datetime.now()

    test_rmse.append(rmse)
    df_shape.append(df.shape)
    print(test_days[-1], rmse)

    def tmsecs(tm):
       return int(tm.total_seconds() * 1000)
    tm_data.append(tmsecs(tmData - tmStart))
    tm_train.append(tmsecs(tmTrain - tmData))
    tm_infer.append(tmsecs(tmInfer - tmTrain))
    tm_eval.append(tmsecs(tmEval - tmInfer))

    if 1:
        out = pd.DataFrame()
        out['rmse'] = test_rmse[-1:]
        out['day'] = test_days[-1:]
        out['DF'] = df_shape[-1:]
        out['tmData'] = tm_data[-1:]
        out['tmTrain'] = tm_train[-1:]
        out['tmInfer'] = tm_infer[-1:]
        out['tmEval'] = tm_eval[-1:]
        out['idx'] = idx
        out.to_csv('NBeats_results_1.csv', mode='a', header=None)

if usa:
    proc_usa()
else:
    proc_delhi()

out = pd.DataFrame()
out['TestRMSE'] = test_rmse
out['day'] = test_days
out['DF'] = df_shape
out['tmData'] = tm_data
out['tmTrain'] = tm_train
out['tmInfer'] = tm_infer
out['tmEval'] = tm_eval
out.to_csv('NBeats_results.csv')

print('RMSE', out['rmse'].mean())