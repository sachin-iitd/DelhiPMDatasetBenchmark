import pandas as pd

from lib import *

class Data:
    def __init__(self, fold, Xcols, get_Xm=False, seed=None, file=None):
        self.Xscaler = StandardScaler()
        self.yscaler = StandardScaler()
        # maxFolds = 3
        self.fold = [fold] # if int(fold) >= 0 else [str(i) for i in range(maxFolds)]
        self.Xcols = Xcols.split('@')
        #######
        self.factor = 4 # Factor of data for Xm
        self.file = file
        self.get_Xm = get_Xm
        assert get_Xm == False
        self.seed = seed
        if file is not None:
            self.all_cont_cols = ['longitude', 'latitude', 'delta_t']
        else:
            self.all_cont_cols = ['longitude', 'latitude', 'temperature', 'humidity', 'wind_speed', 'delta_t']

    def get_suffixes(self, mode):
        suffixes = []
        if 'C' in mode or 'A' in mode:
            suffixes.append('train')
        if 'D' in mode or 'B' in mode:
            suffixes.append('test')
        return suffixes

    def common(self, data):
        X = data[self.Xcols]
        y = data[['PM25_Concentration']]
        return X, y, data

    def rename_cols(self,data):
        data['time'] = data['dateTime']
        data.rename(
            columns={'dateTime': 'delta_t', 'lat': 'latitude', 'long': 'longitude', 'pm2_5': 'PM25_Concentration',
                     'pm10': 'PM10_Concentration'}, inplace=True)

    def load_data(self):
        train_data = None
        offsets = dict()
        prev_offset = 0
        for idx,dt in enumerate(self.file):
            offsets[dt] = dict()
            for suffix in ['train','test']:
                f = Config.data_path + dt + '_f' + self.fold[0] + '_' + suffix + '.csv'
                print('Reading train', f)
                data = pd.read_csv(f)
                if Config.temporal_scaling and idx:
                    data.dateTime += idx * 24 * 60
                train_data = pd.concat((train_data, data))
                offsets[dt][suffix] = (prev_offset,len(train_data))
                prev_offset = len(train_data)
        self.rename_cols(train_data)

        self.X, self.y, self.train_data = self.common(train_data)
        self.offsets = offsets

        self.cont_cols = [i for i in self.Xcols if i in self.all_cont_cols]
        self.cat_indicator = [0 if i in self.cont_cols else 1 for i in self.X.columns]
        self.time_indicator = [1 if i in ['delta_t'] else 0 for i in self.X.columns]

        self.X[self.cont_cols] = self.Xscaler.fit_transform(self.X[self.cont_cols])

    def load_train(self, mode_t, file=None):
        suffixes = self.get_suffixes(mode_t)
        if file is None:
            file = self.file if 'C' in mode_t else self.file[:-1]
        train_data = None
        for idx,dt in enumerate(file):
            for suffix in suffixes:
                if dt == self.file[-1] and suffix == 'test':
                    continue
                for fold in self.fold:
                    f = Config.data_path + dt + '_f' + fold + '_' + suffix + '.csv'
                    print('Reading train', f)
                    data = pd.read_csv(f)
                    if Config.temporal_scaling and idx:
                        data.dateTime += idx * 24 * 60
                    train_data = pd.concat((train_data, data))
        self.rename_cols(train_data)

        X, y, self.train_data = self.common(train_data)

        self.cont_cols = [i for i in self.Xcols if i in self.all_cont_cols]
        X[self.cont_cols] = self.Xscaler.fit_transform(X[self.cont_cols])

        ######
        self.cat_indicator = [0 if i in self.cont_cols else 1 for i in X.columns]
        self.time_indicator = [1 if i in ['delta_t'] else 0 for i in X.columns]

        # print(X.iloc[0])
        # y[y.columns] = self.yscaler.fit_transform(y)
        # Or
        self.y_mean = y.values.mean();y = y - self.y_mean
        self.Xcols = X.columns
        print(X.head(5))
        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)
        if not self.get_Xm:
            return X, y, self.Xcols
        else:
            # tindex = self.train_data.index.unique()
            # print(self.train_data.shape, self.train_data.columns)
            # Xm = self.train_data.loc[tindex[::self.factor], Xcols]
            # Or
            Xm = self.train_data.sample(self.train_data.shape[0]//self.factor, random_state=self.seed)[self.Xcols]
            Xm[Xm.columns] = self.Xscaler.transform(Xm)
            Xm = torch.tensor(Xm.values, dtype=torch.float32)
            return X, y, Xm

    def load_test(self, mode_p, test_file):
        suffixes = self.get_suffixes(mode_p)
        test_data = None
        off = (len(self.file)-1) * 24 * 60
        for suffix in suffixes:
            f = Config.data_path + test_file + '_f' + self.fold[0] + '_{}.csv'.format(suffix)
            print('Reading test', f)
            data = pd.read_csv(f)
            if Config.temporal_scaling:
                data.dateTime += off
            test_data = pd.concat((test_data, data))
        self.rename_cols(test_data)

        X, y, self.test_data = self.common(test_data)

        X[self.cont_cols] = self.Xscaler.transform(X[self.cont_cols])

        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)

        return X, y