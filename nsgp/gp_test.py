import numpy as np
from lib import *
import gpytorch
import os
import datetime
from sklearn.metrics import mean_squared_error

### GP Model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def pprint(*args, end='\n'):
    print(*args, end=end)
    with open(Config.res_path+common_path+'.txt', 'a') as f:
        f.write(' '.join(map(str, args)))
        f.write(end)

tmPre = datetime.datetime.now()

m_name = sys.argv[1]
c_fold = sys.argv[2]
sampling = sys.argv[3]
Xcols = sys.argv[4]
kernel = sys.argv[5]
timekernel = sys.argv[6]
file_idx = sys.argv[7]
mode_t = sys.argv[8]
mode_c = sys.argv[9]
mode_p = sys.argv[10]
canada = int(sys.argv[11])

common_path = 'nsgp_{}{}_fold{}_{}_{}_{}{}'.format(['delhi','canada','canada','usa'][canada], canada if canada in [1,2] else '', c_fold, mode_t + ('T' if Config.temporal_scaling else ''), mode_c, mode_p, '_st' if '_st' in Config.data_path else '')
train_res = pd.read_pickle(Config.res_path+common_path+'.res')

dataloader = train_res['dataloader']

# Context Data
if mode_c == 'C':
    X, y = dataloader.load_test(mode_c, file_idx)
else:
    X, y, _ = dataloader.load_train(mode_t)

# Predict data
test_X, test_y = dataloader.load_test(mode_p, file_idx)

model = torch.load(Config.res_path+common_path+'.model')
model.eval()

tmPre = (datetime.datetime.now() - tmPre).total_seconds()
tmTest = datetime.datetime.now()

with torch.no_grad():
    pred_y, pred_var = model.predict(X.to(Config.device), y.to(Config.device),
                                    test_X.to(Config.device))
    train_y, train_var = model.predict(X.to(Config.device), y.to(Config.device),
                                     X.to(Config.device))
tmTest = (datetime.datetime.now() - tmTest).total_seconds()

dataloader.test_data['pred_mean'] = pred_y.cpu().ravel() + dataloader.y_mean

if m_name in ['nsgp', 'snsgp']:
    dataloader.test_data['pred_var'] = pred_var.diagonal().cpu()
else:
    dataloader.test_data['pred_var'] = pred_var.cpu()

# dataloader.test_data.to_csv(Config.res_path+common_path+'.csv')

rmse = mean_squared_error(test_y, dataloader.test_data['pred_mean'], squared=False)
rmseTrain = mean_squared_error(y.cpu().ravel(), train_y.cpu().ravel(), squared=False) # + dataloader.y_mean
pred_var = pred_var.diagonal()
minVar = pred_var.min().item()
minVarAbs = pred_var.abs().min().item()
maxVar = pred_var.max().item()
avgVar = pred_var.mean().item()
varVar = pred_var.var().item()

mfile = Config.res_path+common_path+'.metrics.csv'
if not os.path.exists(mfile):
    with open(mfile, 'w') as f:
        print('Date,Fold,TestRMSE,MinVar,MinVarAbs,MaxVar,AvgVar,VarVar,TrainRMSE,FinalLoss,AvgLoss,tmPre,tmTrain,tmPre2,tmTest', file = f)
with open(mfile, 'a') as f:
    print('{}|{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}'
          .format('|'.join(dataloader.file),file_idx, c_fold, rmse, minVar, minVarAbs, maxVar, avgVar, varVar, rmseTrain, train_res['loss'], np.mean(train_res['losses']), train_res['tmPre'], train_res['tmTrain'], tmPre, tmTest), file=f)

pprint('Finished')