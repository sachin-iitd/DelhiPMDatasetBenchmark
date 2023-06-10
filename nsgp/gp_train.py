from lib import *
from data_loader import Data
import gpytorch
from gpytorch.likelihoods import likelihood
from torch.utils.data import TensorDataset, DataLoader
import datetime

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
    if logd['w']:
        arg = 'w'
        logd['w'] = False
    else:
        arg = 'a'
    with open(Config.res_path+common_path+'.txt', arg) as f:
        f.write(' '.join(map(str, args)))
        f.write(end)

### Args

m_name = sys.argv[1]
optim_name = sys.argv[2]
c_fold = sys.argv[3]
nsgp_iters = int(sys.argv[4])
gp_iters = int(sys.argv[5])
restarts = int(sys.argv[6])
div = int(sys.argv[7])
sampling = sys.argv[8]
Xcols = sys.argv[9]
kernel = sys.argv[10]
time_kernel = sys.argv[11]
file_idx = sys.argv[12].split(',')
mode_t = sys.argv[13]
mode_c = sys.argv[14]
mode_p = sys.argv[15]
canada = int(sys.argv[16])

logd = {'w':True}
common_path = 'nsgp_{}{}_fold{}_{}_{}_{}{}'.format(['delhi','canada','canada','usa'][canada], canada if canada in [1,2] else '', c_fold, mode_t + ('T' if Config.temporal_scaling else ''), mode_c, mode_p, '_st' if '_st' in Config.data_path else '')

### Data loading

tmStart = datetime.datetime.now()

if m_name == 'nsgp':
    dataloader = Data(c_fold, Xcols=Xcols, file=file_idx)
    X, y, Xcols = dataloader.load_train(mode_t)
    Xcols = '@'.join([i for i in Xcols])
    X_bar = []
    Num_Ind = 100
    for dim in range(X.shape[1]):
        if X[:, dim].unique().shape[0]<Num_Ind:
            X_bar.append(X[:, dim].unique().reshape(-1,1).to(Config.device))
        else:
            # uniq = X[:, dim].unique().shape[0]
            X_bar.append(f_kmeans(X[:, dim:dim+1], num_inducing_points=Num_Ind, random_state=0).to(Config.device))
    pprint('X Data shape', X.shape, 'X_bar len', len(X_bar), 'X_bar0 shape', X_bar[0].shape)
elif m_name == 'snsgp':
    dataloader = Data(c_fold, get_Xm=True)
    # dataloader.load_data()
    X, y, Xm = dataloader.load_train()
    # pprint('Calculated Inducing locs', Xm.shape, 'out of', X.shape)
    X_bar = f_kmeans(X, num_inducing_points=X.shape[0]//10//dataloader.factor, random_state=0)
    pprint('X Data shape', X.shape, 'Xm shape', Xm.shape, 'X_bar shape', X_bar.shape)
elif m_name == 'gp':
    dataloader = Data(c_fold, Xcols=Xcols)
    # dataloader.load_data()
    X, y, Xcols = dataloader.load_train()
    Xcols = '@'.join([i for i in Xcols])
    pprint('X Data shape', X.shape)

### Move things to cuda
if m_name == 'gp':
    X = X.to(Config.device)
    y = y.to(Config.device)
if m_name == 'nsgp':
    # X_bar = X_bar.to(Config.device)
    pass
elif m_name == 'snsgp':
    X_bar = X_bar.to(Config.device)
    Xm = Xm.to(Config.device)

def closure():
    if m_name in ['nsgp', 'snsgp']: 
        loss = model(X_batch.to(Config.device), y_batch.to(Config.device))
    else:
        output = model(X)
    losses.append(loss.item())
    loss.backward()
    return loss

### Optimizer
if m_name in ['nsgp', 'snsgp']:
    iters = nsgp_iters
else:
    iters = gp_iters

best_state = None
best_loss = np.inf
best_losses = None
pprint('Starting Restarts')

train = TensorDataset(X, y)
torch.manual_seed(0)
if sampling == 'uni':
    trainloader = DataLoader(train, batch_size=X.shape[0]//div, shuffle=True)
else:
    trainloader = DataLoader(train, batch_size=X.shape[0]//div, shuffle=False)

init = time()
chk = True
pprint('Yvar', y.var())
ystd = torch.sqrt(y.var())

def init_vars(model):
    with torch.no_grad():
        if time_kernel in ['loc_periodic', 'periodic']:
            torch.nn.init.uniform_(model.period, a=0.03, b=0.07)
        torch.nn.init.uniform_(model.local_gp_ls, a=0.5, b=3)
        torch.nn.init.uniform_(model.local_gp_std, a=0.5, b=3)
        torch.nn.init.uniform_(model.local_gp_noise_std, a=0.5, b=3)
        for param in model.local_ls:
            torch.nn.init.uniform_(param, a=0.5, b=3)

        torch.nn.init.uniform_(model.global_gp_std, a=0.5, b=3)
        torch.nn.init.uniform_(model.global_gp_noise_std, a=0.5, b=3)

rand = 10**7 + 7

tmPre = (datetime.datetime.now() - tmStart).total_seconds()
tmTrain = 0
for restart in range(restarts):
    tmTrainStart = datetime.datetime.now()
    torch.manual_seed(restart)
    model = NSGP(X_bar, dataloader.cat_indicator, dataloader.time_indicator, random_state=restart, 
            jitter=10**-3, use_Trans=False, m=3, kernel=kernel, timekernel=time_kernel).to(Config.device)
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    init_vars(model)
    losses = []
    init_loss = 0
    for i in range(iters):
        loss_val = 0
        for bi, (X_batch, y_batch) in enumerate(trainloader):
            optim.zero_grad()
            if sampling == 'nn':
                rand_ind = torch.randint(0, X.shape[0], size=(1,))
                batch_inds = torch.argsort(torch.norm(X[rand_ind, :] - X, dim=1))[:X.shape[0]//div]
                X_batch = X[batch_inds]
                y_batch = y[batch_inds]
                if chk:
                    pprint('NN activated', X_batch.shape, y_batch.shape)
                    chk = False
            try:
                if optim_name in ['sg', 'ad']:
                    loss_val += closure().item() + init_loss
                    optim.step()
                else:
                    optim.step(closure)
                with torch.no_grad():
                    for param in model.parameters():
                        torch.clamp_(param, min=10**-5)
            except Exception as e:
                init_loss = np.inf
                pprint('Iter', i, 'Batch', bi, 'Failed with', e, 're inited the params')
        final_loss = loss_val/len(trainloader)
        pprint('iter', i, 'avg loss', final_loss)
    tmTrain += (datetime.datetime.now() - tmTrainStart).total_seconds()

    if final_loss < best_loss:
        best_loss = final_loss
        best_losses = list(losses)
        # torch.save(model.state_dict(), Config.res_path+m_name+'_fold'+c_fold+'.model')

        torch.save(model, Config.res_path+common_path+'.model')

        pd.to_pickle({'loss': best_loss,
                      'losses': best_losses,
                      'dataloader': dataloader,
                      'sampling': sampling,
                      'restart': restart,
                      'tmPre': tmPre,
                      'tmTrain': tmTrain},
                Config.res_path+common_path+'.res')
    pprint('Restart', restart, 'Final loss', final_loss, 'Best loss', best_loss)

with open(Config.res_path+common_path+'.time', 'w') as f:
    f.write(str((time()-init)/60)+' minutes')
pprint('Finished')