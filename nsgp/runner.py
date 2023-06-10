import os
import sys
from config import Config
from dates import dates_usa

mode_t = ['AB','AC','C','ABC'][1]
mode_c = ['AB','AC','C','ABC'][1]
mode_p = ['D','CD'][0]
canada = -1
nTrainDays = 3 if 'A' in mode_t else 0

model_name = 'nsgp'
optim_name = 'ad'
c_fold = int(sys.argv[1]) if len(sys.argv)>1 else 0
canada = int(sys.argv[2]) if len(sys.argv)>2 else canada
node = 'gpu1'
nsgp_iters = 40
gp_iters = 50
restarts = 5
div = 4
sampling = 'uni' # cont, nn, uni
Xcols = '@'.join(['longitude', 'latitude', 'delta_t'])
kernel = 'rbf' # Order: RBF, M32
time_kernel = 'local_per' # Order RBF, loc_per

def xform_day(day):
    arr = [0, 30, 61]
    w = 0 if day <= 30 else 1 if day <= 61 else 2
    mon = ['2020-11-', '2020-12-', '2021-01-'][w]
    date = mon + '{:02d}'.format(day - arr[w])
    return date

def create_cmd(date, canada=0):
    cmd = ' '.join(['python run.py', model_name, optim_name, str(c_fold), node, str(nsgp_iters),str(gp_iters),
                    str(restarts), str(div), sampling, Xcols, kernel, time_kernel, mode_t, mode_c, mode_p, str(canada), ','.join(date)])
    return cmd

def check_break():
    if os.path.exists('break'):
        exit(1)

def proc_canada(canada):
    if canada == 1:
        dates = ['2015-02-12', '2015-02-13', '2015-03-19', '2015-03-25', '2015-03-27', '2015-04-01', '2015-04-10', '2015-04-15',
         '2015-04-17', '2015-04-21', '2015-04-24', '2015-05-12',  '2015-05-14', '2015-05-20', '2015-05-22', '2015-09-10', '2015-09-23', '2015-11-23']
    else:
        dates = ['canada_20{:02d}'.format(i) for i in range(6,17)]
    for day in range(nTrainDays, len(dates)):
        date = []
        for i in range(nTrainDays,-1,-1):
            date.append(dates[day-i])
        cmd = create_cmd(date, canada)
        print(cmd)
        os.system(cmd)
        check_break()

def proc_delhi():
    nTestStartDay = 15
    nTotalDays = 91
    for day in range(nTestStartDay, nTotalDays+1):
        date = []
        for i in range(nTrainDays,-1,-1):
            date.append(xform_day(day-i))
        cmd = create_cmd(date)
        print(cmd)
        os.system(cmd)
        check_break()

def proc_usa(country):
    dates = dates_usa
    for day in range(nTrainDays, len(dates)):
        date = []
        for i in range(nTrainDays, -1, -1):
            date.append(dates[day - i])
        cmd = create_cmd(date, country)
        print(cmd)
        os.system(cmd)
        check_break()


if canada < 0:
    proc_delhi()
    proc_canada(1)
    proc_canada(2)
    proc_usa(3)
elif canada == 0:
    proc_delhi()
elif canada == 3:
    proc_usa(3)
else:
    proc_canada(canada)
