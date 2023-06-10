import sys
import os

m_name = sys.argv[1]
optim_name = sys.argv[2]
c_fold = sys.argv[3]
node = sys.argv[4]
nsgp_iters = sys.argv[5]
gp_iters = sys.argv[6]
restarts = sys.argv[7]
div = sys.argv[8]
sampling = sys.argv[9]
Xcols = sys.argv[10]
kernel = sys.argv[11]
time_kernel = sys.argv[12]
mode_t = sys.argv[13]
mode_c = sys.argv[14]
mode_p = sys.argv[15]
canada = sys.argv[16]
dates = sys.argv[17].split(',')

cmd = ' '.join(['python gp_train.py', m_name, optim_name, c_fold, nsgp_iters, gp_iters, restarts, div, sampling, Xcols, kernel, time_kernel, ','.join(dates), mode_t, mode_c, mode_p, canada])
print(cmd)
os.system(cmd)
cmd = ' '.join(['python gp_test.py', m_name, c_fold, sampling, Xcols, kernel, time_kernel, dates[-1], mode_t, mode_c, mode_p, canada])
print(cmd)
os.system(cmd)
