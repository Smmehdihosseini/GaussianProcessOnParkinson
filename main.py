import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sub.utils import *
from sub.models import *

plt.close('all')


parkinson_rawdata = pd.read_csv('data/parkinsons_updrs.csv')

print('***  Lets go for GPR ... \n')

print('1 - Data Loaded ... \n')

dataFrame_describe(parkinson_rawdata)

features=['subject#', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS',
       'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
       'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
       'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']

todrop=['subject#', 'sex', 'test_time',  
       'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
       'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
       'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA']

parkinson_copy = parkinson_rawdata.copy(deep=True)

# Generate The Shuffled Dataframe

np.random.seed(301769)

parkinson_shuffled = shuffle(parkinson_copy, random_seed=301769)

print('2 - Data Shuffled ... \n')

parkinson_shuffled_gpr = parkinson_shuffled.drop(todrop,axis=1)

print('3 - Features Dropped for GPR ... \n')

[Np, Nc] =  parkinson_shuffled_gpr.shape
F = Nc - 1

N_train = int(Np*0.5)
N_val = int(Np*0.25)
N_test = Np - N_train - N_val

parkinson_training = parkinson_shuffled_gpr[0:N_train]

mm = parkinson_training.mean()
ss = parkinson_training.std()

my = mm['total_UPDRS']
sy = ss['total_UPDRS']

# Normalize

parkinson_shuffled_normalized = (parkinson_shuffled_gpr - mm)/ss
y_parkinson = parkinson_shuffled_normalized['total_UPDRS'].values
x_parkinson = parkinson_shuffled_normalized.drop('total_UPDRS', axis=1).values

# Separate Data

x_train_norm = x_parkinson[0:N_train]
y_train_norm = y_parkinson[0:N_train]
y_train = y_train_norm*sy + my

x_val_norm = x_parkinson[N_train:N_train+N_val]
y_val_norm = y_parkinson[N_train:N_train+N_val]
y_val = y_val_norm*sy + my

x_test_norm = x_parkinson[N_train+N_val:]
y_test_norm = y_parkinson[N_train+N_val:]
y_test = y_test_norm*sy + my

print('4 - Data Splitted ... \n')
print('5.1 - Starting Gaussian Process ... \n')

GP = Gaussian_Process(X_train=x_train_norm, Y_train=y_train_norm, X_val=x_val_norm, Y_val=y_val_norm, X_test=x_test_norm, Y_test=y_test_norm)

r2_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
s2_lst = [0.000001, 0.000002, 0.000003, 0.000004, 0.000005, 0.00001, 0.00002, 0.00003, 0.00004, 0.00005]

results = pd.DataFrame(columns=['r2','s2', 'mse_val'])

for s2 in s2_lst:
    for r2 in r2_lst:
        [yhat_sig_train_norm, yhat_sig_val_norm, yhat_sig_test_norm] = GP.run(N=10, r2=r2, s2=s2)
        yhat_val = yhat_sig_val_norm[:,1]*sy + my
        error_val = y_val - yhat_val
        mse_val = mse(error_val)
        results = results.append({'r2':r2 ,'s2':s2, 'mse_val':mse_val}, ignore_index=True)
        print(f'... Finished for S2 = {s2} and R2 = {r2} | MSE : {mse_val} \n')

print('5.2 - Gaussian Process Finished ... \n')

min_index = pd.to_numeric(results['mse_val']).idxmin()

print('5.3 - Minimum MSE Index is: ', min_index)

minimum_stat = results.iloc[min_index]

print('5.4 - Minimum MSE: ', minimum_stat['mse_val'])
print('... R2 Value: ', minimum_stat['r2']) 
print('... S2 Value: ', minimum_stat['s2'])   

for i in results['s2'].unique():
    df = results[results['s2']==i]
    plot_err(df, i, logy=0, logx=0)
    plt.legend()

plt.plot(minimum_stat['r2'], minimum_stat['mse_val'], 'ro')
plt.xlabel('r2')
plt.ylabel('MSE Validation')
plt.grid()
plt.savefig('Optimization.png')
plt.show()
 
print('5.5 - Optimization Figure Plotted ... \n')     


[yhat_sig_train_norm, yhat_sig_val_norm, yhat_sig_test_norm] = GP.run(N=10, r2=minimum_stat['r2'], s2=minimum_stat['s2'])

yhat_train_gp = yhat_sig_train_norm[:,1]*sy + my
yhat_val_gp = yhat_sig_val_norm[:,1]*sy + my
yhat_test_gp = yhat_sig_test_norm[:,1]*sy + my
sigma_test_gp = yhat_sig_test_norm[:,0]

error_train_gp = y_train - yhat_train_gp
error_val_gp = y_val - yhat_val_gp
error_test_gp = y_test - yhat_test_gp

plot_versus(y_test, yhat_test_gp, title='Gaussian Process Regression: Test Data', errorbar=False, sigmahat=None, sy=None)
plot_versus(y_test, yhat_test_gp, title='Gaussian Process Regression: Test Data - With Errorbars', errorbar=True, sigmahat=sigma_test_gp, sy=sy)

errors_gp = [error_train_gp, error_val_gp, error_test_gp]
plot_histogram(errors=errors_gp, title='Gaussian Process Regression: Error Histogram', label=['Training','Validation','Test'])

print('5.6 Results for GPR: ')
print('... MSE Training', mse(error_train_gp))
print('... MSE Validation', mse(error_val_gp))
print('... MSE Test', mse(error_test_gp))

print('\n... Mean Error Training', mean_err(error_train_gp))
print('... Mean Error Validation', mean_err(error_val_gp))
print('... Mean Error Test', mean_err(error_test_gp))

print('\n... St Dev Error Training', st_dev_err(error_train_gp))
print('... St Dev Error Validation', st_dev_err(error_val_gp))
print('... St Dev Error Test', st_dev_err(error_test_gp))

print('\n... R2 Training', r2_score(y_train, error_train_gp))
print('... R2 Validation', r2_score(y_val, error_val_gp))
print('... R2 Test', r2_score(y_test, error_test_gp))

print('***  Lets go for LLS ... \n')

todrop=['subject#', 'test_time']

parkinson_shuffled_lls = parkinson_shuffled.drop(todrop,axis=1)

print('1 - Features Dropped for LLS ... \n')

[Np, Nc] =  parkinson_shuffled_lls.shape
F = Nc - 1

N_train = int(Np*0.5)
N_val = int(Np*0.25)
N_test = Np - N_train - N_val

parkinson_training = parkinson_shuffled_lls[0:N_train]

mm = parkinson_training.mean()
ss = parkinson_training.std()

my = mm['total_UPDRS']
sy = ss['total_UPDRS']

# Normalize

parkinson_shuffled_normalized = (parkinson_shuffled_lls - mm)/ss
y_parkinson = parkinson_shuffled_normalized['total_UPDRS'].values
x_parkinson = parkinson_shuffled_normalized.drop('total_UPDRS', axis=1).values

# Separate Data

x_train_norm = x_parkinson[0:N_train]
y_train_norm = y_parkinson[0:N_train]
y_train = y_train_norm*sy + my

x_val_norm = x_parkinson[N_train:N_train+N_val]
y_val_norm = y_parkinson[N_train:N_train+N_val]
y_val = y_val_norm*sy + my

x_test_norm = x_parkinson[N_train+N_val:]
y_test_norm = y_parkinson[N_train+N_val:]
y_test = y_test_norm*sy + my

print('2 - Data Splitted ... \n')


print('3.1 - Starting LLS Model ... \n')

LLS = LLS_Model(X_train=x_train_norm, Y_train=y_train_norm, X_test=x_test_norm, Y_test=y_test_norm)
[yhat_train_lls_norm, yhat_test_lls_norm] = LLS.run()

print('3.2 - LLS Finished ... \n')

yhat_train_lls = yhat_train_lls_norm*sy + my
yhat_test_lls = yhat_test_lls_norm*sy + my

error_train_lls = y_train - yhat_train_lls
error_test_lls = y_test - yhat_test_lls

plot_versus(y_test, yhat_test_lls, title='Linear Least Squares: Test Data')

errors_lls = [error_train_lls, error_test_lls]
plot_histogram(errors=errors_lls, title='Linear Least Squares: Error Histogram', label=['Training', 'Test'])

print('3.3 Results for LLS: ')
print('... MSE Training', mse(error_train_lls))
print('... MSE Test', mse(error_test_lls))

print('\n... Mean Error Training', mean_err(error_train_lls))
print('... Mean Error Test', mean_err(error_test_lls))

print('\n... St Dev Error Training', st_dev_err(error_train_lls))
print('... St Dev Error Test', st_dev_err(error_test_lls))

print('\n... R2 Training', r2_score(y_train, error_train_lls))
print('... R2 Test', r2_score(y_test, error_test_lls))
