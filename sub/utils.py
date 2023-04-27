import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

status = isnotebook()

if status==True:
    class color:
        PURPLE = '\033[95m'
        CYAN = '\033[96m'
        DARKCYAN = '\033[36m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'
elif status==False:
    class color:
        PURPLE = ''
        CYAN = ''
        DARKCYAN = ''
        BLUE = ''
        GREEN = ''
        YELLOW = ''
        RED = ''
        BOLD = ''
        UNDERLINE = ''
        END = ''
    
def dataFrame_describe(df):
    desc = df.describe().T
    desc_count = pd.DataFrame(desc['count'])
    
    print(color.BOLD+color.RED+"\n *** Data Description: \n \n"+color.END, df.describe().T)
    
    print(color.BOLD+color.RED+'\n *** Check for Missing Values!'+color.END)

    if desc_count.value_counts().shape[0]==1:
        print(color.BOLD+color.GREEN+"\n No Missing Data! Continue ... \n"+color.END)
    else:
        print(color.BOLD+color.YELLOW+"*** There Are Some Columns With Missing Values! Checking ...\n"+color.END)
        columns_list = []    
        for i in df.columns:
            if desc_count['count'].loc[i]!=desc['count'].value_counts().index[0]:
                columns_list.append(i)
        print(color.BOLD+color.YELLOW+'\n *** Here is the list of columns which have missing values! Check them please! \n'+color.END, columns_list)
        
    print(color.BOLD+color.RED+'\n *** Let\'s Print DataFrame Info! \n'+color.END)
    
    df.info()
    
    if df.dtypes.value_counts().shape[0]>1:
        print(color.BOLD+color.YELLOW+'\n *** We Have A Mixture of Datatypes in This DataFrame! \n'+color.END)
        
    
    for val, cnt in df.dtypes.value_counts().iteritems():
        print(color.BOLD+color.RED+f' - There are {cnt} columns in which we have got {val} datatype! \n'+color.END)
        

def normalizer(X, mean, std, mean_n_std=True):
    X_Norm = (X-mean)/std
    if mean_n_std==True:
        return X_Norm, mean, std
    else:
        return X_Norm
    
def shuffle(data, random_seed=1):
    np.random.seed(random_seed)
    index_shuffle = np.arange(data.shape[0])
    np.random.shuffle(index_shuffle) # Shuffling indexes
    data_shuffled = data.copy(deep=True)
    data_shuffled = data_shuffled.set_axis(index_shuffle,axis=0,inplace=False).sort_index(axis=0)
    return data_shuffled
  

def plot_versus(y, yhat, title, errorbar=False, sigmahat=None, sy=None):

    plt.figure()

    if errorbar:
        plt.errorbar(y, yhat, yerr = 3*sigmahat*sy, fmt='o', ms=2)
    else:
        plt.plot(y, yhat, '.b')

    plt.plot(y,y, 'r')
    plt.grid()
    plt.xlabel('y')
    plt.ylabel(r'$\^{y}$')
    plt.title(title)
    plt.show()
    v = plt.axis()
    N1 = (v[0] + v[1])*0.5
    N2 = (v[2] + v[3])*0.5


def plot_histogram(errors, title, label): 

    plt.figure()
    plt.hist(errors, bins=50, density=True, range=[-8,17], histtype='bar', label=label)
    plt.xlabel('Error')
    plt.ylabel('P(Error in Bin)')
    plt.legend()
    plt.grid()
    plt.title(title)
    plt.show()
    v = plt.axis()
    N1 = (v[0] + v[1])*0.5
    N2 = (v[2] + v[3])*0.5

def plot_err(df, i, logy, logx):

    if (logy == 0) & (logx == 0):
        plt.plot(df['r2'], df['mse_val'], label = f"S2 = {i}")
    if (logy == 1) & (logx == 0):
        plt.semilogy(df['r2'], df['mse_val'], label = f"S2 = {i}")           
    if (logy == 0) & (logx == 1):
        plt.semilogx(df['r2'], df['mse_val'], label = f"S2 = {i}")          
    if (logy == 1) & (logx == 1):
        plt.loglog(df['r2'], df['mse_val'], label = f"S2 = {i}")

def mse(error, r=3):
    return round(np.mean((error)**2), r)

def mean_err(error, r=4):
    return round(np.mean(error), r)

def st_dev_err(error, r=3):
    return round(np.std(error), r)

def r2_score(y, error, r=4):
    return round(1-np.mean(error**2)/np.std(y**2), r)