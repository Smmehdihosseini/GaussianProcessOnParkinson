import pandas as pd
import numpy as np

class LLS_Model:
    
    def __init__(self, X_train, Y_train, X_test, Y_test):

        self.matr_train = X_train
        self.matr_test = X_test


        self.vect_train = Y_train
        self.vect_test = Y_test
        
    def run(self):
        
        X_train = self.matr_train
        Y_train = self.vect_train
        
        X_test = self.matr_test
        Y_test = self.vect_test
       
        self.out_lst_lls = []
                 
        weights = np.dot(np.dot(np.linalg.inv(np.dot(X_train.T, X_train)), X_train.T), Y_train)

        yhat_train = np.dot(X_train, weights)
        yhat_test = np.dot(X_test, weights)
        
        self.out_lst_lls = [yhat_train, yhat_test]
        return self.out_lst_lls


class Gaussian_Process:

    def __init__(self, X_train, Y_train, X_val, Y_val, X_test, Y_test):
	
        self.matr_train = X_train
        self.matr_val = X_val
        self.matr_test = X_test
        self.vect_train = Y_train
        self.vect_val = Y_val
        self.vect_test = Y_test
   
    def run(self, N=10, r2=3, s2=1e-3):
        
        train_set = self.matr_train
        val_set = self.matr_val
        test_set = self.matr_test
        train_lab = self.vect_train
        val_lab = self.vect_val
        test_lab = self.vect_test         
        
        set_lst = [train_set, val_set, test_set]
        lab_lst = [train_lab, val_lab, test_lab]

        shape_train = train_set.shape[0]
        shape_val = val_set.shape[0]
        shape_test = test_set.shape[0]
        
        self.yhat_sig_train = np.zeros((shape_train,2))
        self.yhat_sig_val = np.zeros((shape_val,2))
        self.yhat_sig_test = np.zeros((shape_test,2))
      
        self.out_lst_gp = [self.yhat_sig_train, self.yhat_sig_val, self.yhat_sig_test]


        for set_sel in range(len(set_lst)):
            shape_set_sel = set_lst[set_sel].shape[0]
            X = set_lst[set_sel]

            for i_X in range(shape_set_sel):
                X_part = X[i_X,:]
                A = train_set - np.ones((shape_train, 1))*X_part
                dist2 = np.sum(A**2, axis=1)
                ii = np.argsort(dist2)
                ii = ii[0:N-1]
                refX = train_set[ii,:]
                Z = np.vstack((refX, X_part))
                sc = np.dot(Z, Z.T)
                e = np.diagonal(sc).reshape(N, 1)
                D = e + e.T - 2*sc
                R_N = np.exp(-D/(2*r2)) + s2*np.identity(N)
                R_Nm1 = R_N[0:N-1,0:N-1]
                K = R_N[0:N-1,N-1]
                d = R_N[N-1,N-1]
                C = np.linalg.inv(R_Nm1)
                refY = train_lab[ii]
                Mu = K.T@C@refY
                sig2 = d - K.T@C@K
                self.out_lst_gp[set_sel][i_X,0] = np.sqrt(sig2)
                self.out_lst_gp[set_sel][i_X,1] = Mu

        return self.out_lst_gp
    