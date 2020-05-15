# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 11:54:05 2018

@author: obazgir
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.model_selection import KFold
from keras.layers import Dense ,  Dropout
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import pickle
import Toolbox
from Toolbox import NRMSE, Random_Image_Gen, two_d_norm, two_d_eq, Assign_features_to_pixels, MDS_Im_Gen, Bias_Calc, REFINED_Im_Gen
from scipy.stats import pearsonr
from scipy.stats import pearsonr
import os


## Simulating the data
P = [800]                                                                  # Number of features
Results_Dic = {} 
for p in P:
    
    COV_X = 0.5*np.random.random((p,p))
    COV_X = np.maximum( COV_X, COV_X.transpose())                                   # Generating covariance highly correlated covariance matrix    

    
    for i in range(p):
        if i - int(p/20) < 0:
            COV_X[i,0:i+int(p/20)] =  0.2*np.random.random(i+int(p/20)) + 0.5
        elif i+int(p/20) > p:
            COV_X[i,i-int(p/20):] =  0.2*np.random.random(abs(p-i+int(p/20))) + 0.5
        #else:
         #   COV_X[i,i-int(p/20):i+int(p/20)] =  0.2*np.random.random(int(p/10)) + 0.5
    COV_X = np.maximum( COV_X, COV_X.transpose())         
    np.fill_diagonal(COV_X, 1)        
    Columns_PD = p*[None];  index_PD = p*[None]
    
    for i in range(p):
        Columns_PD[i] = "F" + str(i)
        index_PD[i]   = "F" + str(i)
    
    NN = int(math.sqrt(p)) +1
    
    Samples = [10000]																		# Sample size which could be different as described in the REFINED manuscript
    
    ## Synthetic data
    for n in Samples:
        N = round(n)                                                                        # Number of samples
        X= np.random.multivariate_normal(Mu, COV_X, size = N)
        #X = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=500)          
        SPR_Ratio = [0.2,0.5,0.8]															# Spurious features ratio
        for spr in SPR_Ratio:
            sz = round(spr*p)
            B1 = 3*np.random.random((sz,)) + 6;                   B2 = np.zeros(p-sz);      B = np.concatenate((B1,B2))		# Weights
            Y = np.matmul(X,B)  
            Y = (Y - Y.min())/(Y.max() - Y.min())											# Target values	

            CNN_Dic = {}
            # reading the REFINED coordinates
            with open('theMapping_Synth'+str(p)+'.pickle','rb') as file:
                gene_names,coords,map_in_int = pickle.load(file)
            
            
      
            Results_CNN = np.zeros((5,3))
            i = 0
            # Using 5 fold cross validation for performance measurement
            kf = KFold(n_splits=5)
            for train_index, test_index in kf.split(X):
                X_Train, X_Test = X[train_index], X[test_index]
                Y_Train, Y_Test = Y[train_index], Y[test_index]
                Y_Test = Y_Test.reshape(len(Y_Test),1) 

                ################################################
                
                from keras.layers.core import Activation, Flatten
                from keras.layers.convolutional import Conv2D
                from keras.layers.convolutional import MaxPooling2D
                from keras import backend as K
                from sklearn.model_selection import KFold
                #K.set_image_dim_ordering('th')
                from sklearn.model_selection import train_test_split
                from keras.optimizers import RMSprop, Adam, Adadelta, SGD,Nadam
                from keras.layers.normalization import BatchNormalization
                
                Y_Train_CNN = Y[train_index]
                Y_Test_CNN = Y[test_index];         Y_Test_CNN = Y_Test_CNN.reshape(len(Y_Test_CNN),1)
                
                Width = NN
                Height = NN

                
                nn = math.ceil(np.sqrt(p)) 				     # Image dimension
                Nn = p 	
                
                X_REFINED_Train = REFINED_Im_Gen(X_Train,nn, map_in_int, gene_names,coords)
                X_REFINED_Test = REFINED_Im_Gen(X_Test,nn, map_in_int, gene_names,coords)
                
                Width = nn
                Height = nn
                
                X_Training = X_REFINED_Train.reshape(-1,Width,Height,1)
                X_Testing = X_REFINED_Test.reshape(-1,Width,Height,1)
                # Defining the CNN Model
                
                def CNN_model():
                    nb_filters = 8
                    nb_conv = 3
                    
                    model = Sequential()
                    model.add(Conv2D(nb_filters*1, nb_conv, nb_conv,border_mode='valid',input_shape=(Width, Height,1)))
                    model.add(BatchNormalization())
                    model.add(Activation('relu'))
                    #model.add(MaxPooling2D(pool_size=(2, 2)))

                    model.add(Conv2D(nb_filters*3, nb_conv, nb_conv))
                    model.add(BatchNormalization())
                    model.add(Activation('relu'))
                    #model.add(MaxPooling2D(pool_size=(2, 2)))
                	
                    model.add(Flatten())                       

                    
                    
                    model.add(Dense(256))
                    model.add(BatchNormalization())
                    model.add(Activation('relu'))
                    
                    model.add(Dense(128))
                    model.add(BatchNormalization())
                    model.add(Activation('relu'))
                    model.add(Dropout(1 - 0.7))
                
                    model.add(Dense(1))

                    opt = Adam(lr = 0.0001)
                    model.compile(loss='mse', optimizer = opt)
                    return model
                # Training the CNN Model
                model = CNN_model()
                model.fit(X_Training, Y_Train_CNN, batch_size= 100, epochs = 50, verbose=0)# callbacks = callbacks_list)
                Y_Pred_CNN = model.predict(X_Testing, batch_size= 100, verbose=0)
                
                # Printing out the results
                NRMSE_CNN, MSE_CNN = NRMSE(Y_Test_CNN, Y_Pred_CNN)
                print(NRMSE_CNN, "CNN NRMSE")
                Y_Test_CNN = Y_Test_CNN.reshape(len(Y_Test_CNN),1)
                PearsonCorr_CNN, p_value = pearsonr(Y_Test_CNN, Y_Pred_CNN)
                Results_CNN[i,0] = NRMSE_CNN; Results_CNN[i,1] =  MSE_CNN ; Results_CNN[i,2] = PearsonCorr_CNN
                i = i + 1
                
                    
            NRMSE_CNN = np.mean(Results_CNN[:,0]);  MSE_CNN = np.mean(Results_CNN[:,1]);  Corr_CNN = np.mean(Results_CNN[:,2]);
            
            
            Results_Sample = np.zeros((1,3))
            Results_Sample[0,:] = np.array([NRMSE_CNN,MSE_CNN,Corr_CNN])
            Results = pd.DataFrame(data = Results_Sample , index = ["CNN"], columns = ["NRMSE","MSE","Corr"])
            Results_Dic[spr,n,p] =  Results

with open('Results_Dic'+str(p)+'_5.csv', 'w') as f:[f.write('{0},{1}\n'.format(key, value)) for key, value in Results_Dic.items()]

