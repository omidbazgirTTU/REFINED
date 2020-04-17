# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 15:29:12 2019

@author: obazgir
"""

import numpy as np
import pandas as pd
import os
from scipy.stats import pearsonr
from sklearn.metrics import median_absolute_error
import scipy
import Toolbox
from Toolbox import NRMSE, dataframer, GDSC_dataframer, GDSC_NPier,REFINED_Im_Gen
import math
import pickle


# Loading process data
with open('Data_GDSC_CELL_1212.pickle','rb') as file:									# Load training data as tuple
    X_Cell_Train,X_Cell_Val,X_Cell_Test = pickle.load(file)
    
with open('Data_GDSC_Drug.pickle','rb') as file:										# Load validation data as tuple
    X_Drug_Train,X_Drug_Val,X_Drug_Test = pickle.load(file)
    
with open('Data_GDSC_1212.pickle','rb') as file:										# Load test data as tuple
    X_Train,X_Validation,X_Test, Y_Train, Y_Val, Y_Test = pickle.load(file)

# Saving the results into vectores 
Results_Models = np.zeros((5,3))
Results_Points = np.zeros((len(Y_Test),8))
Results_Points[:,0] = Y_Val;    Results_Points[:,1] = Y_Test.reshape(len(Y_Test))

#############
## RANDOM  ##
#############
from Toolbox import Random_position, Random_Image_Gen
# Randomly generating coordinates for the cell line and drug data using the Random_position function provided in the toolbox
# Cell    
sz = X_Cell_Train.shape
p = sz[1];  
Rand_Pos_mat= Random_position(p)
X_Cell_Train_Rand = Random_Image_Gen(X_Cell_Train, Rand_Pos_mat)
X_Cell_Val_Rand = Random_Image_Gen(X_Cell_Val, Rand_Pos_mat)
X_Cell_Test_Rand = Random_Image_Gen(X_Cell_Test, Rand_Pos_mat)

# Drug
sz = X_Drug_Train.shape
p = sz[1];  
Rand_Pos_mat= Random_position(p)
X_Drug_Train_Rand = Random_Image_Gen(X_Drug_Train, Rand_Pos_mat)
X_Drug_Val_Rand = Random_Image_Gen(X_Drug_Val, Rand_Pos_mat)
X_Drug_Test_Rand = Random_Image_Gen(X_Drug_Test, Rand_Pos_mat)

#############
##   PCA   ##
#############
# Generating coordinates in 2D space using PCA for both cell line data and drug data. The coordinates are generated using the training set and will be used for validation and test sets as well.
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

# Cell
PCA_in_Cell = pd.DataFrame(X_Cell_Train)
pca_xy_cell = pca.fit_transform(PCA_in_Cell.T)
nn = math.ceil(np.sqrt(X_Cell_Train.shape[1]))

# Normalizing the PCA coordinates to fit in a [0,n] dimensional 2D image
pca_xy_cell[:,0] = np.round((pca_xy_cell[:,0] - min(pca_xy_cell[:,0]))*(nn-1)/(max(pca_xy_cell[:,0]) - min(pca_xy_cell[:,0])))
pca_xy_cell[:,1] = np.round((pca_xy_cell[:,1] - min(pca_xy_cell[:,1]))*(nn-1)/(max(pca_xy_cell[:,1]) - min(pca_xy_cell[:,1])))

pca_xy_cell = pca_xy_cell.astype(int)
X_Cell_Train_PCA = Random_Image_Gen(X_Cell_Train, pca_xy_cell)
X_Cell_Val_PCA   = Random_Image_Gen(X_Cell_Val, pca_xy_cell)
X_Cell_Test_PCA  = Random_Image_Gen(X_Cell_Test, pca_xy_cell)

# Drug
PCA_in_Drug = pd.DataFrame(X_Drug_Train)
pca_xy_Drug = pca.fit_transform(PCA_in_Drug.T)
nn = math.ceil(np.sqrt(X_Drug_Train.shape[1]))

# Normalizing the PCA coordinates to fit in a [0,n] dimensional 2D image
pca_xy_Drug[:,0] = np.round((pca_xy_Drug[:,0] - min(pca_xy_Drug[:,0]))*(nn-1)/(max(pca_xy_Drug[:,0]) - min(pca_xy_Drug[:,0])))
pca_xy_Drug[:,1] = np.round((pca_xy_Drug[:,1] - min(pca_xy_Drug[:,1]))*(nn-1)/(max(pca_xy_Drug[:,1]) - min(pca_xy_Drug[:,1])))

pca_xy_Drug = pca_xy_Drug.astype(int)
X_Drug_Train_PCA = Random_Image_Gen(X_Drug_Train, pca_xy_Drug)
X_Drug_Val_PCA   = Random_Image_Gen(X_Drug_Val, pca_xy_Drug)
X_Drug_Test_PCA  = Random_Image_Gen(X_Drug_Test, pca_xy_Drug)


#################
##   REFINED   ##
#################
# The REFINED coordinates generated using the mpiHill_Hardcoded.py will be saved and loaded as pickle file then thos coordinates will be used for training, validation and test set to create the REFINED images.
# Note that, the REFINED coordinates of the Cell line data and Drug data are different. Therefore, they have to be loaded and used separately, as the REFINED CNN model has two separate arms, where one of them handles
# the cell line data and the other one handles the drug data.
# Cell
with open('REFINED_GDSC_Cell_1212.pickle','rb') as file:
    gene_names_cell,coords_cell,map_in_int_cell = pickle.load(file)
    
nn = math.ceil(np.sqrt(X_Cell_Train.shape[1]))
X_Cell_Train_REFINED = REFINED_Im_Gen(X_Cell_Train,nn,  map_in_int_cell, gene_names_cell,coords_cell)
X_Cell_Val_REFINED = REFINED_Im_Gen(X_Cell_Val,nn,  map_in_int_cell, gene_names_cell,coords_cell)
X_Cell_Test_REFINED = REFINED_Im_Gen(X_Cell_Test,nn,  map_in_int_cell, gene_names_cell,coords_cell)
# Drug
with open('REFINED_GDSC_Drug.pickle','rb') as file:
    gene_names_drug,coords_drug,map_in_int_drug = pickle.load(file)
    
nn = math.ceil(np.sqrt(X_Drug_Train.shape[1]))
X_Drug_Train_REFINED = REFINED_Im_Gen(X_Drug_Train,nn, map_in_int_drug, gene_names_drug,coords_drug)
X_Drug_Val_REFINED = REFINED_Im_Gen(X_Drug_Val,nn, map_in_int_drug, gene_names_drug,coords_drug)
X_Drug_Test_REFINED = REFINED_Im_Gen(X_Drug_Test,nn, map_in_int_drug, gene_names_drug,coords_drug)

#####################
###    CNN Model  ###
#####################
import keras
from keras.models import Sequential
from keras.layers import Dense ,  Dropout
from keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import layers, models

   
Model_Names = ["Random","PCA","REFINED"]

for Model_i in range(len(Model_Names)):
    if Model_i == 0:
        Shape1 = X_Drug_Train_Rand.shape
        Width1 = int(math.sqrt(Shape1[1]));     Height1 = int(math.sqrt(Shape1[1]))
        
        Shape2 = X_Cell_Train_Rand.shape
        Width2 = int(math.sqrt(Shape2[1]));     Height2 = int(math.sqrt(Shape2[1]))
        # Reshaping the data
        IM_Training = X_Drug_Train_Rand.reshape(-1,Width1,Height1,1)
        IM_Valid = X_Drug_Val_Rand.reshape(-1,Width1,Height1,1)
        IM_Testing = X_Drug_Test_Rand.reshape(-1,Width1,Height1,1)
        
        
        IM_Training2 = X_Cell_Train_Rand.reshape(-1,Width2,Height2,1)
        IM_Valid2 = X_Cell_Val_Rand.reshape(-1,Width2,Height2,1)
        IM_Testing2 = X_Cell_Test_Rand.reshape(-1,Width2,Height2,1)
                
    elif Model_i == 1:
        Shape1 = X_Drug_Train_PCA.shape
        Width1 = int(math.sqrt(Shape1[1]));     Height1 = int(math.sqrt(Shape1[1]))
        
        Shape2 = X_Cell_Train_PCA.shape
        Width2 = int(math.sqrt(Shape2[1]));     Height2 = int(math.sqrt(Shape2[1]))
        # Reshaping the data
        IM_Training = X_Drug_Train_PCA.reshape(-1,Width1,Height1,1)
        IM_Valid = X_Drug_Val_PCA.reshape(-1,Width1,Height1,1)
        IM_Testing = X_Drug_Test_PCA.reshape(-1,Width1,Height1,1)
        
        
        IM_Training2 = X_Cell_Train_PCA.reshape(-1,Width2,Height2,1)
        IM_Valid2 = X_Cell_Val_PCA.reshape(-1,Width2,Height2,1)
        IM_Testing2 = X_Cell_Test_PCA.reshape(-1,Width2,Height2,1)

        
    else:
        Shape1 = X_Drug_Train_REFINED.shape
        Width1 = int(math.sqrt(Shape1[1]));     Height1 = int(math.sqrt(Shape1[1]))
        
        Shape2 = X_Cell_Train_REFINED.shape
        Width2 = int(math.sqrt(Shape2[1]));     Height2 = int(math.sqrt(Shape2[1]))
        # Reshaping the data
        IM_Training = X_Drug_Train_REFINED.reshape(-1,Width1,Height1,1)
        IM_Valid = X_Drug_Val_REFINED.reshape(-1,Width1,Height1,1)
        IM_Testing = X_Drug_Test_REFINED.reshape(-1,Width1,Height1,1)
        
        
        IM_Training2 = X_Cell_Train_REFINED.reshape(-1,Width2,Height2,1)
        IM_Valid2 = X_Cell_Val_REFINED.reshape(-1,Width2,Height2,1)
        IM_Testing2 = X_Cell_Test_REFINED.reshape(-1,Width2,Height2,1)


    def CNN_model():

        
        input1 = layers.Input(shape = (Width1, Height1,1))
        x1 = layers.Conv2D(60, (5, 5),padding='valid',strides=2,dilation_rate=1)(input1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Activation('relu')(x1)
        x1 = layers.Conv2D(72, (6, 6),padding='valid',strides=2,dilation_rate=1)(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Activation('relu')(x1)
        x1 = layers.Conv2D(72, (5, 5),padding='valid',strides=1,dilation_rate=1)(x1)
        x1 = layers.BatchNormalization()(x1)
        x1 = layers.Activation('relu')(x1)
        #x1 = layers.MaxPooling2D(pool_size=(2,2))(x1)
        Out1 = layers.Flatten()(x1)
        
        input2 = layers.Input(shape = (Width2, Height2,1))
        y1 = layers.Conv2D(60, (5, 5),padding='valid',strides=2,dilation_rate=1)(input2)
        y1 = layers.BatchNormalization()(y1)
        y1 = layers.Activation('relu')(y1)
        y1 = layers.Conv2D(72, (6, 6),padding='valid',strides=2,dilation_rate=1)(y1)
        y1 = layers.BatchNormalization()(y1)
        y1 = layers.Activation('relu')(y1)
        y1 = layers.Conv2D(72, (6, 6),padding='valid',strides=2,dilation_rate=1)(y1)
        y1 = layers.BatchNormalization()(y1)
        y1 = layers.Activation('relu')(y1)
        #y1 = layers.MaxPooling2D(pool_size=(2,2))(y1)
        Out2 = layers.Flatten()(y1)
        
        x = layers.concatenate([Out1, Out2])
        
        x = layers.Dense(305)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(1- 0.7)(x)
        
        x = layers.Dense(175)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(1- 0.7)(x)
    

        
        Out = layers.Dense(1)(x)
        model = tf.keras.Model(inputs = [input1, input2], outputs = [Out])
        
        initial_learning_rate = 8.05282076787473e-05
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=438886.2132672289,         # Total number of steps = (trainingsize*epochs)/(batchsize) 
            decay_rate=0.5947358218407578,
            staircase=True)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                      loss='mse')

        return model
    # Training the CNN Model
    CNNmodel = CNN_model()
    ES = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
    CNN_History = CNNmodel.fit([IM_Training, IM_Training2] , Y_Train , batch_size= 128, epochs = 50, verbose=0, validation_data=([IM_Valid , IM_Valid2], Y_Val ),callbacks = [ES])# callbacks = callbacks_list)
    Y_Pred_Val_CNN = CNNmodel.predict([IM_Valid , IM_Valid2], batch_size= 128, verbose=0)
    Y_Pred_CNN = CNNmodel.predict([IM_Testing,IM_Testing2] , batch_size= 128, verbose=0)
    
    
    
    CNN_NRMSE, CNN_R2 = NRMSE(Y_Test, Y_Pred_CNN)
    Y_Pred_CNN = Y_Pred_CNN.reshape(len(Y_Pred_CNN),1)
    Y_Test = Y_Test.reshape(len(Y_Test),1)
    CNN_PCC, RF_p_value = pearsonr(Y_Test, Y_Pred_CNN)
    CNN_MAE = median_absolute_error(Y_Test, Y_Pred_CNN)
    CNN_Dist_Corr = scipy.spatial.distance.correlation(Y_Test, Y_Pred_CNN) 
  
    print(CNN_NRMSE, "CNN NRMSE of " + Model_Names[Model_i])
    print(CNN_R2, "CNN R2 of " + Model_Names[Model_i])
    print(CNN_PCC,"CNN PCC of " + Model_Names[Model_i])
    print(CNN_MAE,"CNN MAE of " + Model_Names[Model_i])
    print(CNN_Dist_Corr,"CNN Distance Correlation of " + Model_Names[Model_i])
    
    Results_Models[0, Model_i] = CNN_NRMSE;   Results_Models[1, Model_i] = CNN_PCC;  Results_Models[2, Model_i] = CNN_R2
    Results_Models[3, Model_i] = CNN_MAE;     Results_Models[4, Model_i] = CNN_Dist_Corr
    
    Y_Pred_Val_CNN = Y_Pred_Val_CNN.reshape(len(Y_Pred_Val_CNN))
    Y_Pred_CNN = Y_Pred_CNN.reshape(len(Y_Pred_CNN))
    Results_Points[:,2 + 2*Model_i] = Y_Pred_Val_CNN;    Results_Points[:,3 + 2*Model_i] = Y_Pred_CNN
    
Results = pd.DataFrame(data = Results_Models, columns = ['CNN Random','CNN PCA', 'CNN REFINED'] 
                       ,index = ['NRMSE','PCC','R2','MAE','Dist Corr'])
PD_Results_Points = pd.DataFrame(data =  Results_Points, columns = ['Y_Val', 'Y_Test','Y_Val_Rand','Y_Test_Rand','Y_Val_PCA','Y_Test_PCA','Y_Val_REFINED','Y_Test_REFINED'])

