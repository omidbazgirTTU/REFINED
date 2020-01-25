import csv
import numpy as np
import pandas as pd
import os
import scipy as sp
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
from sklearn.metrics import median_absolute_error
import cv2
import Toolbox
from Toolbox import Reg_to_Class, floattoint
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import pickle
##########################################
#                                        #                                              
#                                        #                               
#               Data Cleaning            #   
#                                        #   
##########################################



cell_lines = ["HCC_2998","MDA_MB_435", "SNB_78", "NCI_ADR_RES","DU_145", "786_0", "A498","A549_ATCC","ACHN","BT_549","CAKI_1","DLD_1","DMS_114","DMS_273","CCRF_CEM","COLO_205","EKVX"]
Results_Dic = {} 

#%%
for SEL_CEL in cell_lines:
    Class_Res = np.zeros([1,5])
    DF = pd.read_csv("NCI60_GI50_normalized_April.csv")
    FilteredDF = DF.loc[DF.CELL==SEL_CEL]											# Pulling out the selected cell line responses
    FilteredDF = FilteredDF.drop_duplicates(['NSC'])                                # Dropping out the duplicates
    
    Feat_DF = pd.read_csv("normalized_padel_feats_NCI60_672.csv")
    Cell_Features = Feat_DF[Feat_DF.NSC.isin(FilteredDF.NSC)]
    TargetDF = FilteredDF[FilteredDF.NSC.isin(Cell_Features.NSC)]
    
    Y = np.array(TargetDF.NORMLOG50)
    # Features
    X = Cell_Features.values
    X = X[:,2:]
    # fix random seed for reproducibility
    seed = 10
    np.random.seed(seed)
    # split training, validation and test sets based on each sample NSC ID
    NSC_All = np.array(TargetDF['NSC'],dtype = int)
    Train_Ind, Rest_Ind, Y_Train, Y_Rest = train_test_split(NSC_All, Y, test_size= 0.2, random_state=seed)
    Validation_Ind, Test_Ind, Y_Validation, Y_Test = train_test_split(Rest_Ind, Y_Rest, test_size= 0.5, random_state=seed)
    # Sort the NSCs
    Train_Ind = np.sort(Train_Ind)
    Validation_Ind = np.sort(Validation_Ind)
    Test_Ind = np.sort(Test_Ind)
    # Extracting the drug descriptors of each set based on their associated NSCs
    X_Train_Raw = Cell_Features[Cell_Features.NSC.isin(Train_Ind)]
    X_Validation_Raw = Cell_Features[Cell_Features.NSC.isin(Validation_Ind)]
    X_Test_Raw = Cell_Features[Cell_Features.NSC.isin(Test_Ind)]
    
    Y_Train = TargetDF[TargetDF.NSC.isin(Train_Ind)];  Y_Train = np.array(Y_Train['NORMLOG50']) 
    Y_Validation = TargetDF[TargetDF.NSC.isin(Validation_Ind)];  Y_Validation = np.array(Y_Validation['NORMLOG50']) 
    Y_Test = TargetDF[TargetDF.NSC.isin(Test_Ind)];  Y_Test = np.array(Y_Test['NORMLOG50']) 
    
    Threshold = 0.55
    # Convert the drug responses into "Resistive" and "Sensitive" classes given the provided threshold
	Y_Train_Class = Reg_to_Class(Y_Train,Threshold);  
    Y_Validation_Class = Reg_to_Class(Y_Validation,Threshold);
    Y_Test_Class = Reg_to_Class(Y_Test,Threshold);   
    target_names = ['Resistive', 'Sensitive']
   
    X_Dummy = X_Train_Raw.values;     X_Train = X_Dummy[:,2:]
    X_Dummy = X_Validation_Raw.values;     X_Validation = X_Dummy[:,2:]
    X_Dummy = X_Test_Raw.values;      X_Test = X_Dummy[:,2:]

   
    ################
    ##   REFINED  ##
    ################
    from Toolbox import REFINED_Im_Gen
    with open('theMapping_REFINED.pickle','rb') as file:
        gene_names,coords,map_in_int = pickle.load(file)

    X_Train_REFINED = REFINED_Im_Gen(X_Train,nn, map_in_int, gene_names,coords)
    X_Val_REFINED = REFINED_Im_Gen(X_Validation,nn, map_in_int, gene_names,coords)
    X_Test_REFINED = REFINED_Im_Gen(X_Test,nn, map_in_int, gene_names,coords)
   
    #####################
    ###    CNN Model  ###
    #####################
    import keras
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse=False)
    Y_Train_Class = Y_Train_Class.reshape(-1,1)
    Y_Validation_Class = Y_Validation_Class.reshape(-1,1)
    Y_Test_Class = Y_Test_Class.reshape(-1,1)
    Y_Train_Encoded = onehot_encoder.fit_transform(Y_Train_Class)
    Y_Validation_Encoded = onehot_encoder.fit_transform(Y_Validation_Class)
    Y_Test_Encoded = onehot_encoder.fit_transform(Y_Test_Class)
    
	# Train CNN
   
	sz = X_Train_REFINED.shape
	Width = int(math.sqrt(sz[1]))
	Height = int(math.sqrt(sz[1]))
	CNN_Train = X_Train_REFINED.reshape(-1,Width,Height,1)
	CNN_Val = X_Val_REFINED.reshape(-1,Width,Height,1)
	CNN_Test = X_Test_REFINED.reshape(-1,Width,Height,1)

		
	
	def CNN_model(Width,Height):
		nb_filters = 16
		nb_conv = 7
		
		model = models.Sequential()
		# Convolutional layers
		model.add(layers.Conv2D(nb_filters*1, (nb_conv, nb_conv),padding='valid',strides=1,dilation_rate=1,input_shape=(Width, Height,1)))
		model.add(layers.BatchNormalization())
		model.add(layers.Activation('relu'))
		model.add(layers.Dropout(1-0.7))
		model.add(layers.Conv2D(nb_filters*2, (nb_conv, nb_conv),padding='valid',strides=1,dilation_rate=1))
		#model.add(Conv2D(1, (nb_conv, nb_conv),border_mode='valid',strides=(2,2)))
		model.add(layers.BatchNormalization())
		model.add(layers.Activation('relu'))
		
		model.add(layers.Conv2D(nb_filters*4, (nb_conv -4, nb_conv -4),padding='valid',strides=1,dilation_rate=1))
		model.add(layers.BatchNormalization())
		model.add(layers.Activation('relu'))
								
		model.add(layers.Flatten())
		# Dense layers
		model.add(layers.Dense(256))
		model.add(layers.BatchNormalization())
		model.add(layers.Activation('relu'))
		model.add(layers.Dropout(1-0.7))
		
		model.add(layers.Dense(64))
		model.add(layers.BatchNormalization())
		model.add(layers.Activation('relu'))
		model.add(layers.Dropout(1-0.7))
	
		model.add(layers.Dense(2))
		model.add(layers.Activation('softmax'))

		
		initial_learning_rate = 0.0001
		lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
			initial_learning_rate,
			decay_steps=100000,
			decay_rate=0.96,
			staircase=True)

		
		model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
					  loss='binary_crossentropy',
					  metrics=['accuracy'])

		return model
	# Training the CNN Model
	model = CNN_model(Width,Height)
	ES = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=30)
	CNN_History = model.fit(CNN_Train, Y_Train_Encoded, batch_size= 128, epochs = 100, verbose=0, validation_data=(CNN_Val, Y_Validation_Encoded), callbacks = [ES])
	Y_Val_Pred_CNN = model.predict(CNN_Val, batch_size= 128, verbose=0)
	Y_Pred_CNN = model.predict(CNN_Test, batch_size= 128, verbose=0)
	
	Y_Pred_CNN = floattoint(Y_Pred_CNN)

	CNN_ACC = accuracy_score(Y_Test_Encoded, Y_Pred_CNN)
	CNN_Precision , CNN_Recall, CNN_F1_Score, support = precision_recall_fscore_support(Y_Test_Encoded, Y_Pred_CNN, average='weighted')
	CNN_AUC = roc_auc_score(Y_Test_Encoded, Y_Pred_CNN)


	print(CNN_ACC, " CNN ACC of " + SEL_CEL)
	print(CNN_Precision, " CNN Precision of "  + SEL_CEL)
	print(CNN_Recall," CNN Recall of "  + SEL_CEL)
	print(CNN_F1_Score," CNN F1 score of "  + SEL_CEL)
	print(CNN_AUC," CNN AUC of "  + SEL_CEL)
	
	Class_Res[0,:] = np.array([CNN_ACC, CNN_Precision, CNN_Recall, CNN_F1_Score, CNN_AUC])
	tf.keras.backend.clear_session()

    PD_Class_Res = pd.DataFrame(data=Class_Res, columns = ['ACC','Precision','Recall','F1 Score','AUC'], index = ['CNN REFINED'])
    Results_Dic[SEL_CEL] = PD_Class_Res
    
print(Results_Dic)

    
    
