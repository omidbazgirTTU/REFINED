# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 18:41:14 2019

@author: obazgir
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from fancyimpute import KNN
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression




#%% NRMSE
def NRMSE(Y_Target, Y_Predict):
    Y_Target = np.array(Y_Target); Y_Predict = np.array(Y_Predict);
    Y_Target = Y_Target.reshape(len(Y_Target),1);    Y_Predict = Y_Predict.reshape(len(Y_Predict),1);    
    Y_Bar = np.mean(Y_Target)
    Nom = np.sum((Y_Predict - Y_Target)**2);    Denom = np.sum((Y_Bar - Y_Target)**2)
    MSE = np.mean((Y_Predict - Y_Target)**2);   NRMSE = np.sqrt(Nom/Denom)
    R2 = 1 - NRMSE**2
    return NRMSE, R2

def NMAE(Y_Target, Y_Predict):
    Y_Target = np.array(Y_Target); Y_Predict = np.array(Y_Predict);
    Y_Target = Y_Target.reshape(len(Y_Target),1);    Y_Predict = Y_Predict.reshape(len(Y_Predict),1);    
    Y_Bar = np.mean(Y_Target)
    Nom = np.abs((Y_Predict - Y_Target));    Denom = np.abs((Y_Bar - Y_Target))
    NormMAE = np.mean(Nom)/np.mean(Denom)
    return NormMAE
    

#%% Random position generation
import math
def Random_position(p):
    NN = int(math.sqrt(p)) +1
    Feat_num = np.arange(p)
    np.random.shuffle(Feat_num)
    #Feature_List = [f'F{i}' for i in Feat_num]
    Pos_mat = []
    for i in range(p):
        Pos_mat.append([int(Feat_num[i]/NN),int(Feat_num[i]%NN)])
    return(Pos_mat)
    
def Random_Image_Gen(X, Rand_Pos_mat):
    sz = X.shape
    p = sz[1];  N = sz[0]
    NN = int(math.sqrt(p)) +1
    Im = np.zeros((NN,NN))
    X_Gen = np.zeros((N,NN**2))
    for j in range(N):
        for i in range(p):
            P = Rand_Pos_mat[i]
            Im[P[0],P[1]] = X[j,i]
        Image_Store = Im.reshape((NN**2,1)).T
        X_Gen[j,:] = Image_Store
    return X_Gen

#%% MDS by Ruibo
from sklearn.manifold import MDS

def two_d_norm(xy):
    # xy is N x 2 xy cordinates, returns normed-xy on [0,1]
    norm_xy = (xy - xy.min(axis = 0)) / (xy - xy.min(axis = 0)).max(axis = 0)
    return norm_xy

def two_d_eq(xy,Nn):
    # xy is N x 2 xy cordinates, returns eq-xy on [0,1]
    xx_rank = np.argsort(xy[:,0])
    yy_rank = np.argsort(xy[:,1])
    eq_xy = np.full(xy.shape,np.nan)
    for ii in range(xy.shape[0]):
        xx_idx = xx_rank[ii]
        yy_idx = yy_rank[ii]
        eq_xy[xx_idx,0] = ii * 1/Nn
        eq_xy[yy_idx,1] = ii * 1/Nn
    return eq_xy

#embedding = MDS(n_components=2)
#mds_xy = embedding.fit_transform(transposed_input)
# to pixels
def Assign_features_to_pixels(xy,nn,verbose = False):
    # For each unassigned feature, find its nearest pixel, repeat until every ft is assigned
    # xy is the 2-d coordinates (normalized to [0,1]); nn is the image width. Img size = n x n
    # generate the result summary table, xy pixels; 3rd is nan for filling the idx
    Nn = xy.shape[0]
    
    from itertools import product
    pixel_iter = product([x for x in range(nn)],repeat = 2)
    result_table = np.full((nn*nn,3),np.nan)
    ii = 0
    for each_pixel in pixel_iter:
        result_table[ii,:2] = np.array(each_pixel)
        ii+=1
    # Use numpy array for speed 
        
    from sklearn.metrics import pairwise_distances
    
#    xy = eq_xy
    centroids = result_table[:,:2] / nn + 0.5/nn
    pixel_avail = np.ones(nn*nn).astype(bool)
    feature_assigned = np.zeros(Nn).astype(bool)
    
    dist_xy_centroids = pairwise_distances(centroids,xy,metric='euclidean')
    
    while feature_assigned.sum()<Nn:
        # Init the pick-relationship table
        pick_xy_centroids = np.zeros(dist_xy_centroids.shape).astype(bool)
        
        for each_feature in range(Nn):
            # for each feature, find the nearest available pixel
            if feature_assigned[each_feature] == True:
                # if this feature is already assigned, skip to the next ft
                continue
            else:
                # if not assigned:
                for ii in range(nn*nn):
                    # find the nearest avail pixel
                    nearest_pixel_idx = np.argsort(dist_xy_centroids[:,each_feature])[ii]
                    if pixel_avail[nearest_pixel_idx] == True:
                        break
                    else:
                        continue
                pick_xy_centroids[nearest_pixel_idx,each_feature] = True
            
        for each_pixel in range(nn*nn):
            # Assign the feature No to pixels
            if pixel_avail[each_pixel] == False:
                continue
            else:
                # find all the "True" features. np.where returns a tuple size 1
                related_features = np.where(pick_xy_centroids[each_pixel,:] == 1)[0]
                if len(related_features) == 1:
                    # Assign it
                    result_table[each_pixel,2] = related_features[0]
                    pixel_avail[each_pixel] = False
                    feature_assigned[related_features[0]] = True
                elif len(related_features) > 1:
                    related_dists = dist_xy_centroids[each_pixel,related_features]
                    best_feature = related_features[np.argsort(related_dists)[0]] # Sort, and pick the nearest one among them
                    result_table[each_pixel,2] = best_feature
                    pixel_avail[each_pixel] = False
                    feature_assigned[best_feature] = True
        if verbose:
            print(">> Assign features to pixels:", feature_assigned.sum(),"/",Nn)
    result_table = result_table.astype(int)
    
    img = np.full((nn,nn),'NaN').astype(object)
    for each_pixel in range(nn*nn):
        xx = result_table[each_pixel,0]
        yy = result_table[each_pixel,1]
        ft = 'F' + str(result_table[each_pixel,2])
        img[xx,yy] = ft
    return img.astype(object)
print(">>>> MDS")
#eq_xy = two_d_eq(mds_xy)
#Img = Assign_features_to_pixels(eq_xy,nn,verbose=1)
#Init_Corr_MDS = InitCorr(dist_mat,Img,nn)

def MDS_Im_Gen(X,nn, Img):
    [N_sam,P_Feat] = X.shape
    X_Gen = np.zeros((N_sam,nn**2))
    conv_Img = Img.reshape(Img.size,1)
    for i in range(nn**2):
        Feature = np.array(conv_Img[i]);    Feature = Feature[0]; F_Num = int(Feature[1:])
        if abs(F_Num) < nn**2:
            X_Gen[:,i] = X[:,F_Num]
        else:
            X_Gen[:,i] = 0 
    return X_Gen

#%% CCLE functions
def dataframer(Main,Set_in, name_in, name_out):
    A = Set_in[name_in].tolist()
    Set_out = Main[Main[name_out] == A[0]]
    for cell in range(len(A) - 1):
        df = Main[Main[name_out] == A[cell + 1]]
        Set_out = pd.concat([Set_out, df])
    return Set_out

def Reg_to_Class(Y,Threshold):
	Y_Class = np.zeros(len(Y))
	Y_Sens = np.where(Y > Threshold)
	Y_Class[Y_Sens] = 1
	Y_Class =  Y_Class.astype(int)
	Y_Class = Y_Class.tolist()
	Y_Class = np.array(Y_Class)
	return Y_Class
	
def floattoint(Y_Test_Encoded):
	Y_Class = np.zeros(Y_Test_Encoded.shape)
	Y_Sens = np.where(Y_Test_Encoded > 0.5)
	Y_Class[Y_Sens] = 1
	Y_Class =  Y_Class.astype(int)
	Y_Class = Y_Class.tolist()
	Y_Class = np.array(Y_Class)
	return Y_Class
def REFINED_Im_Gen(X,nn, map_in_int, gene_names,coords):
	[N_sam,P_Feat] = X.shape
	X_Gen = np.zeros((N_sam,nn**2))
	for i in range(N_sam):
		data = X[i,:] 
		X_REFINED = pd.DataFrame(data = data.reshape(1,len(data)), columns = gene_names)
		Image = np.zeros(map_in_int.shape)
		for j in range(len(coords)):
			val = np.array(X_REFINED[gene_names[j]])
			Image[coords[j,0],coords[j,1]] = val
		Image = Image.reshape(nn**2)
		X_Gen[i,:] = Image
	return X_Gen
#%% GDSC
def GDSC_dataframer(PD_Set, Set_Name,PD_Attribute,Attribute_Name):
    A = PD_Set[Set_Name].tolist()
    b = PD_Attribute[PD_Attribute[Attribute_Name] == A[0]].reset_index().drop(columns = ['index'])
    Data_arry = np.array(b.values[0,1:],dtype = float)
    Data_arry = Data_arry.reshape(1,len(Data_arry))
    for i in range(len(A) - 1 ):
        b = PD_Attribute[PD_Attribute[Attribute_Name] == A[i + 1]].reset_index().drop(columns = ['index'])
        Arr = np.array(b.values[0,1:],dtype = float)
        Arr = Arr.reshape(1,len(Arr))
        Data_arry = np.append(Data_arry,Arr, axis = 0)
    
    PD_Data_arry = pd.DataFrame(data = Data_arry, columns = PD_Attribute.columns.tolist()[1:], index = A)
    return Data_arry, PD_Data_arry
	
def GDSC_NPier(PD_Set, Set_Name,PD_Attribute,Attribute_Name):
    PD_Set = PD_Set.reset_index()
    PD_Set.shape[0]
    PD_Attribute.shape[1] - 1
    X_NP = np.zeros((PD_Set.shape[0],PD_Attribute.shape[1] - 1))
    Source = list(set(PD_Set[Set_Name].tolist()))
    for name in Source:
        idx = PD_Set.index[PD_Set[Set_Name] == name].tolist()
        XX = np.array(PD_Attribute[PD_Attribute[Attribute_Name] == name].values[0,1:], dtype = float)
        X_NP[idx,:] = XX
    return X_NP
	
	
def Coord_Converter(coords_drug2,nn):
    coords_drug3 = np.full((nn,nn),'NaN').astype(object)
    for i in range(nn):
        for j in range(nn):
            ft = 'F' + str(coords_drug2[i,j])
            coords_drug3[i,j] = ft
    return coords_drug3
	
def Bias_Calc(Y_Test, Y_Pred):
    Error = Y_Test - Y_Pred
    Y_Test = Y_Test.reshape(len(Y_Test),1)
    Error = Error.reshape(len(Error),1)
    
    reg = LinearRegression().fit(Y_Test, Error)
    Bias = reg.coef_[0]
    
    return Bias
	
