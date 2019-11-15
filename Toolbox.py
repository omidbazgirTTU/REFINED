import numpy as np
import pandas as pd
import os
from scipy.stats import pearsonr
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
