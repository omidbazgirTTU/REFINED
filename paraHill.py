# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:30:06 2019

@author: Ruibzhan & Omid Bazgir
"""

from scipy.stats import pearsonr
import numpy as np
import random
from scipy.spatial import distance
import pickle
import pandas as pd
import time
from itertools import product

#%%
def universial_corr(dist_matr, mapping_in_int):
    # dist_matr is a sqr matr
    Nn = dist_matr.shape[0]
    # find what is the int coordinates for each feature, get an array
    # Because np.where returns a tuple (x_position array, y_position array), a generation is used
    coord = np.array([[item[0] for item in np.where(mapping_in_int == ii)] for ii in range(Nn)])
    # get a 1-d form distance the euclidean dist between pixles positions
    pixel_dist = distance.pdist(coord)
    pixel_dist = pixel_dist.reshape(len(pixel_dist),1)
    # convert the 2-d distance to 1d distance
    feature_dist = distance.squareform(dist_matr)
    feature_dist = feature_dist.reshape(len(feature_dist),1)
    ## pearsonr returns a tuple
    #corrs = pearsonr(feature_dist,pixel_dist)[0]
    L2_Norm = np.sqrt(sum((pixel_dist - feature_dist)**2)/sum(feature_dist**2))
    return L2_Norm
#%%
def evaluate_swap(coord1,coord2,dist_matr,mapping_in_int,original_corr = -2):
    # Coord are in list[]
    # Avoid messing up with the origianl map
    # The correlation before swap can be passed to save some calculation
    the_map = mapping_in_int.copy()
    # If out of bound, return NaN. 
    if coord1[0]<0 or coord1[1]<0 or coord2[0]<0 or coord2[1]<0:
        return np.nan
    if coord1[0]>=the_map.shape[0] or coord1[1]>=the_map.shape[0] or coord2[0]>=the_map.shape[0] or coord2[1]>=the_map.shape[0]:
        return np.nan
    # If not given, recompute.
    if original_corr<-1 or original_corr>1:
        original_corr = universial_corr(dist_matr,the_map)
    # Swap
    try:
        temp = the_map[coord1[0],coord1[1]] 
        the_map[coord1[0],coord1[1]] = the_map[coord2[0],coord2[1]]
        the_map[coord2[0],coord2[1]] = temp
        changed_corr = universial_corr(dist_matr,the_map)
        return(changed_corr - original_corr)
    except IndexError:
        raise Warning ("Swap index:", coord1,coord2,"Index error. Check the coordnation.")
        return np.nan
    
def evaluate_centroid(centroid,dist_matr,mapping_in_int):
    original_corr = universial_corr(dist_matr,mapping_in_int)
    results = [100000] # just to skip the 0 position
    for each_direc in product([-1,0,1],repeat = 2):
        #print(each_direc)
        # directions are returned as tuple (-1,1), (-1,0), (-1,1), (0,0), ....
        swap_coord = [centroid[0]+each_direc[0],centroid[1]+each_direc[1]]
        evaluation = evaluate_swap(centroid,swap_coord,dist_matr,mapping_in_int,original_corr)
        results.append(evaluation)
    results_array = np.array(results)
    #best_swap_direc = np.where(results_array == np.nanmax(results_array))[0][0]
    best_swap_direc = np.where(results_array == np.nanmin(results_array))[0][0]
    # Give the best direction as a int
    return best_swap_direc

def evaluate_centroids_in_list(centroids_list,dist_matr,mapping_in_int):
    # and returns a dict
    results = dict()
    for each_centr in centroids_list:
        each_centr = tuple(each_centr)
        evaluation = evaluate_centroid(each_centr,dist_matr,mapping_in_int)
        results.update({each_centr:evaluation})
    return results

#%%
def execute_coordination_swap(coord1,coord2,mapping_in_int):
    # try passing the ref. directly 
    the_map = mapping_in_int#.copy()
    # If out of bound, return NaN. 
    if coord1[0]<0 or coord1[1]<0 or coord2[0]<0 or coord2[1]<0:
        raise Warning("Swapping failed:",coord1,coord2,"-- Negative coordnation.")
        return the_map
    if coord1[0]>the_map.shape[0] or coord1[1]>the_map.shape[0] or coord2[0]>the_map.shape[0] or coord2[1]>the_map.shape[0]:
        raise Warning("Swapping failed:",coord1,coord2,"-- Coordnation out of bound.")
        return the_map

    temp = the_map[coord1[0],coord1[1]] 
    the_map[coord1[0],coord1[1]] = the_map[coord2[0],coord2[1]]
    the_map[coord2[0],coord2[1]] = temp

    return(the_map)

# Initial centriod id & Swapping direction: 
# 1 2 3
# 4 5 6
# 7 8 9
# 0 in swapping is preserved for the header.

def execute_direction_swap(centroid,mapping_in_int,direction = 5):
    # Need to notice that [0] is the vertival coord, [1] is the horiz coord. similar to the matlab images.
    coord1 = list(centroid)
    coord2 = list(centroid)
    if direction not in range(1,10):
        raise ValueError("Invalid swapping direction.")
    if direction == 5:
        return mapping_in_int

    if direction in [1,4,7]:
        coord2[1] -=1
    elif direction in [3,6,9]:
        coord2[1] +=1

    if direction in [1,2,3]:
        coord2[0] -=1
    elif direction in [7,8,9]:
        coord2[0] +=1

    the_map = execute_coordination_swap(coord1,coord2,mapping_in_int)
    return the_map

def execute_dict_swap(swapping_dict, mapping_in_int):
    for each_key in swapping_dict:
        execute_direction_swap(each_key,mapping_in_int,direction = swapping_dict[each_key])
    return mapping_in_int   
