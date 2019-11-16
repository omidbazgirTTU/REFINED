# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 18:04:36 2019

@authors: Ruibzhan & Omid Bazgir
"""

# This code is written based on using Message Passing Interface (MPI) of python to run the hill climbing section of REFINED on HPCC very efficiently. To run tis code make sure to install mpi4py library of python
# Some functions needed to run this code is written in the paraHill.py file do some specific computation
from mpi4py import MPI 
import paraHill
import pickle
import numpy as np
from itertools import product

#%% MPI set up 
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
n_processors = comm.Get_size()
print("Processors found: ",n_processors)

# Distributing the input data among the processors for parallel processing
def scatter_list_to_processors(comm, data_list, n_processors):
    import math
    data_amount = len(data_list)
    heap_size = math.ceil(data_amount/(n_processors-1))

    for pidx in range(1,n_processors):
        try:
            heap = data_list[heap_size*(pidx-1):heap_size*pidx]
        except:
            heap = data_list[heap_size*(pidx-1):]
        comm.send(heap,dest = pidx)

    return True

# Receiving data from each processor and collect them into a vector(tensor)
def receive_from_processors_to_dict(comm, n_processors):
    # receives dicts, combine them and return
    feedback = dict()
    for pidx in range(1,n_processors):
        receved = comm.recv(source=pidx)
        feedback.update(receved)
    return feedback

#%% load data
with open('Init_MDS_Euc.pickle','rb') as file:																# Loading the hill climbing input(initial MDS output)
    gene_names,dist_matr,init_map = pickle.load(file)

Nn = len(gene_names)																						# Number of features

NI = 5 																										# Number of iterations

# Check if the image is not squarred!
if init_map.shape[0] != init_map.shape[1]:
    raise ValueError("For now only square images are considered.")
	
nn = init_map.shape[0]																					    # Squarred output image size 			

# Converting feature numbers from string to integer for example feature 'F34' will be 34, in the MDS initial map 
init_map = np.char.strip(init_map.astype(str),'F').astype(int)
map_in_int = init_map   
#%% Hill climbing
Dist_evol = []																								# Initializing distance evolution vector as an empty list			
if my_rank == 0:
    print("Initial distance: >>>",paraHill.universial_corr(dist_matr,map_in_int))							# Printing out difference between the inital distance matrix and the converted feature map
    for n_iter in range(NI):																				# Begin iterating process NI times
        # 9 initial coordinates. 
        init_coords = [x for x in product([0,1,2],repeat = 2)]												# Use a 3*3 window to exchange feature location in the feature map
        for init_coord in init_coords:
            # Update the mapping. 
            broadcast_msg = map_in_int  																	# Initial map will be broadcasted into all available processors
            comm.bcast(broadcast_msg,root = 0)
            # generate the centroids
            xxx = [init_coord[0]+i*3 for i in range(int(nn/3)+1) if (init_coord[0]+i*3)<nn]						
            yyy = [init_coord[1]+i*3 for i in range(int(nn/3)+1) if (init_coord[1]+i*3)<nn]
            centr_list = [x for x in product(xxx,yyy)]
            # Master send and recv
            scatter_list_to_processors(comm,centr_list,n_processors)										# scatter data
            swap_dict = receive_from_processors_to_dict(comm,n_processors)									# collect data
            print(swap_dict)
            map_in_int = paraHill.execute_dict_swap(swap_dict, map_in_int)									# Perform feature location exchange using *execute_dict_swap function 
            
            print(">",init_coord,"Corr:",paraHill.universial_corr(dist_matr,map_in_int))                    # Report the distance

        print(">>>",n_iter,"Corr:",paraHill.universial_corr(dist_matr,map_in_int))							# Report the overal distance cost after going over a window			
        Dist_evol.append(paraHill.universial_corr(dist_matr,map_in_int))									# Calculate the distance evolution in each iteration and append it to the previous one	
        
    coords = np.array([[item[0] for item in np.where(map_in_int == ii)] for ii in range(Nn)])				# Generate the final REFINED coordinates
    # Save the REFINED coordinates
	with open("theMapping.pickle",'wb') as file:
        pickle.dump([gene_names,coords,map_in_int],file)
    import pandas as pd
    pd.Series(Dist_evol).to_csv("Distance_evolution.csv")													# Save the distance evolution in a csv file    
else:
    # other processors
    for n_iter in range(NI):
        broadcast_msg = init_map    # just for a size

        # 9 initial Centroids
        for ii in range(9):
            #Update the mapping
            map_in_int = comm.bcast(broadcast_msg,root = 0)
            
            centr_list = comm.recv(source = 0)
            each_swap_dict = paraHill.evaluate_centroids_in_list(centr_list,dist_matr,map_in_int)
            comm.send(each_swap_dict,dest = 0)
    #result = dict()
    #for each in data:
    #    result.update({each: -each})
    #comm.send(result,dest = 0)

MPI.Finalize
