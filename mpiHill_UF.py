# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 18:04:36 2019

@author: Ruibzhan
"""


from mpi4py import MPI 
import paraHill
import pickle
import numpy as np
from itertools import product
import time
import datetime
import config
from config import *
#%% Comm set
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
n_processors = comm.Get_size()
print("Processors found: ",n_processors)
current_time = datetime.datetime.now()
print("Time now at the beginning is: ", current_time)

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

def receive_from_processors_to_dict(comm, n_processors):
    # receives dicts, combine them and return
    feedback = dict()
    for pidx in range(1,n_processors):
        receved = comm.recv(source=pidx)
        feedback.update(receved)
    return feedback

#start = time.time()
#%% load data
with open(args.init,'rb') as file:
    gene_names,dist_matr,init_map = pickle.load(file)
# with open('./Synthetic/Init_Synth8000.pickle','rb') as file:
   # gene_names,dist_matr,init_map = pickle.load(file)
Nn = len(gene_names)

kk = args.num

if init_map.shape[0] != init_map.shape[1]:
    raise ValueError("For now only square images are considered.")
nn = init_map.shape[0]
# Convert from 'F34' to int 34
init_map = np.char.strip(init_map.astype(str),'F').astype(int)
map_in_int = init_map   
#%%
corr_evol = []
if my_rank == 0:
    print("Initial corr: >>>",paraHill.universial_corr(dist_matr,map_in_int))
    for n_iter in range(kk):
        # 9 initial coordinates. 
        init_coords = [x for x in product([0,1,2],repeat = 2)]
        for init_coord in init_coords:
            # Update the mapping. 
            broadcast_msg = map_in_int  
            comm.bcast(broadcast_msg,root = 0)
            # generate the centroids
            xxx = [init_coord[0]+i*3 for i in range(int(nn/3)+1) if (init_coord[0]+i*3)<nn]
            yyy = [init_coord[1]+i*3 for i in range(int(nn/3)+1) if (init_coord[1]+i*3)<nn]
            centr_list = [x for x in product(xxx,yyy)]
            # Master send and recv
            scatter_list_to_processors(comm,centr_list,n_processors)
            swap_dict = receive_from_processors_to_dict(comm,n_processors)
            print(swap_dict)
            map_in_int = paraHill.execute_dict_swap(swap_dict, map_in_int)
            
            print(">",init_coord,"Corr:",paraHill.universial_corr(dist_matr,map_in_int))

        print(">>>",n_iter,"Corr:",paraHill.universial_corr(dist_matr,map_in_int))
        corr_evol.append(paraHill.universial_corr(dist_matr,map_in_int))
        
    coords = np.array([[item[0] for item in np.where(map_in_int == ii)] for ii in range(Nn)])
    with open(args.mapping,'wb') as file:
        pickle.dump([gene_names,coords,map_in_int],file)
    #print("Consumed time:",start - time.time())
    Endtime = datetime.datetime.now()
    print("Time now at the end: ", Endtime)
    import pandas as pd
    pd.Series(corr_evol).to_csv(args.evolution)    
else:
    # other processors
    for n_iter in range(kk):
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
