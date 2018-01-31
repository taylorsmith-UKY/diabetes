#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 12:22:57 2018

@author: taylorsmith
"""

lim=15

import h5py
import diabetes_functions as df
#import matplotlib.pyplot as plt
import numpy as np
#import pickle


import_features=['Weight_Index', 'Waist(CM)', 'Hip(CM)', 'Waist_Hip_Ratio',\
                 'systolic_pressure', 'diastolic_pressure', 'Hb', 'Cr', 'Ch',\
                 'TG', 'HDL', 'LDL', 'FBG', 'PBG', 'INS0', 'CP0', 'Ch', 'TG',\
                 'HDL', 'LDL', 'FBG', 'PBG', 'HbA1c', 'INS0','HOMAIR', 'HOMAB',\
                 'CP0', 'CRP',  'FFA', 'visceral_fat', 'subcutaneous_fat','FT3',\
                 'FT4', 'TSH']

all_features=['Weight(KG)','Weight_Index','Waist(CM)','Hip(CM)',\
              'Waist_Hip_Ratio','Heart_rate','systolic_pressure',\
              'diastolic_pressure','WBC','Hb','ALT','AST','rGT','ALP',\
              'prealbumin','bile_acid','total_bilirubin','direct_bilirubin',\
              'BUN','Cr','uric_acid','RBP','CysC','K','Na','Mg','Ca','P','Ch',\
              'TG','HDL','LDL','FBG','PBG','HbA1c','GA','INS0','INS30','INS120',\
              'HOMAIR','HOMAB','CP0','CP30','CP120','HOMAcp','ALB1','ALB2',\
              'ALB3','Average_uric_ALB','GFR','ACR','CRP','folic_acid','VitB12',\
              'PTH','OH25D','Serum_Fe','serum_Fe_protein','CA199','FFA',\
              'visceral_fat','subcutaneous_fat','FT3','FT4','TSH','Reversed_T3',\
              'BG30','AAINS0','AAINS2','AAINS4','AAINS6','AAINS_index','AACP0',\
              'AACP2','AACP4','AACP6','AACP_index','urinary_uric_acid','Urine_creatinine']
f=h5py.File('all_data.h5','r')
ds=f['interpo']
all_ids = f['ids']

npts = lim
nfts = lim

all_features = all_features

#ce = df.cost_entry(0,[],[])
#cost = [[df.cost_entry(0,[],[]) for x in range(nfts)] for y in range(npts)]
#cost = np.array(cost)

#cost = df.build_cm_iter(ds,cost,all_ids,all_features,import_features)

cost = df.build_cm(ds,import_features,all_features,all_ids)
np.savetxt('cost_matrix.txt',cost,fmt='%.3f')
#with open('cost_matrix_'+str(lim)+'x'+str(lim)+'.pkl','w') as pf:
#    pickle.dump(cost, pf)
#pf.close()

#cost_val = [[cost[y][x].cost for x in range(nfts)] for y in range(npts)]

#cost_val = np.array(cost_val,dtype=float)
#np.savetxt('cost_matrix_values_'+str(lim)+'x'+str(lim)+'.txt',cost_val)
