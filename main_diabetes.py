import diabetes_functions as df
import numpy as np
#import matplotlib.pyplot as plt
#------------------------------- PARAMETERS -----------------------------------#
inFile='data_cleaned.xlsx'
outFile='all_data.h5'

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

#------------------------------------------------------------------------------#



ds=df.get_data(inFile=inFile,outFile=outFile)
f=ds.parent

f.copy(ds,'/interpo')
interpo=f['interpo']
#Interpolate
time_location=[0,1,2,4,8,12,16,20]
for idx in range(ds.shape[0]):    #patient
    for f_idx in range(ds.shape[2]):    #feature
        interpo[idx,:,f_idx]=df.linear_interpolation(time_location,interpo[idx,:,f_idx])
#np.savetxt('interpolate.txt',interpo,delimiter=" ",fmt="%s")

ds=f['interpo'][:-2,:,:]
all_ids = f['ids'][0:-2]

cost = df.build_cm(ds,import_features,all_features,all_ids)
np.savetxt('cost_matrix.txt',cost,fmt='%.3f')

'''
ce = df.cost_entry(0,[],[])
cost = [[df.cost_entry(0,[],[]) for x in range(nfts)] for y in range(npts)]
#cost = np.array(cost)

cost = df.build_cm_iter(ds,cost,all_ids,all_features,import_features)

cost_val = [[cost[y,x].cost for x in range(nfts)] for y in range(npts)]

plt.figure()
plt.pcolormesh(cost)
plt.savefig('iter_cost_test.png')

all_ids = f['ids'][0:-2]
interpo=f['interpo']
df.build_cm(interpo,import_features,all_features,all_ids)
'''
f.close()


