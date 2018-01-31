from __future__ import division
import h5py
import numpy as np

'''
PARAMETERS
'''
#savefig()
outFile='all_data.hdf5'

def main():
	f=h5py.File(outFile,'r')
	ds = f['data'][:,0:6,:]
	data = f['interpo']
	import_features=['Weight_Index', 'Waist(CM)', 'Hip(CM)', 'Waist_Hip_Ratio','systolic_pressure', 'diastolic_pressure', 'Hb', 'Cr', 'Ch', 'TG', 'HDL', 'LDL', 'FBG', 'PBG', 'INS0', 'CP0', 'Ch', 'TG', 'HDL', 'LDL', 'FBG', 'PBG', 'HbA1c', 'INS0','HOMAIR', 'HOMAB', 'CP0', 'CRP',  'FFA', 'visceral_fat', 'subcutaneous_fat','FT3', 'FT4', 'TSH']
	all_features=['Weight(KG)','Weight_Index','Waist(CM)','Hip(CM)','Waist_Hip_Ratio','Heart_rate','systolic_pressure','diastolic_pressure','WBC','Hb','ALT','AST','rGT','ALP','prealbumin','bile_acid','total_bilirubin','direct_bilirubin','BUN','Cr','uric_acid','RBP','CysC','K','Na','Mg','Ca','P','Ch','TG','HDL','LDL','FBG','PBG','HbA1c','GA','INS0','INS30','INS120','HOMAIR','HOMAB','CP0','CP30','CP120','HOMAcp','ALB1','ALB2','ALB3','Average_uric_ALB','GFR','ACR','CRP','folic_acid','VitB12','PTH','OH25D','Serum_Fe','serum_Fe_protein','CA199','FFA','visceral_fat','subcutaneous_fat','FT3','FT4','TSH','Reversed_T3','BG30','AAINS0','AAINS2','AAINS4','AAINS6','AAINS_index','AACP0','AACP2','AACP4','AACP6','AACP_index','urinary_uric_acid','Urine_creatinine']
	all_ids = f['ids'][0:-2]
	build_cm(data,import_features,all_features,all_ids)
	return

def is_important(feature,important):
	if feature in important:
		return True
	else:
		return False

def build_cm(data,import_features,all_features,all_ids):
	pt_idx = data.shape[0] - 1
	ft_idx = data.shape[2] - 1
	pt_list = all_ids
	ft_list = all_features
	temp_mat = data[:,:,:]
	path_f = open('path.txt','w')
	order_f = open('order.txt','w')
	rem_f = open('removal_order.txt','w')
	cm = np.zeros([data.shape[0],data.shape[2]])
	f_cm = open('cm.txt','w')
	while pt_idx != 0 and ft_idx != 0:
		path_f.write('(%d,%d)\n' % (pt_idx,ft_idx))
		p_order, f_order, p_max, f_max = sort_by_nan(temp_mat,pt_list,ft_list,import_features)
		temp_mat = temp_mat[p_order,:,:]
		temp_mat = temp_mat[:,:,f_order]
		pt_list = [pt_list[p_order[x]] for x in range(len(p_order))]
		ft_list = [ft_list[f_order[x]] for x in range(len(f_order))]
		for i in range(pt_idx,-1,-1):
			cm[i,ft_idx] = np.count_nonzero(np.isnan(temp_mat[0:i+1,:,0:ft_idx+1]))#/(data.shape[-1]*data.shape[1])
		for i in range(ft_idx,-1,-1):
			cm[pt_idx,i] = np.count_nonzero(np.isnan(temp_mat[0:pt_idx+1,:,0:i+1]))#/(data.shape[0]*data.shape[1])

		order_f.write('%s' % (pt_list[p_order[0]]))
		for i in range(1,len(p_order)):
			order_f.write(', %s' % (pt_list[p_order[i]]))
		order_f.write('\n')
		order_f.write('%s' % (ft_list[f_order[0]]))
		for i in range(1,len(f_order)):
			order_f.write(', %s' % (ft_list[f_order[i]]))
		order_f.write('\n')
		order_f.write('\n')
		if p_max > f_max or ft_idx == 0:
			rem_f.write('%s\n' % (pt_list[-1]))
			temp_mat = temp_mat[0:pt_idx,:,:]
			pt_idx -= 1
			pt_list = pt_list[0:-1]
		else:
			rem_f.write('%s\n' % (ft_list[-1]))
			temp_mat = temp_mat[:,:,0:ft_idx]
			ft_idx -= 1
			ft_list = ft_list[0:-1]

	for i in range(cm.shape[0]):
		f_cm.write('%f' % (cm[i,0]))
		for j in range(1,cm.shape[1]):
			f_cm.write(', %f' % (cm[i,j]))
		f_cm.write('\n')
	f_cm.close()

def sort_by_nan(data,patients,features,important):
	pt_pcts = np.zeros(len(patients))
	ft_pcts = np.zeros(len(features))

	n_pts = data.shape[0]
	n_feats = data.shape[2]
	n_tpts = data.shape[1]

	#percent (# empty) / (total #) for each patient
	for i in range(n_pts):#patient id
		pt_pcts[i] = float(np.count_nonzero(np.isnan(data[i,:,:])))/(n_feats*n_tpts)
	#percent (# empty) / (total #) for each feature
	for i in range(n_feats):
		ft_pcts[i] = float(np.count_nonzero(np.isnan(data[:,:,i])))/(n_pts*n_tpts)
	p_order = np.argsort(pt_pcts)
	f_order = np.argsort(ft_pcts)
	p_max = np.nanmax(pt_pcts)
	f_max = np.nanmax(ft_pcts)
	# count = 0
	# for i in range(len(f_order)):
	# 	if is_important(features[f_order[i]],important):
	# 		continue
	# 	else:
	# 		if count != i and count < len(important):
	# 			j = i
	# 			while j < n_feats and is_important(features[f_order[j]],important):
	# 				j += 1
	# 			if j == len(f_order):
	# 					break
	# 			temp = f_order[j]
	# 			for k in range(j,i,-1):
	# 				f_order[k] = f_order[k-1]
	# 			f_order[i] = temp
	# 	count += 1
	return p_order, f_order, p_max, f_max


main()
