import h5py
import pandas as pd
import numpy as np
from diabetes_functions import *
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
'''
PARAMETERS
'''
#savefig()
inFile='data_cleaned.xlsx'
outFile='all_data.hdf5'

def main():
	#try:
	#	f=h5py.File(outFile,'r+')
	#except:
	f=get_data(inFile=inFile,outFile=outFile)
	#get the interpo result after the get_data function
	#then delete the nan value based on the interpo result
	
	#delete_nan1=delete_nan(f,import_features,all_features)
	
	#gen_test_set(f)
	#get_normalize(f)
	#plot_hist(f)
	stats = get_stats(f)
	f.close()



def get_data(inFile,outFile):
	f=h5py.File(outFile,'w')
	## Extract Meta-data
	meta_grp=f.create_group('metadata')
	metadata=pd.read_excel(inFile,'Metadata')
	ids = f.create_dataset('ids',data=np.array(to_str_list(metadata['Patient_ID'])))
	meta_grp.create_dataset('sex',data=np.array(to_str_list(metadata['Sex'])))
	meta_grp.create_dataset('age',data=np.array(to_int_list(metadata['Age'])))
	meta_grp.create_dataset('bday',data=np.array(to_str_list(metadata['Birthday'])))
	meta_grp.create_dataset('diag_date',data=np.array(to_str_list(metadata['First_Dignosis_Date'])))
	meta_grp.create_dataset('diab_yr',data=np.array(to_str_list(metadata['Diabetes(Year)'])))
	hptn=metadata['High_Blood_pressure'].as_matrix()
	hptn[hptn=='yes']=True
	hptn[hptn=='no']=False
	meta_grp.create_dataset('hptn',data=np.array(hptn,dtype=bool))
	meta_grp.create_dataset('fam_hist',data=np.array(to_str_list(metadata['Family_History'])))
	meta_grp.create_dataset('height',data=np.array(to_flt_list(metadata['Height(CM)'])))
	meta_grp.create_dataset('op_date',data=np.array(to_str_list(metadata['Operation_Date'])))
	meta_grp.create_dataset('cholec',data=np.array(to_str_list(metadata['cholecystopathy_history'])))

	tpts = ['preop','mo3','mo6','mo12','mo24','mo36','mo48','mo60']


#	tpt_dic={'preop':	'before operation',
#		  'mo3':	'3mon aft operation',
#		  'mo6':	'6mon aft operation',
#		  'mo12':	'12mon aft operation',
#		  'mo24':	'24mon aft operation',
#		  'mo36':	'36mon aft operation',
#		  'mo48':	'48mon aft operation',
#		  'mo60':	'60mon after operation'}


	tpt_dic={'preop':	'before operation',
		  'mo3':	'3mon aft operation',
		  'mo6':	'6mon aft operation',
		  'mo12':	'12mon aft operation',
		  'mo24':	'24mon aft operation',
		  'mo36':	'36mon aft operation',
		  'mo48':	'48mon aft operation',
		  'mo60':	'60mon after operation'}
	count = 0
	for timept in tpts:
		this_sheet = pd.read_excel(inFile,tpt_dic[timept])
		keys = [str(this_sheet.keys()[x]) for x in range(len(this_sheet.keys()))]
		date_list=[str(this_sheet['Date'][x]) for x in range(len(this_sheet))]
		for i in range(len(date_list)):
			if len(date_list[i])<8:
				date_list[i] = ''
		try:
			ds = f.create_dataset('data',data=np.zeros([len(this_sheet),len(tpts),len(keys)-2]))
			dates = f.create_dataset('dates',shape=(len(this_sheet),len(tpts)),dtype='|S19')
			ds.attrs.create('column_names',np.array(keys[2:]))
			ds[:,count,:]=np.array(this_sheet[keys[2:]],dtype=float)
			dates[:,count]=date_list
		except:
			ds = f['data']
			for i in range(len(this_sheet)):
				this_id=str(this_sheet[keys[0]][i])
				targ_list=[ids[x].decode('UTF-8') for x in range(len(ids))]
				targ_idx=targ_list.index(this_id)
				ds[targ_idx,count,:]=np.array(this_sheet[keys[2:]].as_matrix()[i,:])
				dates[targ_idx,count]=np.array(date_list[i],dtype='|S19')
		count+=1
		
	f.copy(ds,'/interpo')
	interpo=f['interpo']
	#Interpolate
	time_location=[0,1,2,4,8,12,16,20]
	for idx in range(ds.shape[0]):	#patient
		for f_idx in range(ds.shape[2]):	#feature
			#print("start---"+str(interpo[idx,:,f_idx]))
			interpo[idx,:,f_idx]=linear_interpolation(time_location,interpo[idx,:,f_idx])
			#print("result---"+str(interpo[idx,:,f_idx]))
			#linear_interpolation(time_location,ds[idx,:,f_idx])
			#for t_idx in range(ds.shape[1]):	#timepoint
				#if np.isnan(ds[idx,t_idx,f_idx]):
					#if t_idx > 0:
						#start = t_idx - 1
						#for j in range(t_idx+1,ds.shape[1]):
							#if not np.isnan(ds[idx,j,f_idx]):
								#stop=j
								#npts=stop-start-1
								#start_value=ds[idx,start,f_idx]
								#stop_value=ds[idx,stop,f_idx]
								#step=(stop_value-start_value)/(npts+1)
								#for i in range(npts):
									#interpo[idx,start+i+1,f_idx]=interpo[idx,start+i,f_idx]+step
								#break
					#else:
						#break
	#for idx in range(ds.shape[0]):
		#for f_idx in range(ds.shape[2]):
			#interpo[idx,:,f_idx] = normalize_base(interpo[idx,:,f_idx])
	#for f_idx in range(ds.shape[2]):
		#feature_array=interpo[:,:,f_idx].ravel()
		#plt.figure(f_idx)
		#plt.hist(feature_array,bins=30)
		#plt.show()
	np.savetxt('interpolate.txt',interpo,delimiter=" ",fmt="%s")
	import_features=['Weight_Index', 'Waist(CM)', 'Hip(CM)', 'Waist_Hip_Ratio','systolic_pressure', 'diastolic_pressure', 'Hb', 'Cr', 'Ch', 'TG', 'HDL', 'LDL', 'FBG', 'PBG', 'INS0', 'CP0', 'Ch', 'TG', 'HDL', 'LDL', 'FBG', 'PBG', 'HbA1c', 'INS0','HOMAIR', 'HOMAB', 'CP0', 'CRP',  'FFA', 'visceral_fat', 'subcutaneous_fat','FT3', 'FT4', 'TSH']
	all_features=ds.attrs.get('column_names')
	all_id = ids[:]
	build_cm(interpo,import_features,all_features,all_ids)
	# after_delete,rest_features,remid,remft,rest_id =delete_nan(interpo,import_features,all_features,all_id)
	# normalization=get_normalize(after_delete)
	# plot_hist(normalization,rest_features)
	# d={}
	# d['all_data']=ds[:,:,:]
	# d['interpo'] = interpo
	# d['data']=after_delete
	# d['normalize']=normalization
	# d['removed_patientid']=remid
	# d['removed_feature']=remft
	# d['rest_features']=rest_features
	# d['rest_id']=rest_id
	# d['all_ids']=ids[:]
	# sio.savemat('all_data',d)
	return f

def get_stats(f):
	data=f['data']
	#calculate number of records/measurements per patient at each time point
	rec_per_tpt = np.zeros(data.shape[0:2])
	for i in range(data.shape[1]):
		count=0
		cum=0
		for j in range(data.shape[0]):
			rec_per_tpt[j,i]=sum([1 for x in range(data.shape[2]) if str(data[j,i,x]) != 'nan' and data[j,i,x] != 0])
	np.savetxt('all_stats.txt',rec_per_tpt,fmt='%d')

main()
