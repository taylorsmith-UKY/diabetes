from __future__ import division
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

class cost_entry():
    def __init__(self,cost,pt_list,ft_list):
        self.cost = cost
        self.pt_list = pt_list
        self.ft_list = ft_list

def get_data(inFile,outFile):
    f=h5py.File(outFile,'w')
    ## Extract Meta-data
    meta_grp=f.create_group('metadata')
    metadata=pd.read_excel(inFile,'Metadata')
    ids = f.create_dataset('ids',data=np.array(metadata['Patient_ID'],dtype='|S5'))
    meta_grp.create_dataset('sex',data=np.array(metadata['Sex'],dtype='|S5'))
    meta_grp.create_dataset('age',data=np.array(metadata['Age'],dtype=int))
    meta_grp.create_dataset('bday',data=np.array(metadata['Birthday'],dtype='|S32'))
    meta_grp.create_dataset('diag_date',data=np.array(metadata['First_Dignosis_Date'],dtype='|S7'))
    meta_grp.create_dataset('diab_yr',data=np.array(metadata['Diabetes(Year)'],dtype=float))
    hptn=metadata['High_Blood_pressure'].as_matrix()
    hptn[hptn=='yes']=True
    hptn[hptn=='no']=False
    meta_grp.create_dataset('hptn',data=np.array(hptn,dtype=bool))
    meta_grp.create_dataset('fam_hist',data=np.array(metadata['Family_History'],dtype='|S24'))
    meta_grp.create_dataset('height',data=np.array(metadata['Height(CM)'],dtype=float))
    meta_grp.create_dataset('op_date',data=np.array(metadata['Operation_Date'],dtype='|S32'))
    meta_grp.create_dataset('cholec',data=np.array(metadata['cholecystopathy_history'],dtype='|S24'))

    tpts = ['preop','mo3','mo6','mo12','mo24','mo36','mo48','mo60']

    tpt_dic={'preop':    'before operation',
          'mo3':    '3mon aft operation',
          'mo6':    '6mon aft operation',
          'mo12':    '12mon aft operation',
          'mo24':    '24mon aft operation',
          'mo36':    '36mon aft operation',
          'mo48':    '48mon aft operation',
          'mo60':    '60mon after operation'}
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
    return ds

def get_stats(f):
    data=f['data']
    #calculate number of records/measurements per patient at each time point
    rec_per_tpt = np.zeros(data.shape[0:2])
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            rec_per_tpt[j,i]=sum([1 for x in range(data.shape[2]) if str(data[j,i,x]) != 'nan' and data[j,i,x] != 0])
    np.savetxt('all_stats.txt',rec_per_tpt,fmt='%d')

#linear_interpolate one patient one columns
def linear_interpolation(timelocation,allvalues):
    #print("start----"+str(allvalues))
    for i in range(1,len(timelocation)):
        if str(allvalues[i]) == 'nan':#if the value missing,check if it is 3moth missing or not
            #if it is not, then do interpolate
            if i != 1:
                start_index = i-1
                for j in range(i+1,len(timelocation)):
                    if str(allvalues[j]) !='nan':
                        if start_index != -1 and str(allvalues[start_index]) != 'nan':
                            stop_index = j
                            t_start = timelocation[start_index]
                            t_stop = timelocation[stop_index]
                            value_start = allvalues[start_index]
                            value_stop = allvalues[stop_index]
                            slope=(value_stop - value_start) / (t_stop - t_start)
                            for k in range(start_index+1,stop_index):
                                diff_t=timelocation[k] - timelocation[k-1]
                                allvalues[k] = allvalues[k-1] + slope * diff_t
                            start_index = stop_index
    return allvalues

#========================================
def plot_picture(data,item):
    fig =plt.figure()
    plt.hist(data,bins=10)
    plt.ylabel('# Patients')
    plt.xlabel('Normalized Unit')
    plt.title(item)
    return fig
#========================================
def plot_hist(data,rest_features):
    #data = f['normalize']
    pp=PdfPages('histogram_result.pdf')
    for i in range(data.shape[2]):
        feature_array=np.ravel(data[:,:,i])
        temp=[feature_array[i] for i in range(len(feature_array)) if not np.isnan(feature_array[i])]
        item = rest_features[i]
        plotpicture=plot_picture(temp,item)
        pp.savefig(plotpicture)
    pp.close()

def is_important(feature,important):
    if feature in important:
        return True
    else:
        return False

def build_cm(data,import_features,all_features,all_ids):
    #data   -   Numpy array: dtype = float
    #       -   dimensions:     patients x timepoints x features
    pt_idx = data.shape[0] - 1# - 2  #patient index starts at bottom (see below about -2)
    ft_idx = data.shape[2] - 1  #feature index starts at right
    pt_list = all_ids   #list of remaining patients
    ft_list = all_features  #list of remaining features
    temp_mat = data[:,:,:]    #-2 is to remove 2 'useless' patients
    path_f = open('path.txt','w')
    rem_f = open('removal_order.txt','w')
    cm = np.zeros([data.shape[0],data.shape[2]])    #final density matrix
    f_cm = open('cm.txt','w')
    while pt_idx != 0 and ft_idx != 0:  #loop until reaching top left corner
        path_f.write('(%d,%d)\n' % (pt_idx,ft_idx)) #i.e. all patients/features removed
        #sort patients(rows) and features(cols) in order of increasing number of NaN's
        p_order, f_order, p_max, f_max = sort_by_nan(temp_mat,pt_list,ft_list,import_features)
        #reorder current matrix and lists
        temp_mat = temp_mat[p_order,:,:]
        temp_mat = temp_mat[:,:,f_order]
        pt_list = [pt_list[p_order[x]] for x in range(len(p_order))]
        ft_list = [ft_list[f_order[x]] for x in range(len(f_order))]
        #fill in density matrix corresponding to the current row/column
        #NOTE: the opposite of the row/column that is removed will be assessed
        #again and the value replaced if better
        for i in range(pt_idx,-1,-1):
            this_count = np.count_nonzero(np.isnan(temp_mat[0:i+1,:,0:ft_idx+1]))
            if this_count > cm[i,ft_idx]:
                cm[i,ft_idx] = this_count#/(data.shape[-1]*data.shape[1])
        for i in range(ft_idx,-1,-1):
            this_count = np.count_nonzero(np.isnan(temp_mat[0:pt_idx+1,:,0:i+1]))#/(data.shape[0]*data.shape[1])
            if this_count > cm[pt_idx,i]:
                cm[pt_idx,i] = this_count
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
    return cm

def build_cm_iter(data,cost,pt_list,ft_list,import_features):
    #global cost
    pt_idx = data.shape[0]-1#patient index starts at bottom
    ft_idx = data.shape[2]-1#feature index starts at right (back)
    #sort the patients and features in order of decreasing 'fullness'
    p_order, f_order, p_max, f_max = sort_by_nan(data,pt_list,ft_list,import_features)
    data = data[p_order,:,:]
    data = data[:,:,f_order]
    pt_list = [pt_list[i] for i in p_order]
    ft_list = [ft_list[i] for i in f_order]
    #calculate the density at this point and update if better than any previous
    #paths to this point
    this_count = np.count_nonzero(np.isnan(data))/np.prod(data.shape)
    if cost[pt_idx][ft_idx].cost > 0:
        if this_count < cost[pt_idx][ft_idx].cost:
            cost[pt_idx][ft_idx].cost = this_count
            cost[pt_idx][ft_idx].pt_list = pt_list
            cost[pt_idx][ft_idx].ft_list = ft_list
    #if this is the first time visiting this point (assumes no all-0 sub-matrices)
    else:
        cost[pt_idx][ft_idx].cost = this_count
        cost[pt_idx][ft_idx].pt_list = pt_list
        cost[pt_idx][ft_idx].ft_list = ft_list
    #recursive call one step up
    if pt_idx > 0:
        cost = build_cm_iter(data[:-1,:,:],cost,pt_list[:-1],ft_list,import_features)
    #recursive call one step to the left
    if ft_idx > 0:
        cost = build_cm_iter(data[:,:,:-1],cost,pt_list,ft_list[:-1],import_features)
    return cost

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
    #     if is_important(features[f_order[i]],important):
    #         continue
    #     else:
    #         if count != i and count < len(important):
    #             j = i
    #             while j < n_feats and is_important(features[f_order[j]],important):
    #                 j += 1
    #             if j == len(f_order):
    #                     break
    #             temp = f_order[j]
    #             for k in range(j,i,-1):
    #                 f_order[k] = f_order[k-1]
    #             f_order[i] = temp
    #     count += 1
    return p_order, f_order, p_max, f_max
