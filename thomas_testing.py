#!/usr/bin/env python
# coding: utf-8

# In[53]:


import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

###
# FUNCTIONS
def is_file(path):
    return os.path.isfile(path)

def runNumToString(num):
    numstr = str(num)
    while len(numstr)<4:
        numstr = '0'+numstr
    return numstr
    
def is_leaf(dataset):
    return isinstance(dataset,h5py.Dataset)

def get_leaves(f,saveto,verbose=False):
    def return_leaf(name):
        if is_leaf(f[name]):
            if verbose:
                print(name,f[name][()].shape)
            saveto[name] = f[name][()]
    f.visit(return_leaf)

def combineRuns(runNumbers,folder,verbose=False):
    data_array = []
    for i,runNumber in enumerate(runNumbers):
        data = {}
        filename = f'{folder}cxilx9320_Run{runNumToString(runNumber)}.h5'
        if is_file(filename):
            print('%s, FILE EXISTS, CONTINUE...' % filename)
        else:
            print('%s, FILE DOES NOT EXIST!' % filename)
        with h5py.File(filename,'r') as f:
            get_leaves(f,data,verbose=verbose)
            data_array.append(data)
    hf1 = h5py.File(filename, 'r')
#    for key in hf1['epicsAll']:
#        print(key)
    data_combined = {}
    for key in keys_to_combine:
        arr = np.squeeze(data_array[0][key])
        for data in data_array[1:]:
            arr = np.concatenate((arr,np.squeeze(data[key])),axis=0)
        data_combined[key] = arr
    run_indicator = np.array([])
    for i,runNumber in enumerate(runNumbers):
        run_indicator = np.concatenate((run_indicator,runNumber*np.ones_like(data_array[i]['tt/FLTPOS'])))
    data_combined['run_indicator'] = run_indicator
    for key in keys_to_sum:
        arr = np.zeros_like(data_array[0][key])
        for data in data_array:
            arr += data[key]
        data_combined[key] = arr
    for key in keys_to_check:
        arr = data_array[0][key]
        for i,data in enumerate(data_array):
            if not np.array_equal(data[key],arr):
                print(f'Problem with key {key} in run {runNumbers[i]}')
        data_combined[key] = arr
    return data_combined
# END FUNCTIONS
###

# KEYS
keys_to_combine = ['tt/FLTPOS',
                   'tt/AMPL',
                   'tt/FLTPOSFWHM',
                   'tt/ttCorr',
                   'tt/FLTPOS_PS',
                   'tt/AMPL',
                   'tt/AMPLNXT',
                   'tt/FLTPOS',
                   'tt/FLTPOS_PS',
                   'tt/REFAMPL',
                   #'scan/lxt_ttc',
                   'ipm_dg2/sum',
                   'ipm_dg3/sum',
                   'gas_detector/f_11_ENRC',
                   #'epicsAll/gasCell_pressure',
                   'jungfrau4M/azav_azav',
                   'evr/code_183',
                   'evr/code_162',
                   'evr/code_141',
                   #'epicsAll/gasCell_pressure',
                   'lightStatus/laser',
                   'lightStatus/xray',
                   # 'scan/var0',
                   #  'scan/LAS:FS5:MMS:PH',
                  ]

keys_to_sum = ['Sums/jungfrau4M_calib',
              'Sums/jungfrau4M_calib_thresADU1']

keys_to_check = ['UserDataCfg/jungfrau4M/azav__azav_q',
                'UserDataCfg/jungfrau4M/azav__azav_qbin',
                'UserDataCfg/jungfrau4M/azav__azav_qbins',
                'UserDataCfg/jungfrau4M/x',
                'UserDataCfg/jungfrau4M/y',
                'UserDataCfg/jungfrau4M/z',
                'UserDataCfg/jungfrau4M/azav__azav_matrix_q',
                'UserDataCfg/jungfrau4M/azav__azav_matrix_phi',
                'UserDataCfg/jungfrau4M/cmask',
                #'UserDataCfg/jungfrau4M/Full_thres__Full_thres_thresADU',
                #'UserDataCfg/jungfrau4M/Full_thres__Full_thres_bound',
                'UserDataCfg/jungfrau4M/common_mode_pars']

# END KEYS
###

# CREATE DATA FROM COMBINED RUNS
# 50 Dark
# 51 Blank
# 52 Neon
# 53 SF6
runNumbers = [53]
folder = '/cds/data/drpsrcf/cxi/cxilx9320/scratch/hdf5/smalldata/'
azav_total = np.zeros(35)
for i in [[53], [53]]:
#for i in [[52]]:

    runNumbers = i
    data = combineRuns(runNumbers, folder=folder, verbose=False)

    # q definition
    q     = data['UserDataCfg/jungfrau4M/azav__azav_q'    ]
    qbin  = data['UserDataCfg/jungfrau4M/azav__azav_qbin' ]
    qbins = data['UserDataCfg/jungfrau4M/azav__azav_qbins']
    # azimuthal average
    azav  = data['jungfrau4M/azav_azav']
    print('azav shape: '); print(azav.shape)
    #azav = np.squeeze(data['jungfrau4M/azav_azav'])
    #print(azav.shape)
    azav_sum = np.sum(azav, axis=0)
    azav_sum_normalised = azav_sum / np.max(azav_sum)
    azav_total += azav_sum_normalised

# Normalise again outside of loop
azav_total /= np.max(azav_total) 
plt.figure()
plt.plot(q, azav_total)
plt.xlabel('~q')
plt.ylabel('Counts')
plt.title('Azav')
plt.savefig('azav_SF6.png')

# may not need:
x = data['UserDataCfg/jungfrau4M/x']
y = data['UserDataCfg/jungfrau4M/y']
z = data['UserDataCfg/jungfrau4M/z']

print('Vector q ='); print(q)
print('Size of qbin increment: %f (A-1)' % qbin)
print('qbins:'); print(qbins)
#print(x)
#print(z)


# In[ ]:




