import os
import h5py
import numpy as np

class Thomas(object):
    def __init__(self, indexstart):
        # these have to be arrays for generalisation
        self.dark_run_numbers  = [50]
        self.blank_run_numbers = [51]
        self.ne_run_numbers    = [52]
        self.sf6_run_numbers   = [53]

        self.folder = '/cds/data/drpsrcf/cxi/cxilx9320/scratch/hdf5/smalldata/'

        self.ne_theory_file = 'Ne_total.txt'
        #self.sf6_theory_file = 'SF6_Debye_Total.npy'
        self.sf6_theory_file = 'SF6_total.txt'
        #self.sf6_theory_file = 'I_total_abintio_SF6_struct1.dat'

        # ignore first azav indices close to centre of jungfrau
        self.indexstart = indexstart
        
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
    ###
    # FUNCTIONS
    def is_file(self, path):
        return os.path.isfile(path)

    def runNumToString(self, num):
        numstr = str(num)
        while len(numstr)<4:
            numstr = '0'+numstr
        return numstr

    def is_leaf(self, dataset):
        return isinstance(dataset,h5py.Dataset)

    def get_leaves(self, f, saveto, verbose=False):
        def return_leaf(name):
            if self.is_leaf(f[name]):
                if verbose:
                    print(name,f[name][()].shape)
                saveto[name] = f[name][()]
        f.visit(return_leaf)

    def combineRuns(self, runNumbers, folder, verbose=False):
        data_array = []
        for i, runNumber in enumerate(runNumbers):
            data = {}
            filename = f'{folder}cxilx9320_Run{self.runNumToString(runNumber)}.h5'
            if self.is_file(filename): print('%s EXISTS, CONTINUE...' % filename)
            else: print('%s, FILE DOES NOT EXIST!' % filename)
            with h5py.File(filename, 'r') as f:
                self.get_leaves(f, data, verbose=verbose)
                data_array.append(data)
        hf1 = h5py.File(filename, 'r')
        data_combined = {}
        for key in self.keys_to_combine:
            arr = np.squeeze(data_array[0][key])
            for data in data_array[1:]:
                arr = np.concatenate((arr,np.squeeze(data[key])),axis=0)
            data_combined[key] = arr
        run_indicator = np.array([])
        for i,runNumber in enumerate(runNumbers):
            run_indicator = np.concatenate((run_indicator,runNumber*np.ones_like(data_array[i]['tt/FLTPOS'])))
        data_combined['run_indicator'] = run_indicator
        for key in self.keys_to_sum:
            arr = np.zeros_like(data_array[0][key])
            for data in data_array:
                arr += data[key]
            data_combined[key] = arr
        for key in self.keys_to_check:
            arr = data_array[0][key]
            for i,data in enumerate(data_array):
                if not np.array_equal(data[key],arr):
                    print(f'Problem with key {key} in run {runNumbers[i]}')
            data_combined[key] = arr
        return data_combined
    # END FUNCTIONS
    ###


    # THOMAS FUNCTIONS
    def normalise(self, x, f):
        dx = abs(x[1] - x[0])
        area = dx * np.sum(f)
        f /= area
        return f

    def get_azav(self, runNumbers, indexstart):
        '''
        Returns normalised azimuthal average of Jungfrau for array of 
        run numbers, returns data[indexstart:]
        '''
        data = self.combineRuns(runNumbers, folder=self.folder, verbose=False)
        # azimuthal average
        #azav = np.squeeze(data['jungfrau4M/azav_azav'])
        azav  = data['jungfrau4M/azav_azav']
        #print('azav shape: '); print(azav.shape)
        #print(azav.shape)
        azav = np.sum(azav, axis=0)
        # q definition
        q = data['UserDataCfg/jungfrau4M/azav__azav_q']
        #qbin  = data['UserDataCfg/jungfrau4M/azav__azav_qbin' ]
        #qbins = data['UserDataCfg/jungfrau4M/azav__azav_qbins']
        return q[indexstart:], azav[indexstart:]

    def load_theory(self, theory_file):
        print('Loading theory file %s' % theory_file)
        # Load theory file data
        split_tup = os.path.splitext(theory_file)
        filetype = split_tup[1]
        if filetype == '.txt' or filetype == '.dat':
            data = np.loadtxt(theory_file)
        elif filetype == '.npy':
            data = np.load(theory_file)
        if (data.shape[0] != 2):
            data = np.transpose(data)
        q_theory = data[0]
        f_theory = data[1]
        return q_theory, f_theory

    def interp_theory(self, q_exp, q_theory, f_theory):
        # interpolate to shape of exp data
        interpolated_values = np.interp(q_exp, q_theory, f_theory)  
        return interpolated_values

    def get_irf(self, azav_blank, q_exp, azav_exp, q_theory, azav_theory):
        '''
        Calculates the instrument response function (IRF)
        '''
        # Subtract blank
        azav_subtracted = azav_exp - azav_blank
        # interpolation (make theory curve same shape as experiment q-bins)
        theory_interp = self.interp_theory(q_exp, q_theory, azav_theory)
        # Normalise before creating IRF
        exp_normalised = self.normalise(q_exp, azav_subtracted)
        theory_normalised = self.normalise(q_exp, theory_interp)
        # instrument response function definition
        irf = np.divide(theory_normalised, exp_normalised)
        # convert nan and +/-inf to 0 and return
        return np.nan_to_num(irf, copy=True, posinf=0, neginf=0)

    def irf_blank_correction(self, azav, azav_blank, irf):
        '''
        Applies blank subtraction and IRF correction to an azav curve
        '''
        azav_corrected = (azav - azav_blank) * irf
        return azav_corrected
    # END THOMAS FUNCTIONS
    ###