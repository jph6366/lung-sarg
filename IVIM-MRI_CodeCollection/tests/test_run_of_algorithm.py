import sys

import pytest
sys.path.append(r"/home/jphardee/Desktop/GSOC/RADPI_IVIM-MRI_CodeCollection")


import os
import numpy as np
from pathlib import Path
#from ivim.standardized.ETP_SRI_LinearFitting import ETP_SRI_LinearFitting
from ivim.standardized.IAR_LU_biexp import IAR_LU_biexp
# from ivim.standardized.IAR_LU_modified_mix import IAR_LU_modified_mix
#from ivim.standardized.IAR_LU_segmented_2step import IAR_LU_segmented_2step
#from ivim.standardized.PvH_KB_NKI_IVIMfit import PvH_KB_NKI_IVIMfit
#from ivim.standardized.PV_MUMC_biexp import PV_MUMC_biexp
# from ivim.standardized.OGC_AmsterdamUMC_biexp import OGC_AmsterdamUMC_biexp


@pytest.fixture
def model():
    return IAR_LU_biexp(bounds=([0, 0, 0, 0], [1, 1, 1, 1]))


## Simple test code... 
# Used to just do a test run of an algorithm during development
def test_run(model, **kwargs):
    bvalues = np.array([0, 50, 100, 150, 200, 500, 800])

    def ivim_model(b, S0=1, f=0.1, Dstar=0.01, D=0.001):
        return S0*(f*np.exp(-b*Dstar) + (1-f)*np.exp(-b*D))

    signals = ivim_model(bvalues)
    data = np.array([signals, signals, signals])
    #print(data)
    signals = data

    #model = ETP_SRI_LinearFitting(thresholds=[200])
    if kwargs:
        results = model.osipi_fit(signals, bvalues, **kwargs)
    else:
        results = model.osipi_fit(signals, bvalues)
    print(results)
    #test = model.osipi_simple_bias_and_RMSE_test(SNR=20, bvalues=bvalues, f=0.1, Dstar=0.03, D=0.001, noise_realizations=10)
    
#model1 = ETP_SRI_LinearFitting(thresholds=[200])
#model2 = IAR_LU_modified_mix()
# model2 = OGC_AmsterdamUMC_biexp()

#dev_test_run(model1)

def test_run_level_1(model):
    test_run(model)