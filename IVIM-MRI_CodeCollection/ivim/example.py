import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import nibabel
import os

from wrappers.ivim_fit import ivim_fit
from wrappers.OsipiBase import OsipiBase
from standardized.IAR_LU_biexp import IAR_LU_biexp




# load and init b-values
bvec = np.genfromtxt('./downloads/Data/brain.bvec')
bval = np.genfromtxt('./downloads/Data/brain.bval')
#load nifti
data = nibabel.load('./downloads/Data/brain.nii.gz')
datas = data.get_fdata() 
#choose a voxel
x=60
y=60
z=30
sx, sy, sz, n_bval = datas.shape 
# b-values without normalization
data_vox=np.squeeze(datas[x,y,z,:])
# normalise data
selsb = np.array(bval) == 0
S0 = np.nanmean(data_vox[selsb], axis=0).astype('<f')
data_vox = data_vox / S0

direction = 6 #choose: 1, 2, 3, 4, 5, or 6
signal_1dir=data_vox[direction:None:6]
signal_1dir=np.insert(signal_1dir, 0, 1)

algorithm = IAR_LU_biexp()

fit = OsipiBase.osipi_fit(algorithm, signal_1dir, np.unique(bval))


#plot the results of algorithm
plt.subplot(121)
plt.plot(np.unique(bval),signal_1dir,'x')
plt.plot(np.unique(bval),fit['f']*np.exp(-np.unique(bval)*fit['Dp'])+(1-fit['f'])*np.exp(-np.unique(bval)*fit['D']))
plt.plot(np.unique(bval),fit['f']*np.exp(-np.unique(bval)*fit['Dp']))
plt.plot(np.unique(bval),(1-fit['f'])*np.exp(-np.unique(bval)*fit['D']))
plt.legend(['measured data','model fit','Dp','D'])
plt.ylabel('S/S0')
plt.xlabel('b-value [s/mm^2]')
plt.title('algorithm IAR_LU_biexp')




#pick a slice
slice_data=np.squeeze(datas[:,:,30,:])
#fit the IVIM model for 1 direction
direction = 6 #choose: 1, 2, 3, 4, 5, or 6
slice_data_1dir=slice_data[:,:,direction:None:6] #pick the signal with the same bvec
slice_data_1dir=np.insert(slice_data_1dir, 0 , slice_data[:,:,0], axis=2) #add S0
#reshape data for fitting
sx, sy, n_bval = slice_data_1dir.shape 
X_dw = np.reshape(slice_data_1dir, (sx * sy, n_bval))
#select only relevant values, delete background and noise, and normalise data
selsb = np.array(np.unique(bval)) == 0
S0 = np.nanmean(X_dw[:, selsb], axis=1)
S0[S0 != S0] = 0
S0=np.squeeze(S0)
valid_id = (S0 > (0.5 * np.median(S0[S0 > 0]))) 
slice_data_norm = X_dw[valid_id, :]


#plot example of b=0 image
fig = plt.figure(figsize=(15, 4))
plt.subplot(131)
plt.imshow(np.squeeze(slice_data[:,:,0]), cmap='gray')
plt.title('image at b=0 mm2/s')
plt.colorbar()
plt.show()


# ### Apply the fit to 3D image and save the IVIM parameter maps

# #fit the IVIM model for 1 direction
# direction = 6 #choose: 1, 2, 3, 4, 5, or 6
# data_1dir=datas[:,:,:,direction:None:6] #pick the signal with the same bvec
# data_1dir=np.insert(data_1dir, 0 , datas[:,:,:,0], axis=3) #add S0
# #reshape data for fitting
# sx, sy, sz, n_bval = data_1dir.shape 
# X_dw = np.reshape(data_1dir, (sx * sy * sz, n_bval))
# #select only relevant values, delete background and noise, and normalise data
# selsb = np.array(np.unique(bval)) == 0
# S0 = np.nanmean(X_dw[:, selsb], axis=1)
# S0[S0 != S0] = 0
# S0=np.squeeze(S0)
# valid_id = (S0 > (0.5 * np.median(S0[S0 > 0]))) 
# data_norm = X_dw[valid_id, :]
# data_norm.shape

# #apply algorithm 1 to the diffusion data to obtain f, D* and D for each voxel in the slice
# maps = OsipiBase.osipi_fit(algorithm,data_norm,np.unique(bval))

# f_array=maps["f"]
# Dstar_array=maps["D*"]
# D_array=maps["D"]

# f_map = np.zeros([sx * sy * sz])
# f_map[valid_id] = f_array[0:sum(valid_id)]
# f_map = np.reshape(f_map, [sx, sy, sz])

# Dstar_map = np.zeros([sx * sy * sz])
# Dstar_map[valid_id] = Dstar_array[0:sum(valid_id)]
# Dstar_map = np.reshape(Dstar_map, [sx, sy, sz])

# D_map = np.zeros([sx * sy * sz])
# D_map[valid_id] = D_array[0:sum(valid_id)]
# D_map = np.reshape(D_map, [sx, sy, sz])

# # save these volumes as nii.gz files
# savedir=('./downloads/Data')
# nibabel.save(nibabel.Nifti1Image(f_map, data.affine, data.header),'{folder}/f.nii.gz'.format(folder = savedir))
# nibabel.save(nibabel.Nifti1Image(Dstar_map, data.affine, data.header),'{folder}/Dstar.nii.gz'.format(folder = savedir))
# nibabel.save(nibabel.Nifti1Image(D_map, data.affine, data.header),'{folder}/D.nii.gz'.format(folder = savedir))