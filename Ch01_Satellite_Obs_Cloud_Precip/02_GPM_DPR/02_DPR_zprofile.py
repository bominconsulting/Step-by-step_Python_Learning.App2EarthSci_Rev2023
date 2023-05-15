
# Code Description ===============================================
# 12/26/2019 Jihoon Ryu
# Satellite part I-2: GPM/DPR
# Purpose   : How to read and plot a GPM/DPR Ku-band reflectivity profile along track in python 3.7
# Used data : GPM/DPR V6. / granule: 024778 (typhoon 'MARIA' case)
# Variable  : Near-surface rain rate
# Tested under: Python 3.7.3 (IPython 7.10.1) / Anaconda 4.8.0

# 01/30/2023 Jihoon Ryu
#  Tested under: Python 3.9.13 (IPython 7.31.1) / Anaconda 23.1.0
# ================================================================

# imported modules ======================
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pdb
# =======================================

# file name
filename = './data/sample_2A.GPM.DPR.V9-20211125.20180709-S090918-E104150.024778.V07A.HDF5'

# --------------------------------------------
# 1. Read HDF5 data
# --------------------------------------------
with h5py.File(filename, 'r') as f:
	var  = '/FS/SLV/zFactorFinal'				# Variable information from HDF5 header file
	longitude  = '/FS/Longitude'						# Longitude
	latitude   = '/FS/Latitude'						# Latitude
	data = f[var][:]										# Read data
	unit = f[var].attrs['units'].decode('ascii') # Read unit
	lon  = f[longitude][:]								# Read longitude
	lat  = f[latitude][:]								# Read latitude

# --------------------------------------------
# 2. Plot HDF5 data
# --------------------------------------------
# 2.1 Pre-setting to plot ========
# 1) Domain setting
st_lon = 120; ed_lon = 140; st_lat = 10; ed_lat = 30

# 2-1) Track and cross-section of Z profile from A to B
tlon = lon[:,15]
tlat = lat[:,15]

track=np.where( (tlat>20) & (tlat<25) & (tlon>125) & (tlon<135) )
tlon=tlon[track]
tlat=tlat[track]

# 2-2) Extract cross-section of Z profile
zprof = data[track,15,:,0]
fail= np.where(zprof < 0)
zprof[fail] = 0
zprof=zprof[0,:,:]
zprof=zprof.T   # Transpose (x,y) -> (y,x)

# 2-3) Setting Axises
xaxis = np.arange(124)*5.
yaxis = 22 - np.arange(176)*0.125

# 3) Data range
vmin = 0.
vmax = 50.
# 2.2 Plot =======================
plt.close()

# 1) Figure size
fig,ax =plt.subplots(figsize=(16,6))
ctable='jet'  ##plt.cm.jet

# 2) Contour
res=5 # contour resolution
level_f = np.arange(vmax/res+1)*res  # levels of contour
m = ax.contourf(xaxis, yaxis, zprof,levels = level_f, extend='max')
m.set_cmap(ctable)

# 3) Information of figure
plt.title('DPR Ku-band reflectivity from A to B [2018.07.09. 10:11 UTC]',fontsize=30)
plt.xticks(fontsize=20)
plt.yticks(np.arange(6)*3, fontsize=20)
plt.ylim((0,15))
plt.xlabel('Distance from A [km]', fontsize=20)
plt.ylabel('Height [km]', fontsize=20)

# 4) Colorbar
cb=plt.colorbar(m,orientation="horizontal",fraction=0.05,aspect=50,pad=0.2)
cb.set_label('Radar reflectivity ['+unit+']',fontsize=20)
cb_thick = 5
level_cb = np.arange(vmin,vmax+cb_thick,cb_thick) 				# colorbar tick
cb.set_ticks(np.int_(level_cb))
cb.set_ticklabels(np.int_(level_cb))
#cb.ax.tick_params(labelsize=20)

# 5) Save figure
plt.tight_layout() # image fitting to layout
fig.savefig('./pics/DPR_Zprofile_sample.png') # save image
plt.show()
plt.close()
