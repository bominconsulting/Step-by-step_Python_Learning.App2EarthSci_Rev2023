
# Code Description ===============================================
# 12/24/2019 Jihoon Ryu
# Satellite part I-1: GPM/DPR
# Purpose	: How to read and plot a GPM/DPR L2 data in python 3.7
# Used data	: GPM/DPR V6. / granule number: 024778 (typhoon 'MARIA' case)
# Variable	: Near-surface rain rate
# Tested under: Python 3.7.3 (IPython 7.10.1) / Anaconda 4.8.0
# You can access more information on: https://hdfeos.org/zoo/index_openGESDISC_Examples.php

# 01/30/2023 Jihoon Ryu
#  Tested under: Python 3.9.13 (IPython 7.31.1) / Anaconda 23.1.0
# ================================================================

# Imported modules ======================
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
	var  = '/FS/SLV/precipRateNearSurface'			# Variable
	longitude  = '/FS/Longitude'						# Longitude
	latitude   = '/FS/Latitude'						# Latitude
	data = f[var][:]										# Read variable(Near-surface rain rate)
	unit = f[var].attrs['units'].decode('ascii') # Read unit
	lon  = f[longitude][:]								# Read longitude
	lat  = f[latitude][:]								# Read latitude


# --------------------------------------------
# 2. Plot HDF5 data
# --------------------------------------------
# 2.1 Pre-setting to plot ========

# 1) Domain setting
st_lon = 120; ed_lon = 140; st_lat = 10; ed_lat = 30

# 2) Find zero value
succ = np.where(data > 0)  # RR > 0
fail = np.where(data == 0) # RR = 0

# 2-1) Cross-section line
tlon = lon[:,15]
tlat = lat[:,15]

track=np.where( (tlat>20) & (tlat<25) & (tlon>125) & (tlon<135) )
tlon=tlon[track]
tlat=tlat[track]

# 3) Data range
vmin = 0.   # [mm/hr]
vmax = 20.  # [mm/hr]

# 2.2 Plot =======================
plt.close()

# 1) Figure size
fig,ax =plt.subplots(figsize=(8,8))
ctable='jet'  ##plt.cm.jet

# 2) Setting up the map
m = Basemap(llcrnrlon=st_lon,llcrnrlat=st_lat,urcrnrlon=ed_lon,urcrnrlat=ed_lat,
            resolution='l',area_thresh=1000.,lon_0=st_lon, lat_0=st_lat,projection='cyl')
m.drawcoastlines(color='black', linewidth=1)    # Coast lines
m.drawcountries(color='black', linewidth=1)     # Country lines
### More information of map projection in python: https://matplotlib.org/basemap/users/mapsetup.html

# Map grid
dlon = dlat = 5
parallels  = np.arange(st_lat,ed_lat+dlat,dlat)
m.drawparallels(parallels,labels=[1,0,0,0],linewidth=0.2,fontsize=12)
meridians = np.arange(st_lon,ed_lon+dlon,dlon)
m.drawmeridians(meridians,labels=[0,0,0,1],linewidth=0.2,fontsize=12)

# 3) Plot
# Non-precipitation area -> grey
m.scatter(lon[fail], lat[fail], c='grey', s=2, edgecolors=None)
# Precipitation area -> rainbow
m.scatter(lon[succ], lat[succ], c=data[succ], s=2, cmap=ctable, edgecolors=None, linewidth=0, vmin=vmin, vmax=vmax)

# 4) Colorbar setting
cb_thick = 5
level_cb = np.arange(vmin,vmax+cb_thick,cb_thick) 				# colorbar tick
cb = m.colorbar(location="bottom", pad='5%', extend='max',cmap=ctable)
cb.set_label('Rain rate ['+unit+']',fontsize=15)
cb.set_ticks(level_cb)
cb.set_ticklabels(level_cb)
cb.ax.tick_params(labelsize=13)

# Cross-section line
m.scatter(tlon,tlat, c='red', s=1.5, edgecolors=None)
#plt.title('GPM/DPR rain rate [2018.07.09. 10:11 UTC]',fontsize=22)
plt.title('GPM/DPR rain rate [2018.07.09. 0909-1041 UTC]',fontsize=20)
plt.tight_layout() 				# image fitting to layout

# 5) Show (or Save) image
plt.show() 		 				# show image
fig.savefig('./pics/DPR_RR_sample.png') # save image
plt.close()
