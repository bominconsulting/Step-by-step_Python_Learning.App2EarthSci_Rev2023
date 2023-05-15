# Code Description ===============================================
# 12/24/2019 Jihoon Ryu
# Satellite part II: GPM/GMI
# Purpose   : How to read and plot a GPM/GMI L1B data in python 3.7
# Used data : GPM/GMI V5. / granule number: 031275~31289
# Variable  : Near-surface rain rate
# Tested under: Python 3.7.3 (IPython 7.10.1) / Anaconda 4.8.0
# You can access more informatio on: https://hdfeos.org/zoo/index_openGESDISC_Examples.php

# 01/30/2023 Jihoon Ryu
#  Tested under: Python 3.9.13 (IPython 7.31.1) / Anaconda 23.1.0
# ================================================================

# Imported modules ======================
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pdb
import os
# =======================================

nf=0
flist = os.listdir('./data/')

for gr in np.arange(np.size(flist)):
	filename = './data/'+flist[gr]
	print(filename)
	if filename.strip().split('.')[-1].lower() != 'hdf5':
		continue
	# --------------------------------------------
	# 1. Read HDF5 data
	# --------------------------------------------

	with h5py.File(filename, 'r') as f:
	   h5keys=[]
	   f.visit(h5keys.append)  # Visit every key and save in list
	   for i,key_name in enumerate(h5keys):
	       print("{}: {}".format(i,key_name))
	   var  = '/S1/Tb'								      # Variable adress in HDF5 header file
	   longitude  = '/S1/Longitude'                 # Longitude adress
	   latitude   = '/S1/Latitude'                  # Latitude adress
	   data = f[var][:]                             # Read variable(Near-surface rain rate)
	   unit = f[var].attrs['units'].decode('ascii') # Read unit
	   lon  = f[longitude][:]                       # Read longitude
	   lat  = f[latitude][:]                        # Read latitude

	# --------------------------------------------
	TB_89V=data[:,:,7]
	TB_89H=data[:,:,8]
	PCT89 = 1.7*TB_89V - 0.7*TB_89H # PCT: Polarization-corrected Brightniss temperature / coefficient 0.7 (Cecil and Chronis, 2018)
	if nf == 0: PCT89_1D = PCT89; lon_1d = lon; lat_1d = lat
	if nf != 0:
		PCT89_1D=np.concatenate((PCT89_1D,PCT89),axis=0)
		lon_1d = np.concatenate((lon_1d, lon), axis=0)
		lat_1d = np.concatenate((lat_1d, lat), axis=0)
	nf=nf+1

# 1) Domain setting
st_lon =-180; ed_lon = 180; st_lat = -90; ed_lat = 90

# data range
vmin = 200 # [K]
vmax = 300 # [K]

# 2.2 Plot =======================
plt.close()

# 1) Figure size
fig,ax =plt.subplots(figsize=(16,8))
ctable=plt.cm.jet

# 2) Setting up the map
m = Basemap(resolution='l',area_thresh=1000.,llcrnrlon=st_lon,urcrnrlon=ed_lon,llcrnrlat=st_lat,urcrnrlat=ed_lat,projection='cyl',lon_0=0, lat_0=0.)
m.drawcoastlines(color='black', linewidth=1)    # Coast lines
m.drawcountries(color='black', linewidth=1)     # Country lines
### More information about map projection in python: https://matplotlib.org/basemap/users/mapsetup.html

# Map grid
dlon = dlat = 30
parallels  = np.arange(st_lat,ed_lat+dlat,dlat)
m.drawparallels(parallels,labels=[1,0,0,0],linewidth=0.2,fontsize=12)
meridians = np.arange(st_lon,ed_lon+dlon,dlon)
m.drawmeridians(meridians,labels=[0,0,0,1],linewidth=0.2,fontsize=12)

# 3) Data plot on the map
# Non-precipitation area -> grey
#m.scatter(lon[fail], lat[fail], c='grey', s=2, edgecolors=None)
# Precipitation area -> rainbow
m.scatter(lon_1d, lat_1d, c=PCT89_1D, s=1, cmap=ctable, vmin=vmin, vmax=vmax)
#nl=vmax-vmin
#level_f=np.arange(vmin,vmax+(vmax-vmin)/(nl),(vmax-vmin)/(nl))
#im = ax.contourf(lon_1d, lat_1d, PCT89_1D,levels = level_f,symmetric_cbar='auto',extend='both')
#m.pcolormesh(lon_1d,lat_1d, PCT89_1D, cmap=ctable)
cb = m.colorbar(extend='both',pad=0.2,location='right',fraction=0.05,aspect=50)
#cb.set_ticks(level_cb)
#cb.set_ticklabels(level_cb)
cb.ax.tick_params(labelsize=20)
plt.tight_layout() # image fitting to layout
fig.savefig('./pics/GPM_PCT89_1DAY.png') # save image
#cb=plt.colorbar(m,orientation="horizontal",fraction=0.05,aspect=50,pad=0.2)

#plt.show()
