# Code Description ===============================================
# 12/24/2019 Jihoon Ryu
#  Satellite part III: Terra/MODIS
#  Purpose   : How to read and plot a MODIS MOD012KM data in python 3.7
#  Used data : Terra/MODIS MOD021KM Band 31 (11 um)
#  Variable  : Radiance Brightness temperature
#  Tested under: Python 3.7.3 (IPython 7.10.1) / Anaconda 4.8.0

# 01/30/2023 Jihoon Ryu
#  Tested under: Python 3.9.13 (IPython 7.31.1) / Anaconda 23.1.0

# More band information of MODIS on: http://ocean.stanford.edu/gert/easy/bands.html
# More comprehensive examples: http:/`/hdfeos.org/zoo/index_openLAADS_Examples.php#MODIS
# ================================================================

# Imported modules ======================
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from pyhdf.SD import SD, SDC
import rad
import pdb
# =======================================

   # --------------------------------------------
   # 1. File list and variable name
   # --------------------------------------------

data_flist = os.listdir('./data/radiance') # MOD021KM (Radiance data)
geo_flist = os.listdir('./data/geo') 		 # MOD03 (Geolocation data)

# Variable name from MOD021KM header file
var_name = 'EV_1KM_Emissive'
lat_name = 'Latitude'
lon_name = 'Longitude'

bt_all, lon_all,  lat_all= [], [], []
for data_fn in sorted(data_flist):

	if data_fn.strip().split('.')[-1].lower() != 'hdf':
		continue
	else:
		fn_indicator= '.'.join(data_fn.strip().split('.')[1:4])
		geo_fname= []
		for geo_fn in geo_flist:
			if fn_indicator in geo_fn:
				geo_fname= './data/geo/'+geo_fn
				break
		if len(geo_fname)==0:
			sys.exit('No matching geo file for {}'.format(fn_indicator))

	data_fname = './data/radiance/'+data_fn
	print(data_fn,geo_fn)

	# --------------------------------------------
	# 2. Read HDF4 data
	# --------------------------------------------

	# 2-1) Raw data of radiance
	hdf = SD(data_fname, SDC.READ)
	data_2d = hdf.select(var_name)
	raw_data = data_2d[:,:]
	raw_data = raw_data[10,:,:] # Band 31

	# 2-2) Get Attibutes and radiance calculation
	attr = data_2d.attributes(full=1)
	aoa  = attr["radiance_offsets"]; 	add_offset = aoa[0][10]		# 10: Band 31
	sfa  = attr["radiance_scales"];		scale_factor = sfa[0][10]
	# Get Radiance
	data = (raw_data-add_offset)*scale_factor
	BT = rad.RAD2BT(11., data)

	# 2-3) Geolocation data
	hdf_geo = SD(geo_fname, SDC.READ)
	latitude  = hdf_geo.select(lat_name); 	lat = latitude[:,:]
	longitude = hdf_geo.select(lon_name); 	lon = longitude[:,:]

	# 2-4) Concatenate two data
	bt_all.append(BT)
	lon_all.append(lon)
	lat_all.append(lat)

bt_all= np.concatenate(bt_all,axis=0)
lon_all= np.concatenate(lon_all,axis=0)
lat_all= np.concatenate(lat_all,axis=0)
	# --------------------------------------------

   # --------------------------------------------
   # 3. Plot
   # --------------------------------------------

# 3-1) Domain and data range setting
# Domain
st_lon =105; ed_lon = 155; st_lat = 15; ed_lat = 60
# data range
vmin = 220 # [K]
vmax = 300 # [K]

# 3-2) Plot
plt.close()

# Figure size
fig,ax =plt.subplots(figsize=(20,17))
##ctable= 'jet' ##plt.cm.jet

# Setting up the map
#m = Basemap(projection='ortho', resolution='l',area_thresh=1000.,llcrnrlon=st_lon,urcrnrlon=ed_lon,llcrnrlat=st_lat,urcrnrlat=ed_lat,lon_0=125, lat_0=30.)
m = Basemap(projection='lcc', resolution='l',area_thresh=1000.,llcrnrlon=st_lon,urcrnrlon=ed_lon,llcrnrlat=st_lat,urcrnrlat=ed_lat,lon_0=125, lat_0=30.)
#m = Basemap(resolution='l',area_thresh=1000.,llcrnrlon=st_lon,urcrnrlon=ed_lon,llcrnrlat=st_lat,urcrnrlat=ed_lat,projection='cyl',lon_0=0, lat_0=0.)
#m = Basemap(projection='cyl', resolution='l',area_thresh=1000.,lon_0=125, lat_0=40.)
m.drawcoastlines(color='black', linewidth=1)    # Coast lines
m.drawcountries(color='black', linewidth=1)     # Country lines
#m.bluemarble(scale=0.5,alpha=0.7)
m.etopo(scale=2.0,alpha=0.5)
### More information about map projection in python: https://matplotlib.org/basemap/users/mapsetup.html

# Map grid
dlon = dlat = 10
parallels  = np.arange(st_lat,ed_lat+dlat,dlat)
#m.drawparallels(parallels,linewidth=1,color='black')
m.drawparallels(parallels,labels=[1,0,0,0],linewidth=1,fontsize=24)
meridians = np.arange(st_lon,ed_lon+dlon,dlon)
m.drawmeridians(meridians,labels=[0,0,0,1],linewidth=1,fontsize=24)
#m.drawmeridians(meridians,linewidth=1,color='black')

# Data plot on the map

#m.scatter(lon_all, lat_all, c=bt_all, s=1, cmap=ctable, vmin=vmin, vmax=vmax)
#m.scatter(lon, lat, c=bt, s=1, cmap=ctable, vmin=vmin, vmax=vmax)
xaxis,yaxis = m(lon_all,lat_all)
#xaxis=lon_all
#yaxis=lat_all
zprof=bt_all
#level_f = np.arange(vmax/res+1)*res  # levels of contour
#cs = m.contourf(xaxis, yaxis, zprof,levels = np.arange(70)+240, extend='both',cmap=plt.cm.RdYlBu_r)
cs = m.contourf(xaxis, yaxis, zprof,levels = np.arange(71)+240, extend='both',cmap='jet') ##plt.cm.jet)
#cs.set_cmap=(ctable)
#m.set_cmap=('bone')
#cb = m.colorbar(extend='both',pad=0.2,location='right',fraction=0.05,aspect=50)
#cb.ax.tick_params(labelsize=30)
#cb.set_ticks(np.arange(240,320,10))
#cb.set_label('Brightness Temperature [K]' ,size=20)
plt.tight_layout() # image fitting to layout

fig.savefig('./pics/MOD021KM_sample_1_jet_tp2_lcc_large.png') # save image
plt.show() # show image
