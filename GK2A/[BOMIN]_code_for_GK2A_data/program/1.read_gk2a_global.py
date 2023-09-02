# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 22:42:25 2023

@author: ryujih
"""

import h5py
import numpy as np
import calrad
import calbt
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# 파일 경로
path='../data/'
fname='gk2a_ami_le1b_ir105_fd020ge_202307170000.nc'
filename=path+fname


# 1. 엑셀파일의 Look-up table 읽고 int16 값을 Brightness temperatrue로 치환
with h5py.File(filename, 'r') as f:
	var  = '/image_pixel_values'           # Variable
	data = f[var][:]                       # Read variable(Near-surface rain rate)

# 채널정보 확인
chan_info=fname[14:19] # 파일명에서의 채널정보 텍스트 위치

# 파일명에 사용되는 GK2A의 채널명들
chan_names=['vi004','vi005','vi006','vi008','nr013','nr016','sw038','wv063',
            'wv069','wv073','ir087','ir096','ir105','ir112','ir123','ir133']

# 파일명에서 얻은 채널정보를 바탕으로 몇번째 채널인지 index 추출
chan_num = chan_names.index(chan_info)

rad=calrad.dn2rad(chan_num,data)
bt=calbt.rad2bt(chan_num,rad)



    
# 2. 좌표파일 읽어오기
fname_lon='Lon_2km.bin'
with open(fname_lon,'r') as f:
    lon_a=np.fromfile(f,dtype=np.float32)
lon=lon_a.reshape(5500,5500)

fname_lat='Lat_2km.bin'
with open(fname_lat,'r') as f:
    lat_a=np.fromfile(f,dtype=np.float32)
lat=lat_a.reshape(5500,5500)


nan_lat=np.where(lat == -999.)
nan_lon=np.where(lon == -999.)
lat[nan_lat]=np.nan; lat[nan_lon]=np.nan
lon[nan_lat]=np.nan; lon[nan_lon]=np.nan

idx=np.where(lon < 0)
lon[idx]=lon[idx]+360

st_lon =0.; ed_lon = 360.; st_lat = -90.; ed_lat = 90.
# data range 
vmin = 220 # [K]
vmax = 300 # [K]

# 2.2 Plot ======================= 
plt.close()

# 1) Figure size
fig,ax =plt.subplots(figsize=(20,20))
ctable=plt.cm.jet

# 2) Setting up the map
m = Basemap(resolution='l',area_thresh=10000.,lon_0=128.2, lat_0=0.,projection='cyl')
m.drawcoastlines(color='black', linewidth=1)    # Coast lines
#m.drawcountries(color='black', linewidth=1)     # Country lines
### More information of map projection in python: https://matplotlib.org/basemap/users/mapsetup.html

# Map grid
dlon = dlat = 30 
parallels  = np.arange(st_lat,ed_lat+dlat,dlat)
m.drawparallels(parallels,labels=[1,0,0,0],linewidth=0.2,fontsize=12)
meridians = np.arange(st_lon,ed_lon+dlon,dlon)
m.drawmeridians(meridians,labels=[0,0,0,1],linewidth=0.2,fontsize=12)

# 3) Plot
# Non-precipitation area -> grey
#m.scatter(lon, lat, c='grey', s=2, edgecolors=None)
# Precipitation area -> rainbow
m.scatter(lon, lat, c=bt, s=1, cmap=ctable, edgecolors=None, linewidth=0, vmin=vmin, vmax=vmax)

# 4) Colorbar setting 
cb_thick = 5
level_cb = np.arange(vmin,vmax+cb_thick,cb_thick) 				# colorbar tick
cb = m.colorbar(location="bottom", pad='5%', extend='max',cmap=ctable)  
#cb.set_label('Rain rate ['+unit+']',fontsize=15)
cb.set_ticks(level_cb)
cb.set_ticklabels(level_cb)
cb.ax.tick_params(labelsize=13)

# Cross-section line
#m.scatter(tlon,tlat, c='red', s=1.5, edgecolors=None)
#plt.title('GPM/DPR rain rate [2018.07.09. 10:11 UTC]',fontsize=22)
#plt.title('GPM/DPR rain rate [2018.07.09. 0909-1041 UTC]',fontsize=20)
plt.tight_layout() 				# image fitting to layout

# 5) Show (or Save) image
plt.show() 		 				#
    




    