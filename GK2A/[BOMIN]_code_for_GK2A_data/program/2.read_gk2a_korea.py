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
import pdb

# 파일 경로
path='../data/'
fname1='gk2a_ami_le1b_ir112_ko020lc_202307170000.nc'
fname2='gk2a_ami_le1b_ir123_ko020lc_202307170000.nc'
filename1=path+fname1
filename2=path+fname2

# 1. 엑셀파일의 Look-up table 읽고 int16 값을 Brightness temperatrue로 치환
with h5py.File(filename1, 'r') as f:
	var  = '/image_pixel_values'           # Variable
	data1 = f[var][:]                       # Read variable(Near-surface rain rate)

with h5py.File(filename2, 'r') as f:
	var  = '/image_pixel_values'           # Variable
	data2 = f[var][:]

# 채널정보 확인
chan_info1=fname1[14:19] # 파일명에서의 채널정보 텍스트 위치
chan_info2=fname2[14:19]

# 파일명에 사용되는 GK2A의 채널명들
chan_names=['vi004','vi005','vi006','vi008','nr013','nr016','sw038','wv063',
            'wv069','wv073','ir087','ir096','ir105','ir112','ir123','ir133']

# 파일명에서 얻은 채널정보를 바탕으로 몇번째 채널인지 index 추출
chan_num1 = chan_names.index(chan_info1)
chan_num2 = chan_names.index(chan_info2)

rad105=calrad.dn2rad(chan_num1,data1)
bt105=calbt.rad2bt(chan_num1,rad105)

rad112=calrad.dn2rad(chan_num2,data2)
bt112=calbt.rad2bt(chan_num2,rad112)

diff_bt=bt105-bt112

# 2. 좌표파일 읽어오기
fname='latlon_ko_2000.txt'
lonlat=np.loadtxt(fname, dtype="float")

lat=lonlat[:,0].reshape(900,900)
lon=lonlat[:,1].reshape(900,900)


nan_lat=np.where(lat == -999.)
nan_lon=np.where(lon == -999.)
lat[nan_lat]=np.nan; lat[nan_lon]=np.nan
lon[nan_lat]=np.nan; lon[nan_lon]=np.nan


st_lon =110.; ed_lon = 150.; st_lat = 25; ed_lat = 50.
# data range
vmin = 0 # [K]
vmax = 3 # [K]

# 2.2 Plot =======================
plt.close()

# 1) Figure size
fig,ax =plt.subplots(figsize=(20,20))
ctable=plt.cm.Greys
#ctable=plt.cm.jet

# 2) Setting up the map

m = Basemap(projection='lcc', resolution='l',area_thresh=1000.,
            llcrnrlon=st_lon,urcrnrlon=ed_lon,
            llcrnrlat=st_lat,urcrnrlat=ed_lat,
            lon_0=125, lat_0=30.)


#m = Basemap(resolution='l',
#            llcrnrlon=st_lon,llcrnrlat=st_lat,
#            urcrnrlon=ed_lon,urcrnrlat=ed_lat,
#            area_thresh=1000.,lon_0=128.2, lat_0=37.,projection='cyl')

m.drawcoastlines(color='black', linewidth=1)    # Coast lines
#m.drawcountries(color='black', linewidth=1)     # Country lines
### More information of map projection in python: https://matplotlib.org/basemap/users/mapsetup.html

# Map grid
dlon = dlat = 5
parallels  = np.arange(st_lat,ed_lat+dlat,dlat)
m.drawparallels(parallels,labels=[1,0,0,0],linewidth=0.2,fontsize=20)
meridians = np.arange(st_lon,ed_lon+dlon,dlon)
m.drawmeridians(meridians,labels=[0,0,0,1],linewidth=0.2,fontsize=20)

# 3) Plot
# Non-precipitation area -> grey
#m.scatter(lon, lat, c='grey', s=2, edgecolors=None)
# Precipitation area -> rainbow
x,y=m(lon,lat) # 격자 만들기
m.scatter(x, y, c=diff_bt, s=1, cmap=ctable, edgecolors=None, linewidth=0, vmin=vmin, vmax=vmax)


# 4) Colorbar setting
cb_thick = 0.5
level_cb = np.arange(vmin,vmax+cb_thick,cb_thick) 				# colorbar tick
cb = m.colorbar(location="bottom", pad='5%', extend='both',cmap=ctable)
cb.set_label('Difference of brightness temperature [K]',fontsize=40)
cb.set_ticks(level_cb)
cb.set_ticklabels(level_cb)
cb.ax.tick_params(labelsize=30)

# Cross-section line
#m.scatter(tlon,tlat, c='red', s=1.5, edgecolors=None)
#plt.title('GPM/DPR rain rate [2018.07.09. 10:11 UTC]',fontsize=22)
#plt.title('GPM/DPR rain rate [2018.07.09. 0909-1041 UTC]',fontsize=20)
plt.tight_layout() 				# image fitting to layout

# 5) Show (or Save) image
fig.savefig('../figures/Cloud_KR_sample.png')
plt.show() 		 				#


# IR10.5, IR12 는 수증기에 대한 흡수가 다름
# 상층에는 수증기가 거의 없기 때문에 구름이 상층에 있을 경우 밝기온도가 비슷하게 나타남
# 반면 구름이 없을때에는 지표에서의 밝기온도가 우주로 방출되는 동안 흡수되는 정도가 달라
# 밝기온도 차이가 크게 나타나게 됨
# 이 차이를 이용하여
