# 5.check_data.py
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

idir='./Data/OMPS/Y2018/'
ifnam='TCO.20180101_20181231.bin'

nx=360
ny=180
nt=365
# 바이너리 파일 불러오기
with open(idir+ifnam,"rb") as f:
    buff=np.fromfile(f,dtype=np.float32)

buff=buff.reshape((nt,ny,nx))

# 2019년 1월 1일 자료만 추출
data=buff[0,:,:]
# -999인 missing 값을 nan 처리
data[data==-999]=np.nan
lons=np.arange(0.5,360.5,1)
lats=np.arange(-89.5,90.5,1)

ax=plt.axes(projection=ccrs.PlateCarree())
c=ax.pcolormesh(lons, lats, data,\
        transform=ccrs.PlateCarree(), cmap='coolwarm')

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,\
        linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.left_labels = False
gl.xlines = False
gl.xlabel_style={'size':8}
gl.ylabel_style={'size':8}

cbar=plt.colorbar(c,ax=ax, orientation='horizontal')
cbar.set_label('TCO (DU)')

plt.title('2018.01.01 TCO')

plt.gca().coastlines()
plt.show()
