#5.read_grb_data.py
import pygrib as pg
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec

idir='./data/'
ifnam='U.19500101.grb'
fi=pg.open(idir+ifnam)
# 00시 500 hPa 자료 추출
grb=list(fi)[21]

# data 메소드를 통해 북위 30~60, 동경 100~150도 영역을 추출
data,lats,lons=grb.data(lat1=30,lat2=60,lon1=100,lon2=150)

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
cbar.set_label('Zonal Wind (m/s)')

plt.title('U at 500 hPa over East Asia')

plt.gca().coastlines()
plt.show()
