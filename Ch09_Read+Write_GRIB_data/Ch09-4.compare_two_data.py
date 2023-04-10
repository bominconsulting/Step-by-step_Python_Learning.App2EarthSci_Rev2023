# 4.compare_two_data.py
import pygrib as pg
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec


idir='./Data/'
ifnam='U.19500101.grb'
ifnam2='U.19500101.mdf.grb'
fi=pg.open(idir+ifnam)
fi2=pg.open(idir+ifnam2)
# 인덱스 21번은 00시 500 hPa 자료에 해당합니다.
grb=list(fi)[21]
grb2=list(fi2)[21]

lats,lons=grb.latlons()

data=grb.values
data2=grb2.values
gs1=gridspec.GridSpec(1, 2, left=0.1, right=0.9, top=0.9,\
 bottom=0.1)
for i in range(2):
        if i==0:
                result=list(data)
        if i==1:
                result=list(data2)
        ax=plt.subplot(gs1[i],projection=ccrs.PlateCarree())
        c=ax.pcolormesh(lons, lats, result,\
 transform=ccrs.PlateCarree(), cmap='coolwarm')

        gl = ax.gridlines(crs=ccrs.PlateCarree(),\
 draw_labels=True, linewidth=1, color='gray',\
 alpha=0.5, linestyle='--')
        ##gl.xlabels_top = False
        ##gl.ylabels_left = False
        gl.top_labels = False
        gl.left_labels = False
        gl.xlines = False
        gl.xlabel_style={'size':8}
        gl.ylabel_style={'size':8}

        cbar=plt.colorbar(c,ax=ax, orientation='horizontal')
        cbar.set_label('Zonal Wind (m/s)')

        plt.title("U at 500 hPa")
plt.gca().coastlines()

plt.show()
