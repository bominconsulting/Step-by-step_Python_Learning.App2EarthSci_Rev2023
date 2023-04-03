import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec
import netCDF4 as nc
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
#
minlon, maxlon = 70, 160 # 경도 범위 설정
minlat, maxlat = 0, 60 # 위도 범위 설정

fpath = './Data/ERA5_TQUVZ_20200101_00.nc' # ERA5 데이터 파일
ncfile = nc.Dataset(fpath, mode='r')
lon, lat = ncfile.variables['longitude'][:], ncfile.variables['latitude'][:]
latid = np.where((lat>=minlat) * (lat<=maxlat))[0]
lonid = np.where((lon>=minlon) * (lon<=maxlon))[0]
yy0,yy1 = latid[0], latid[-1]+1
xx0,xx1 = lonid[0], lonid[-1]+1
# 위경도 범위에 해당하는 인덱스 파악

lat, lon = lat[latid], lon[lonid] # 위경도 범위에 따라 lon, lat slicing
lon2d, lat2d = np.meshgrid(lon, lat) # 2차원 위경도 자료 생성

temp = ncfile.variables['t'][0,yy0:yy1, xx0:xx1] # 위경도 범위의 온도자료 불러오기
shum = ncfile.variables['q'][0,yy0:yy1, xx0:xx1] # 위경도 범위의 비습자료 불러오기
geop = ncfile.variables['z'][0,yy0:yy1, xx0:xx1] # 위경도 범위의 지오포텐셜자료 불러오기
udat = ncfile.variables['u'][0,yy0:yy1, xx0:xx1] # 위경도 범위의 동서풍 자료 불러오기
vdat = ncfile.variables['v'][0,yy0:yy1, xx0:xx1] # 위경도 범위의 남북풍 자료 불러오기
# 위에서 t, q, z, u, v는 ERA5 파일 내의 변수 명을 의미합니다. nc 파일이 가지고 있는 변수에 대한 상세 정보는 해당 nc파일을 nc.Dataset으로 연 이후에 print (ncfile.variables)를 적용하면 손쉽게 확인 가능합니다.
ncfile.close()

fig = plt.figure(figsize = (8,6), dpi = 100)
gs1 = gridspec.GridSpec(1,1, left=0.01, right=0.9 , top = 0.95, bottom = 0.1)
cgs = gridspec.GridSpec(1,1, left=0.91, right=0.93, top = 0.90, bottom = 0.15)
cax = plt.subplot(cgs[0])
projection_type = ccrs.Mercator()
ax = plt.subplot(gs1[0], projection = projection_type) # Cartopy subplot
ax.set_extent([minlon,maxlon,minlat, maxlat],
            crs = ccrs.PlateCarree()) # ax의 위경도 범위 설정
gl = ax.gridlines(crs = ccrs.PlateCarree() ,color = 'k',
            linestyle = ':',xlocs = np.arange(minlon,maxlon+1,15),
            ylocs = np.arange(minlat,maxlat+1,15) ,zorder = 7, draw_labels=True)
# ax에 표현될 위도선 (ylocs) 및 경도선 (xlocs) 위치와 선 스타일 등을 결정합니다.
# np.arange(A,B,D)는 A 이상 B “미만”까지 D 간격으로 증가하는 어레이를 생성합니다.
# 이를 고려하여 np.arange(minlon,maxlon+1,15)로 기입, minlon부터 maxlon까지 모두 표현되도록 적용해줍니다.

gl.top_labels = False #그림 위쪽에 표현될 경도선 라벨을 보이지 않게 처리
gl.right_lables = False  #그림 오른쪽에 표현될 위도선 라벨을 보이지 않게 처리


gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
#cartopy.mpl.gridliner.LONGITUDE_FORMATTER 및 LATITUDE_FORMATTER는 위경도 라벨의 형식을 보기 좋게 변경해줍니다. (ex) 경도 -30.0  => 30ºW, 위도 50.0  => 50ºN)
ax.coastlines(linewidth = 0.5, zorder=7)  #해안선 표현

qmin, qmax = 0., 20. #표현할 비습의 최솟값, 최댓값을 설정
clevs = np.linspace(qmin,qmax,5)
#np.linspace를 이용하여 qmin부터 qmax까지를 5개의 구간으로 분할 (0,5,10,15,20)
cs = ax.pcolormesh(lon2d, lat2d, shum*1000., cmap = 'YlGnBu', transform=ccrs.PlateCarree(), vmin = qmin, vmax = qmax, zorder = 1)
# plt.pcolormesh를 적용하여 비습을 shading으로 표현
# ERA5의 경우, 비습의 단위가 kg/kg이므로 1000을 곱하여 g/kg으로 환산
# 해안선이나 위경도선, 향후 그려질 plt.contour 등을 가리지 않도록 zorder는 낮게 설정하여 그림을 배경처럼 깔아줍니다.
cbar = plt.colorbar(cs, extend='both', cax=cax, ticks = clevs, orientation = 'vertical')
#위에서 선언한 서브플롯 cax에 컬러바를 그림.

tmin, tmax = 240, 300 #일 평균 온도의 범위 설정 (unit: K)
tlev = np.arange(tmin, tmax+1, 10.) # tmin부터 tmax까지 10K 간격으로 레벨 설정.
cont1 = ax.contour(lon2d, lat2d, temp, colors = 'r', levels = tlev, zorder = 8,
                vmin = tmin, vmax = tmax,  transform=ccrs.PlateCarree(),)
#plt.pcolormesh와 동일한 방식으로 그림을 그려줍니다. 단, plt.contour가 plt.pcolormesh에 깔려서 가려지지 않도록, zorder를 더 높게 설정해줍니다.
# 주의) plt.pcolormesh와 마찬가지로, Cartopy를 이용하여 그릴 때에는 transform을 잊으면 안됩니다. transform 설정을 하지 않는 경우, 그림이 그려지지 않을 수 있습니다.
ax.clabel(cont1, inline = True, fmt = '%.1f', fontsize = 8)
# plt.contour 기능을 통하여 그려진 등온선 (cont1)을 따라 라벨을 추가해줍니다.
# inline=True인 경우, 라벨이 적히는 지점의 등온선을 생략하여 가독성을 높여줍니다.
# fmt = ‘%.Nf’는 소숫점 N 번째 자리까지 표현한다는 의미입니다.
# ‘%.1f’를 적용하는 경우에, 123.4567은 반올림되어 ‘123.5’로 표현됩니다.

zmin, zmax = 1000., 1600. # 850 hPa의 지위고도 범위
zlev = np.arange(zmin, zmax+1, 50.)  # zmin부터 zmax까지 50 m 간격으로 레벨 설정
cont2 = ax.contour(lon2d, lat2d, geop/9.8, colors = 'k', levels = zlev, zorder = 8,
                vmin = zmin, vmax = zmax,  transform=ccrs.PlateCarree(),)
# ERA5의 지오포텐셜을 지위고도 (Geopotential height)로 변환하기 위하여
# 중력가속도 9.8 m s-2로 나눠주었음.

ax.clabel(cont2, inline = True, fmt = '%.0f', fontsize = 8)
# 그려진 등지위고도선 (cont2)을 따라 라벨을 추가해줍니다.
skipp = 10 # 데이터 건너뛸 간격.
windscale = 10 # 바람장 화살표 범례에 이용할 기준 풍속값
idx = np.where(np.sqrt(udat**2. + vdat**2.) < 10.) # 풍속 10 m/s 이하 지역 파악
udat[idx] = np.nan # 동서풍 예외처리 수행
vdat[idx] = np.nan # 남북풍 예외처리 수행

quiv = ax.quiver(lon2d[::skipp, ::skipp], lat2d[::skipp, ::skipp],
    udat[::skipp, ::skipp], vdat[::skipp, ::skipp] , units = 'xy', angles = 'uv',
    zorder = 10, scale_units='inches', scale = 40, headwidth =5, linewidths = 3,
    transform=ccrs.PlateCarree(),color = 'grey')
# plt.quiver는 바람장을 시각화할때 이용되는 함수입니다. 2차원 경도, 위도, 동서풍, 남북풍 정보를 주어서 흔히 볼 수 있는 바람장 그림을 그릴 수 있습니다. 여기에서는 모든 데이터를 10 (skip) 간격으로 입력([::skipp, ::skipp])하여 성기게 바람장을 그려보았습니다. 상세한 정보는 이하의 표를 참조해주세요.

qk = ax.quiverkey(quiv, 0.9, 0.95, windscale, repr(windscale) + ' m/s', labelpos='E', coordinates='figure')
# plt.quiver를 바탕으로 화살표 범례를 생성하는 함수입니다. 앞에서 선언된 quiv를 바탕으로, figure fraction coordinates를 이용하여 좌측으로부터 90%, 하단으로부터 95% 위치에 주어진 값 (windscale)을 대상으로 화살표 범례를 그리고, 그 밑에 텍스트 정보 (repr(windscale)+‘m/s’)를 표기합니다.

plt.show()
