import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec

fpath = './Data/ndvi_0.5deg_2015.npy'
# 샘플로 제공하는 2015년도 NDVI 데이터.
# 필요 시 데이터 경로를 변경하여 사용할 것.

raw_ndvi = np.load(fpath)# npy 형태의 NDVI 공간분포데이터 읽기
# (24,360,720) 차원의 데이터로, 각각 (시간, 위도, 경도) 차원을 의미.
# 공간해상도 0.5도, 시간해상도 15일 간격의 자료. 1월부터 12월까지 월별 2회씩 총 24회의 시간변화를 보여줌.
lon = 0.5* np.arange(720) - 180. +0.25 # NDVI 경도정보 만들기
lat = 0.5* np.arange(360) - 90. +0.25  # NDVI 위도정보 만들기
jul = 15. + 15.* np.arange(24) # NDVI 시간정보 (Julian day) 만들기
latid = np.where((lat>=30.) * (lat<=80.))# 북위 30~80도 지역 좌표 확인
lat = lat[latid]
yy0,yy1 = latid[0][0], latid[0][-1]

ndvi = raw_ndvi[::2, yy0:yy1+1,:]
#분석 대상 자료 읽어오기
#해당 자료는 15일 간격의 자료로, 30일 간격으로[::2] 건너뛰며 데이터 로드.
idx = np.where((ndvi < -0.3) + (ndvi>1.0)) # 정상범위 이외의 데이터 좌표 파악
ndvi[idx] = np.nan #에러 데이터 예외처리 수행

projection_type = ccrs.Orthographic(central_longitude=0., central_latitude=90.)
minlon,maxlon,minlat,maxlat = -180,180.,30.,90.
# 이 예제에서는 Cartopy를 이용하여 시각화를 수행하였음.
# ccrs.Orthographic은 정사영도법 기반의 2차원 시각화로, 고위도 지역의 변화를 살펴보는 경우에 효과적인 도법.

fig = plt.figure(figsize = (10,6), dpi = 100)
gs1 = gridspec.GridSpec(2,3, left=0.01, right=0.9 , top = 0.95, bottom = 0.1)
# gridspec.GridSpec 의 경우, 대소문자에 유의.
# GridSpec 상세 옵션은 아래 표 참조
cgs = gridspec.GridSpec(1,1, left=0.91, right=0.93, top = 0.95, bottom = 0.1)
cax = plt.subplot(cgs[0]) # 컬러바를 할당할 서브플롯인 cax

minr,maxr = -0.3, 1.
labset = ['(a)','(b)','(c)','(d)','(e)','(f)',]
lon2d, lat2d = np.meshgrid(lon, lat)  # np.meshgrid를 이용하여 2차원 위경도 자료 생성
for mm in range(6): # 1월부터 6월까지
    result = ndvi[mm,:,:]
    ax = plt.subplot(gs1[mm], projection = projection_type)
    # 앞서 projection_type이라는 변수로 지정된 도법을 바탕으로 하는 서브플롯을 선언.
    ax.set_extent([minlon,maxlon,minlat, maxlat],
            crs = ccrs.PlateCarree())
    #ccrs.PlateCarree()), 즉 위경도 좌표를 기반으로 해당 서브플롯의 공간범위를 지정.
    cs = ax.pcolormesh(lon2d, lat2d, result, cmap = 'YlGn', transform=ccrs.PlateCarree() ,
                        vmin = minr, vmax = maxr, zorder = 4)
    # 선언된 서브플롯 ax에 자료 공간분포를 음영으로 표현
    # Cartopy를 바탕으로 2차원 위경도 좌표계를 이용하여 공간분포를 그리므로, transform=ccrs.PlateCarree() 옵션을 적용하여야 함.
    # 만약 transform=ccrs.PlateCarree()을 적용하지 않는 경우, 좌표계가 위경도 시스템을 따르지 않아 plt.pcolormesh 결과가 제대로 표출되지 않을 수 있습니다.

    gl = ax.gridlines(crs = ccrs.PlateCarree() ,color = 'k',
         linestyle = ':',xlocs = np.arange(-180,181,30), ylocs = np.arange(30,90,20.),
         zorder = 10, draw_labels = False)
    # 위경도 격자선 세팅.
    ax.coastlines(linewidth = 0.5, zorder=7) # 해안선 표현
    ax.set_title(labset[mm], loc = 'left')
    # 주의) plt.title, plt.ylim, plt.xlim 등의 일부 함수의 경우, 서브플롯에 적용할 때에는 ax.set_title, ax.set_ylim, ax.set_xlim로 바꾸어 적어야 합니다.

clevs = np.linspace(minr,maxr,5)
cbar = plt.colorbar(cs, extend='both', cax=cax, ticks = clevs, orientation = 'vertical')
plt.show()
