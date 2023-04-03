import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.gridspec as gridspec

fpath = './Data/ndvi_0.5deg_2015.npy' # 예제 NDVI 공간분포데이터
raw_ndvi = np.load(fpath)

lon = 0.5* np.arange(720) - 180. +0.25 # NDVI 경도정보 만들기
lat = 0.5* np.arange(360) - 90. +0.25  # NDVI 위도정보 만들기
jul = 15. + 15.* np.arange(24) # NDVI 시간정보 (Julian day) 만들기
latid = np.where((lat>=30.) * (lat<=80.)) # 북위 30도~80도 지역 좌표 확인
lat = lat[latid] ## 앞 절에서의 방법과 동일한 방법으로 NDVI를 처리합니다.
yy0,yy1 = latid[0][0], latid[0][-1]
ndvi = raw_ndvi[11, yy0:yy1+1,:]
### 샘플데이터를 위도에 따라 잘라내고, 12번째 시간 자료, 즉 180일경 (6월 말)의 NDVI를 불러옵니다.

idx = np.where((ndvi < -0.3) + (ndvi>1.0)) # 정상범위 이외의 데이터 좌표 파악
ndvi[idx] = np.nan #에러 데이터 예외처리 수행

pft = np.zeros(ndvi.shape)
# NDVI와 동일한 차원을 가지는 pft 행렬을 임의로 생성합니다. 해당 행렬에 각 지역의 위도에 따른 인덱스를 부여하고, 아래에서 상자수염도를 그릴 때에 해당 인덱스를 이용하여 각 위도범위의 자료를 불러오는데에 이용됩니다.
# 이와 같은 방법을 응용하여, 특정 국가나 특정 지면 유형 등을 대상으로 지역별/유형별 상자수염도를 쉽게 그릴 수 있습니다. 여기에서는 편의상 위도 범위를 이용하여 인덱스를 이하 같이 생성합니다.

for ii in range(10): ## 10개 구간을 대상으로
    lat0 = 30. + ii*5.
    lat1 = 30. + (ii+1)*5. # 30도부터 5도 간격으로 구간 설정
    latid = np.where((lat >= lat0) * (lat < lat1))[0]
    yy0, yy1 = latid[0], latid[-1]+1 # 해당구간의 경도 인덱스 파악
    pft[yy0:yy1,:] = ii ### 해당 구간에 ii 값 부여

projection_type = ccrs.PlateCarree(central_longitude=0.)
minlon,maxlon,minlat,maxlat = -180,180.,30.,90. # 공간 범위를 미리 설정.

fig = plt.figure(figsize = (12,4), dpi = 100)
gs0 = gridspec.GridSpec(1,1, left=0.1, right=0.90 , top = 0.95, bottom = 0.5)
ax0 = plt.subplot(gs0[0], projection = projection_type)
#앞서 다룬 gridspec을 이용하여 공간분포용 서브플롯의 위치를 잡아줍니다.
cgs = gridspec.GridSpec(1,1, left=0.91, right=0.93, top = 0.95, bottom = 0.5)
cax0 = plt.subplot(cgs[0]) #컬러바를 위한 서브플롯 선언
gs1 = gridspec.GridSpec(1,1, left=0.1, right=0.90 , top = 0.43, bottom = 0.1)
ax1 = plt.subplot(gs1[0])
#마찬가지로 gridspec을 이용하여 상자수염도를 위한 서브플롯 위치를 잡아줍니다.
minr,maxr = 0., 0.8 #공간분포 그림을 위한 NDVI 최솟값 및 최댓값 설정
lon2d, lat2d = np.meshgrid(lon, lat)

result = ndvi
ax0.set_extent([minlon,maxlon,minlat, maxlat],
            crs = ccrs.PlateCarree()) #ax0에 공간범위를 설정합니다.
cs = ax0.pcolormesh(lon2d, lat2d, result, cmap = 'YlGn', transform=ccrs.PlateCarree() ,
                        vmin = minr, vmax = maxr, zorder = 4) #NDVI 공간분포 그리기
gl = ax0.gridlines(crs = ccrs.PlateCarree() ,color = 'k',
            linestyle = ':',xlocs = np.arange(-180,180.1,30),
            ylocs = np.arange(30., 90.1, 10.),zorder = 7,
            draw_labels = True, x_inline =False, y_inline = False)

gl.top_labels = False
gl.left_labels = True
gl.bottom_labels = True
gl.right_labels = False
gl.rotate_labels = False
#격자선 및 위경도 레이블 세팅

gl.xlabel_style = {'size': 7, 'color': 'k', 'zorder' : 10}
gl.ylabel_style = {'size': 7, 'color': 'k', 'zorder' : 10}

ax0.coastlines(linewidth = 0.5, zorder=7) # 해안선 그리기
ax0.set_title('(a)', loc = 'left') # 그림 번호 라벨링
clevs = np.linspace(minr,maxr,5) # 컬러바를 위한 레벨 생성
cbar = plt.colorbar(cs, extend='both', cax=cax0, ticks = clevs, orientation = 'vertical')
# 미리 선언해두었던 cax0에 컬러바를 그려줍니다.

datadict = {} # 딕셔너리(dictionary) 생성

for ii in range(10): # 인덱스 ii 반복문
    lat0 = 30. + ii*5.
    lat1 = 30. + (ii+1)*5.
    bandname = format(lat0, '2.0f') +'-'+format(lat1, '2.0f')+ '°N' # 각 위도 구간의 이름 생성
    idx = np.where((pft == ii) * ~np.isnan(result))
    # pft가 ii이며 NDVI가 np.nan이 아닌 지점 파악
    datadict[bandname] = result[idx] #datadict 행렬을 대상으로 위도구간의 이름(bandname)에 조건을 만족하는 해당 위도 구간 데이터 (result[idx])를 할당.

ax1.boxplot(list(datadict.values()), showfliers=False, whis = [10,90],                    medianprops={'color':'k'}, showmeans =True, meanprops={'markerfacecolor':'k', 'markeredgecolor':'None','marker':'o'})
# 상세 설명 이하 참조.

ax1.set_xticklabels(list(datadict.keys()))
# datadict.keys(), 즉 datadict에 입력되었던 key 값을 이용하여 각 위도구간의 이름을 x축 tick에 달아줍니다.
ax1.set_title('(b)', loc = 'left') # 그림 번호 라벨링
plt.show()
