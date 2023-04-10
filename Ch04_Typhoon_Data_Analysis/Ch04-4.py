#!/usr/bin/env python
# coding: utf-8

# -! 라이브러리 및 인터페이스 호출
import os
import matplotlib.pyplot as plt, matplotlib as mpl
import numpy as np
import iris

# -! 태풍 관측 자료 파일 선언
tynum = "2000"     # 태풍 발생연도, 번호 지정
tyobsf = "./data/ObsTrack.TC"+tynum     # 폴더, 파일 지정

# -! remove_blank5list() 함수 정의: 리스트의 공백 요소 제거
def remove_blank5list(list) :
   newlist = []
   for l in list :
      if not '' == l :
        newlist.append(l)
   return newlist

# -! 태풍 관측 자료 파일 읽기
f = open(tyobsf, 'r')     # 파일 열기
list = f.readlines() # 라인별 오른쪽 공백 제거
list =[l.strip() for l in list]
ty_cat, ty_name, ty_date, ty_lat, ty_lon, ty_pmin, ty_vmax = [], [], [], [], [], [], []   # 리스트 초기화
for i in np.arange(np.shape(list)[0]) :
    if '99999' in list[i] :      # 99999 지표 이용 헤더 구분
      ilist = list[i].rstrip().split(" ")    # 문자열, 공백 구분
      print(ilist)
      header = remove_blank5list(ilist)   # remove_blank5list() 함수 이용 공백 요소 제거
      tyname = header[7]   # 태풍 이름 할당
      nline = int(header[2])   # 관측 데이터 수 할당
      for j in range(nline+1) :    # 관측 데이터 수 반복
         dataline = remove_blank5list(list[i+j].split(" "))    # remove_blank5list() 함수 이용 공백 요소 제거
         if '111' in dataline[1] :    # 111 지표 이용 데이터라인 구분
           ty_date.append(dataline[0][2:])    # 관측 시간
           ty_cat.append(dataline[2])    # 등급
           ty_lat.append(float(dataline[3])*0.1)    #  중심 위도
           ty_lon.append(float(dataline[4])*0.1)    # 중심 경도
           ty_vmax.append(float(dataline[5])*0.5144)     # 최대풍속, knot → m/s
           ty_pmin.append(float(dataline[6]))     # 최소해면기압

# -! 등압면 예측 자료 파일 읽기
udate = "2020070212"   # 분석 시간 지정
tfname = "./data/temp_"+udate+"_f00.nc"    # 폴더, 파일 지정
rfname = "./data/rh_"+udate+"_f00.nc"      # 폴더, 파일 지정
ufname = "./data/xwind_"+udate+"_f00.nc"   # 폴더, 파일 지정
vfname = "./data/ywind_"+udate+"_f00.nc"   # 폴더, 파일 지정

temp = iris.load(tfname)[0]    #  iris.load() 함수 이용 큐브 읽고 온도 할당
rh = iris.load(rfname)[0]      #  iris.load() 함수 이용 큐브 읽고 습도 할당
xwind = iris.load(ufname)[0]   #  iris.load() 함수 이용 큐브 읽고 u벡터 할당
ywind = iris.load(vfname)[0]   #  iris.load() 함수 이용 큐브 읽고 v벡터 할당

print(temp)
print(temp[0])

# 분석 시간의 관측 태풍 중심 위·경도 추출
idx = np.where(np.array(ty_date) == udate[4:])[0][0]
tylat = ty_lat[idx]; tylon = ty_lon[idx]

# -! 해면기압 이용 예측 초기 태풍 중심 격자 위치 찾기
fname="./data/mslp_"+udate+"_f00.nc"       # 폴더, 파일 지정
mslp0_stash = iris.AttributeConstraint(STASH = 'm01s16i222') & iris.Constraint(ForecastPeriod = 0) # iris.AttributeContraint() 함수, STASH코드 이용 변수와 iris.Constraint() 함수 이용 예측 초기 시간 지정
mslp = iris.load_cube(fname, mslp0_stash)    # iris.load_cube() 함수 이용 큐브 읽기
yy, xx = np.where(mslp.data == np.min(mslp.data))    # 최소해면기압의 격자 위치 찾기
nx = xx[0]; ny = yy[0]   # x축, y축의 격자 위치 할당


# -! 예측 초기 태풍 중심의 격자 위치의 위·경도 추출
lon = mslp.coord('longitude').points[:]     # 경도 좌표계에서 데이터 추출
lat = mslp.coord('latitude').points[:]         # 위도 좌표계에서 데이터 추출
print(lon, lat)
lons, lats = np.meshgrid(lon, lat)     # 1차원 위·경도, 2차원 배열 선언
ftylon=lons[yy, xx][0]         # 예측 초기 태풍 중심의 경도 할당

# -! 격자 위치와 수 이용 연직 단면 분석 영역 할당 및 예측 데이터 슬라이싱
dnx = 150
ext_temp = temp[:, ny, nx - dnx : nx + dnx ]
ext_rh = rh[:, ny, nx - dnx : nx + dnx]
ext_xwind = xwind[:, ny, nx - dnx : nx + dnx]
ext_ywind = ywind[:, ny, nx - dnx : nx + dnx]

# -! 상당온위 계산을 위한 기압 배열, 2차원 할당
nz, nx = ext_rh.shape   # 2차원 배열 수 확인
p2d = np.zeros((nz,nx))   # 2차원 배열, 0으로 초기화
ext_press = ext_rh.coord('pressure').points[:]    # 분석영역의 큐브의 기압데이터 추출
for n, p in enumerate(ext_press):
    p2d[n, :]=p   # 경도별 기압값 할당

# -! uv벡터 이용 풍속 계산
u2d = ext_xwind.data; v2d = ext_ywind.data      # uv벡터별 데이터 추출
ws = np.sqrt(u2d ** 2 + v2d ** 2)                         # uv벡터 이용 풍속 계산

# -! cal_specific_humidity() 함수 정의 : 온도, 상대습도, 기압이용 등압면별 비습 계산
def cal_specific_humidity(tempK, Rh, p2d):
    pevaps = 6.11 * np.exp(17.67 * (tempK - 273.15) / (tempK - 29.65)) # 온도 단위(K>C) 변환 후 포화수증기압 계산
    shums = (0.622 * pevaps) / (p2d - pevaps)    # 건조공기에 대한 수증기의 질량 혼합비 계산
    shumi = Rh * shums / 100.     # 상대습도, shums 이용 비습 계산
    return shumi    # 비습 반환

# -! cal_sh_to_td() 함수 정의 : 온도, 비습, 기압 이용 등압면별 노점온도 계산
def cal_sh_to_td(tempK, p2d, shumi):
    pe = shumi * p2d / (0.622 + shumi)    # 비습 이용 포화수증기압 계산
    pe_ezero = pe / 6.112
    pelog = np.log(pe_ezero)
    Td = tempK - (29.65 * pelog - 17.67 * 273.15) / (pelog - 17.67)   # 노점온도 = 온도-건구온도
    return Td    # 노점온도 반환

# -! cal_theta_e() 함수 정의 : 온도, 노점온도 이용 등압면별 상당온위 계산
def cal_theta_e(tempK, dtempC, p2d):
    tempC = tempK - 273.15
    Tdif = tempC - dtempC
    pt = tempK * ((1000. / p2d) ** 0.285857)     # 온위 계산
    evap = 6.11 * np.exp((17.269 * (Tdif + 273.15) - 4717.3) / ((Tdif + 273.15) - 35.86))    # 포화수증기압 계산
    rmix = (0.622 * evap) / (p2d - evap)      # 혼합비 계산
    theta_e = pt * np.exp((2.5 * 10. ** 6 * rmix) / (1004. * tempK))      # 상당온위 계산
    return theta_e       # 상당온위 반환

# -! cal_specific_humidity(), cal_sh_to_td(), cal_theta_e() 함수 이용 상당온위 계산
sh = cal_specific_humidity(ext_temp.data, ext_rh.data, p2d)  # 비습 계산
td = cal_sh_to_td(ext_temp.data, p2d, sh)     # 노점온도 계산
theta_e = cal_theta_e(ext_temp.data, td, p2d)      # 상당온위 계산

# -! 분석 영역의 큐브 좌표계에서 위·경도 추출
ext_lon = ext_rh.coord('longitude').points[:]
ext_lat = ext_rh.coord('latitude').points[:]

# -! 분석 영역의 큐브 좌표계에서 기압, 경도의 값, 이름, 단위 추출
xname = ext_rh.coord('longitude').standard_name     #  x축 이름, 큐브에서 추출
xunit = ext_rh.coord('longitude').units      #  x축 단위, 큐브에서 추출
yname = ext_rh.coord('pressure').long_name     #  y축 이름, 큐브에서 추출
yunit = ext_rh.coord('pressure').units      # y 축 단위, 큐브에서 추출


# -! y축 눈금의 표시 값, 라벨 지정
p_val = [100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 950, 1000]
p_label = [str(p) for p in p_val]

# -! 등고선 간격 지정
cbar_labw = np.arange(0, 100, 5)
cbar_labep = np.arange(320, 400, 5)

# -! reverse_colormap() 함수 정의 : 색 지도(colormap) 순서 뒤집기
def reverse_colormap(cmap, name = 'my_cmap_r'):
    reverse, k = [], []
    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]    #  key별 rgb 데이터 선언
        data = []
        for t in channel:
            data.append((1 - t[0], t[2], t[1]))
        reverse.append(sorted(data))     # sorted() 함수 이용 데이터 정렬
    LinearL = dict(zip(k, reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)    # Lookup표 기반 색 지도(colormap) 형태로 할당
    return my_cmap_r      # 뒤집은 색 지도 반환

# -! reverse_colormap() 함수 이용 색 지도(colormap) 순서 뒤집기
cmaps_r = reverse_colormap(plt.cm.CMRmap)
cmaps = cmaps_r(np.linspace(0, 1, len(cbar_labw)))

# -! 연직 단면도 그리기
fig = plt.figure(figsize=(11,9))
crss_ws = plt.contourf(ext_lon, ext_press, ws, cbar_labw, colors = cmaps)   # 풍속의 채운 등고선 그리기
crss_ep = plt.contour(ext_lon, ext_press, theta_e, cbar_labep, colors = 'k') ##, hold = 'on')    # 상당온위의 등고선 그리기
# 태풍의 관측과 예측 초기의 중심 경도, x축선 표시
plt.axvline(tylon, color = 'k', linestyle = '--', linewidth=2., alpha=0.8)    # 관측 태풍 중심 경도선 표시
plt.axvline(ftylon, color = 'r', linestyle = '-', linewidth=1.5)     # 예측 초기 태풍 중심 경도선 표시
cbar = plt.colorbar(crss_ws, orientation = 'horizontal', aspect = 30, pad = 0.12)   # 컬러바
cbar.set_label("Wind speed (m/s)")       # 컬러바 제목 지정 및 표시
plt.clabel(crss_ep, inline = 1, fmt = '%1.0f')     # 상당온도 등고선 라벨 지정 및 표시
plt.gca().invert_yaxis()      # y축 뒤집기
plt.ylim(1000, 99) ##, 100)      # y축 범위 지정
plt.yticks(p_val, p_label, fontsize = 12.)     # y축 눈금 라벨 지정 및 표시
plt.xlabel(str(xname).title() + " [" + str(xunit) + "]", fontsize = 16)     # x 축 제목 지정 및 표시
plt.xticks(fontsize = 16.)     # x축 눈금 라벨 크기 지정
plt.ylabel(str(yname).title() + " [" + str(yunit) + "]", fontsize = 16)     # y축 제목 지정 및 표시
plt.yticks(fontsize = 11.)     # y축 눈금 라벨 크기 지정
plt.title(udate + "UTC", fontsize=20)     # 그림 제목 지정 및 표시
plt.tight_layout()     # 그림 확대
plt.savefig("./Pics/fig_4_4.png")       # 그림 저장
plt.show()      # 그림 확인
