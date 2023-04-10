#!/usr/bin/env python
# coding: utf-8

# -! 라이브러리 및 인터페이스 호출
import os, sys
import numpy as np
import datetime
from datetime import timedelta
from numpy.lib.recfunctions import append_fields
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# -! 태풍번호, 분석 시간, 태풍 예측 자료에서 읽을 열(Column)의 위치와 이름 선언
tynum = "2000"     # 태풍 발생연도, 번호 지정
date = "2020070100"
usecols = (0, 1, 2, 3, 4)
names = ['fcst', 'flon', 'flat', 'fvmax', 'fpmin']

# -! 태풍 예측 자료 파일 선언 및 읽기
ctlf = "./data/CTL_TC"+tynum+"."+date     # ctl실험 파일 지정
expf = "./data/EXP_TC"+tynum+"."+date    # exp실험 파일 지정
if os.path.exists(ctlf) :      # ctl 실험 파일이 존재하면 자료처리
     ctldata = np.genfromtxt(ctlf, skip_header = 1, usecols = usecols, dtype = None, names = names)    # np.genfromtxt() 함수 이용 구조체 배열로 읽기

if os.path.exists(expf) :    # exp 실험 파일이 존재하면 자료처리
     expdata = np.genfromtxt(expf, skip_header = 1, usecols = usecols, dtype = None, names = names)

ctldata['fcst'][0], ctldata['fcst'][-1]

# -! 태풍 관측 자료 파일 선언
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

# -! cal_date_byfcsthr() 함수 정의 : 예측 기간의 날짜 계산
def cal_date_byfcsthr(date, hr) :
   dt = timedelta(hours=int(hr))    # timedelta 이용 예측 기간 할당
   idate = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:8]), int(date[8:])) + dt # 지정한 시간 기준 시간 계산
   fdate = idate.strftime('%Y%m%d%H')    # 문자열 변환
   return fdate   # 예측 시간 반환

# -! narr_match_date () 함수 정의: 관측 시간기준 예측 시간의 인덱스 값 찾기
def narr_match_date(adate, ty_date) :
   xs = len(ty_date) # 관측 시간의 마지막 인덱스 값 할당
   for i, d in enumerate(ty_date) : # enumerate() 함수 이용 인덱스 값, 요소 값 반복
      if adate[4:] == d :   # 예측 시간과 관측 시간이 일치하면
        xs=i    # 인덱스 값, xs에 할당
   return(xs)   # 인덱스 값 반환

# -! cal_date_byfcsthr() 함수 이용 예측 시작과 마지막 시간 계산 및 할당
tysdate = cal_date_byfcsthr(date, ctldata['fcst'][0]);
tyedate = cal_date_byfcsthr(date, ctldata['fcst'][-1])

# narr_match_date() 함수 이용 관측 시간 기준 예측 시작과 마지막 시간의 인덱스 값 찾기
ns = narr_match_date(tysdate, ty_date)
ne = narr_match_date(tyedate, ty_date) + 1

# 예측 시작과 마지막 날짜의 인덱스 값 이용 태풍 관측 데이터 슬라이싱
ty_lon = ty_lon[ns : ne]; ty_lat = ty_lat[ns : ne]; ty_pmin = ty_pmin[ns : ne]
ty_date = ty_date[ns : ne]; ty_vmax = ty_vmax[ns : ne]; ty_cat = ty_cat[ns : ne]

# -! ext_ms_tycat1() 함수 정의 : 최대풍속 이용 태풍 등급별 마커 할당
def ext_ms_tycat1(vmax) :
   if not vmax == -1 :    #  미씽값이 아니면 다음을 수행
     if vmax >= 33.0 :    #  TY급
        return('s')
     elif vmax < 33.0 and vmax >= 25.0 :    #  STS 급
        return('o')
     elif vmax < 25.0 and vmax >= 17.0 :    #  TS 급
        return('x')
     elif vmax < 17.0 :
        return('*')
     else :
        return('2')
   else :
     return('2')

# -! ext_ms_tycat1() 함수, 최대풍속 이용 ctl 마커 리스트 할당
ms4=[]
for i, ws in enumerate(ctldata['fvmax']) :
   # ext_ms_tycat1() 함수 이용 예측 최대풍속의 태풍 등급별 마커 할당 후 리스트에 추가
   ms4.append(ext_ms_tycat1(ws))

# -! ext_ms_tycat1() 함수, 최대풍속 이용 exp 마커 리스트 할당
ms1=[]
for i, ws in enumerate(expdata['fvmax']) :
   ms1.append(ext_ms_tycat1(ws))

# -! ext_ms_tycat1() 함수, 태풍 등급 이용 관측 마커 리스트 할당
mso=[]
for i, c in enumerate(ty_vmax) :
   mso.append(ext_ms_tycat1(c))

# -! 진로도 그리기
fig = plt.figure(figsize = (8, 10))

# cartopy 지도 그리기
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
ax.set_extent([115, 140, 10, 44]) # west, east, south, north
ax.coastlines(resolution='50m', color='black', linewidth=1)
gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=False)
##gl.xlabels_top = False; gl.ylabels_right=False
gl.top_labels = False; gl.right_labels=False
# 위경도 표시
gl.xlocator = mticker.FixedLocator(np.arange(115, 141, 5))
gl.ylocator = mticker.FixedLocator(np.arange(10, 46, 5))
ax.set_xticks(np.arange(115, 141, 5), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(10, 45, 5), crs=ccrs.PlateCarree())
ax.set_xticklabels(np.arange(115, 141, 5),  fontsize=15.)
ax.set_yticklabels(np.arange(10, 45, 5),  fontsize=15.)
lon_formatter = LongitudeFormatter(number_format='.0f', \
                                       degree_symbol='°', \
                                       dateline_direction_label=True)
lat_formatter = LatitudeFormatter(number_format='.0f', \
                                      degree_symbol='°')
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)


plt.title(tyname + " : " + date, y = 1.03, fontsize = 20.) # 그림 제목 지정 및 표시

# 예측과 관측의 위·경도 선언
lon4 = ctldata['flon']; lat4 = ctldata['flat']
lon1 = expdata['flon']; lat1 = expdata['flat']

# ctl의 진로도 그리기
p0 = ax.plot(lon4, lat4,  c='blue', label = 'CTL', linestyle = 'dotted', linewidth=2., transform=ccrs.PlateCarree())
for xx, yy, ms in zip(lon4, lat4, ms4) : # zip() 함수 이용 같은 위치의 값을 함께 반복
   ax.plot(xx, yy, c = 'blue', marker=ms, markeredgecolor='blue', markersize=4.)

# exp의 진로도 그리기
p1 = ax.plot(lon1, lat1, 'red', label = 'EXP', linestyle = '-', linewidth=2.,  transform=ccrs.PlateCarree())
for xx, yy, ms in zip(lon1, lat1, ms1) :
   ax.plot(xx, yy, c = 'red', marker=ms, markeredgecolor='red', markersize=4.)

# 관측의 진로도 그리기
p2 = ax.plot(ty_lon, ty_lat, c='black', label = 'OBS', linewidth=2., transform=ccrs.PlateCarree())
for xx, yy, ms in zip(ty_lon, ty_lat, mso) :
   ax.plot(xx, yy, c = 'black', marker = ms, markersize = 4.)

#  plot정보로 부터 범례 표기
leg = plt.legend(loc = 'best', fontsize = 'x-large')

# color_legend_text() 함수 정의 : plot 유형별 색과 라벨 색, 동일하게 표기
def color_legend_texts(leg) :
   for line, txt in zip(leg.get_lines(), leg.get_texts()) : # leg정보 이용 plot 유형과 라벨 반복
      txt.set_color(line.get_color()) # plot 유형 색을 가져와 plot 라벨 색 표기


color_legend_texts(leg)    # p0, p1, p2의 유형 색과 라벨 색 같게 표기
plt.savefig("./Pics/fig_4_2.png")   # 그림 저장
plt.show()   # 그림 확인
