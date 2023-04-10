#!/usr/bin/env python
# coding: utf-8

# -! 라이브러리 및 인터페이스 호출
import iris, iris.plot as iplt
import numpy as np, matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import time, datetime
from datetime import timedelta

# -! 참조파일 이용 강수강도와 색 할당 
rain_rgb_dir = "./data"    # 폴더 지정
rain_rgb_fname = "rain_amount.rgb"     # 파일 지정
precip_levs, red, green, blue = np.loadtxt(rain_rgb_dir+"/"+rain_rgb_fname, skiprows = 1, unpack = True, delimiter = ",")     #  np.loadtxt() 함수 이용  아스키 파일 읽기
red = red[::-1]; green = green[::-1]; blue = blue[::-1]   #  리스트 순서 뒤집기
colors = np.array([red, green, blue]).T   # np.array().T 이용 빨강, 초록, 파랑 순 치환

# -! iris이용 파일 읽기
fname="./data/mslp_rain1h_2020070300_f00.nc"   # 폴더, 파일 지정
mslp_stash = iris.AttributeConstraint(STASH = 'm01s16i222')    # 해면기압의 STASH 코드 지정
precip_stash = iris.AttributeConstraint(STASH = 'm01s04i201')   # 1시간 강수의 STASH 코드 지정
mslp = iris.load_cube(fname, mslp_stash)        # 해면기압, 큐브로 읽기 
acc_precip = iris.load_cube(fname, precip_stash)     # 강수량, 큐브로 읽기

# -! 캡션 추가를 위한 시간 할당
rain_time = acc_precip.coord('time')        # 강수량 큐브 에서 시간 좌표 정보 추출 
print(rain_time)
rain_udate = rain_time.units.num2date(rain_time.bounds)    # 시간 좌표 정보에서 bounds 정보 추출 후 yyyymmddhh형태로 변경
print(rain_udate)

rain_udate = rain_udate[-1]     # 표시할 분석 시간 지정
print(rain_udate)

udate_str = rain_udate[-1].strftime('%Y.%m.%d.%HUTC')    # yyyy.mm.dd.hh UTC 형태 할당 
ch_kdate = rain_udate + timedelta(hours = 9)      # 로컬 시간 계산 : UTC → KST 
kdate_str = ch_kdate[-1].strftime('%Y.%m.%d.%HKST')     # yyyy.mm.dd.hhKST 형태 할당
print(udate_str)
print(ch_kdate)

# -! 그리기
fig = plt.figure(figsize = (11, 10))
mslp_levs = np.arange(mslp.data.min(), mslp.data.max(), 2)     # 등고선 레벨 지정
p_mslp = iplt.contour(mslp, levels = mslp_levs, colors = ('k'), linewidths = (0.7))    # iplt.contour() 함수 이용 해면기압의 등고선 그리기
plt.clabel(p_mslp, fmt = '%4.0f', colors = 'k', fontsize = 8.)    # 등고선 라벨 표시
p_precip = iplt.contourf(acc_precip, levels = precip_levs, colors=colors)  # iplt.contourf() 함수 이용 색채운 강수량 등고선 그리기

plt.gca().coastlines('10m')      # 10m 해상도의 해안선 표시
gl = plt.gca().gridlines()      # 격자선 선언
gx = np.arange(100,180,10)     # 경도축 격자 범위 지정
gy = np.arange(10,90,10)      # 위도축 격자 범위 지정
gl.xlocator = mticker.FixedLocator(gx)    # 경도축 격자선 표시
gl.ylocator = mticker.FixedLocator(gy)    # 위도축 격자선 표시
cbar = plt.colorbar(p_precip, ticks = precip_levs[: len(precip_levs) - 1], pad = 0.03, shrink = 0.8, format = '%.1f')     # 채운 등고선 정보에 의한 컬러바 표시
cbar.ax.tick_params(size = 6.5)     # 컬러바 축 눈금 라벨 크기 지정
cbar.ax.set_ylabel('Precipitation Amount (mm)')   # 컬러바의 y축 제목 지정 및 표시

# -! plt.annotate() 함수 이용 캡션 추가
plt.annotate("MSLP&1H-PRECIP.", (0.01, 1.0362), xycoords = "axes fraction", xytext = (0, -10), textcoords = "offset points", color='g', size = 13.);  
plt.annotate("VALID TIME : " + udate_str + " (+00h FCST)", (0.01, -0.015), xycoords = "axes fraction", xytext = (0, -10), textcoords = "offset points", color = 'red', size = 12.);  # 예측 기간 지정 및 표시
plt.annotate(kdate_str, (0.166, -0.04), xycoords = "axes fraction", xytext = (0, -10), textcoords = "offset points", color = 'black', size = 12.);  # 예측 기간의 로컬 시간 지정 및 표시
plt.annotate("TIME : " + udate_str, (0.75, -0.015), xycoords = "axes fraction", xytext = (0, -10), textcoords = "offset points", color = 'red', size = 12.);   # 분석 시간 지정 및 표시
plt.annotate(kdate_str, (0.834, -0.04), xycoords = "axes fraction", xytext = (0, -10), textcoords = "offset points", color = 'black', size = 12.);    # 분석 시간의 로컬 시간 지정 및 표시 
plt.tight_layout()      # 그림 확대
plt.savefig("./Pics/fig_4_5.png")
plt.show()       # 그림 확인