#!/usr/bin/env python
# coding: utf-8

# -! 라이브러리 및 인터페이스 호출
import os
import numpy as np
import datetime
from datetime import timedelta
from numpy.lib.recfunctions import append_fields
import matplotlib.pyplot as plt

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


del(list)

# -! 분석 시작과 마지막 시간 지정
tysdate = "2020070100"
tyedate = "2020070518"

#  -! narr_match_date() 함수 정의 : 분석 시간과 관측 시간이 일치하는 인덱스 값 찾기
def narr_match_date(adate, ty_date) :
   xs = len(ty_date)    # 관측 시간의 마지막 인덱스 값 할당
   for i, d in enumerate(ty_date) :   # enumerate() 함수 이용 인덱스 값, 요소 값 반복
      if adate[4:] == d :    # 분석 시간과 관측 시간이 일치하면
         xs = i   # 인덱스 값, xs에 할당
   return(xs)   # 인덱스 값 반환

# -! narr_match_date() 함수 이용 분석 시작과 마지막 시간의 인덱스값 찾기
ns = narr_match_date(tysdate, ty_date)
ne = narr_match_date(tyedate, ty_date) + 1

# -! 분석 시작과 마지막 시간의 인덱스 값 이용 태풍 관측 데이터 슬라이싱
ty_lon = ty_lon[ns : ne]; ty_lat = ty_lat[ns : ne]; ty_pmin = ty_pmin[ns : ne]
ty_date = ty_date[ns : ne]; ty_vmax = ty_vmax[ns : ne]; ty_cat = ty_cat[ns : ne]

a=[ty_cat, ty_date, ty_lon, ty_lat, ty_pmin, ty_vmax]
names='ocat, odate, olon, olat, opmin, ovmax'
formats='U1, U6, f8, f8, f8, f8'
tydata=np.rec.fromarrays(a, names=names, formats=formats)
print(tydata)

# -! 관측 시간 리스트 요소의 형태 변경 후 할당
tydate_list = []   # 리스트 초기화
for x in ty_date :    # mmddhh 형태 반복
   tydate_list.append(''.join(['2020', x]))    # join() 함수 이용 yyyymmddhh  형태 변경 후 추가

# -! 태풍 예측 자료에서 읽을 열의 위치, 이름 선언
usecols = (0, 1, 2, 3, 4)   # 읽을 열의 위치 지정
names = ['fcst', 'flon', 'flat', 'fvmax', 'fpmin']    # 읽을 열의 이름 지정

def fdata_append_odata_bydate(date, fdata, tydata) :
     # 예측 시간의 태풍 관측 데이터 추출
    idata = ('XXX','999999', -1., -1., -1., -1.)    # idata, 미씽값으로 초기화
    fdate_list, odata = [], []   # 리스트 초기화
    for i, hr in enumerate(fdata['fcst']) :     #  예측 기간 반복
      dt = timedelta(hours=hr)    # timedelta() 이용 더할 예측 기간 할당
      idate = datetime.datetime(int(date[:4]), int(date[4:6]), int(date[6:8]), int(date[8:])) + dt # datetime.datetime() 함수 이용 문자열을 시간 형태 변경 후 예측 기간 더하기
      fdate = idate.strftime('%Y%m%d%H')    # 계산한 예측 시간, 문자열 변환
      fdate_list.append(fdate)    # 예측 시간 리스트에 추가
      idx = np.where(tydata['odate'] == fdate[4:])[0]    # np.where() 함수 이용 tydata의 관측 시간 기준 인덱스 값 찾기
      if len(idx) == 1 :    # 인덱스 값이 존재한다면
        odata.append(tydata[idx[0]])   # tydata 값 추가
      else :     # 인덱스 값이 없으면
        odata.append(idata)   # 미씽값 (idata) 추가
    fdata = append_fields(fdata, 'fdate', fdate_list, usemask=False)   # 구조체 배열 fdata 에 예측 시간 리스트 추가
    ff = list(zip(*fdata)); oo = list(zip(*odata))    # 변수별 구분 후 zip() 함수 이용 묶기
    for o in oo :    # odata 자료 반복
       ff.append(o)   # fdata에 odata 추가
    fotype=np.dtype({
    'names':['fcst', 'flon', 'flat', 'fvmax', 'fpmin', 'fdate','ocat', 'odate', 'olon', 'olat', 'opmin', 'ovmax'],
    'formats':['f8', 'f8', 'f8', 'f8', 'f8', 'U10', 'U3', 'U6','f8','f8','f8','f8']
})
    # 관측과 예측 데이터의 열별 이름과 자료유형 합치기
    ff = list(zip(*ff))         # 변수별 구분 후 다시 묶기
    fodata = np.array(ff, dtype = fotype)  # 구조체 배열로 할당
    return fodata      # 합친 예측과 관측 데이터의 구조체배열 반환

# -! 태풍 예측 자료 읽기 : 읽은 예측과 관측의 데이터 구조체 배열, 딕셔너리로 할당
ctldata_dict = dict()     # 딕셔너리 초기화
for date in tydate_list :    # 관측 시간 리스트 반복
   ctlf = "./data/CTL_TC"+tynum+"."+date    # ctl실험 파일 선언
   if os.path.exists(ctlf) :    # ctl 실험 파일이 존재하면 자료처리
     ctldata = np.genfromtxt(ctlf, skip_header = 1, usecols = usecols, names = names)    # np.genfromtxt() 함수 이용 구조체 배열로 읽기
     ctldata1 = fdata_append_odata_bydate(date, ctldata, tydata)  # fdata_append_odata_bydate() 함수 이용 예측 데이터와 관측 데이터 합치기
     ctldata_dict[date] = ctldata1   # 딕셔너리에 날짜별 구조체 배열을 할당

# -! 실험명, 변수 라벨 딕셔너리, 그릴 변수 선언
nwpname = "CTL"
varDict={'pmin' : "Min. Sea level Pressure (hPa)", 'vmax' :  "Max. Wind Speed (m/s) "}
var = 'pmin'

# -! x축 간격이 6시간 간격인 x축 배열 선언
n = np.shape(tydata['odate'])[0]
N = range(n)

# x축 수만큼 컬러 선언
color = plt.cm.gist_rainbow(np.linspace(0, 1, n))

# -! 시계열 그리기
plt.figure(1, figsize = (14, 10))

# 관측 시계열 그리기
plt.plot(N, tydata["o"+var], 'k-', label = 'OBS', linewidth = 3.5, alpha = 0.6)

# 관측 시간 기준 예측데이터 지정 및 시계열 그리기
tydatelist = tydate_list         # 날짜 리스트 할당
for i, c in zip (N, color) :
     # 관측 시간의 지정한 변수의 예측 값과 시간 할당
     ctldata = ctldata_dict[tydatelist[i]]['f' + var]
     ctldate = ctldata_dict[tydatelist[i]]['fdate']
     ctldata1 = []  # 리스트 초기화
     for o in tydatelist[i:] :    # i로 관측 시간 조정 후 반복
         idx = np.where(np.array(ctldate) == o)[0]
         if len(idx) >= 1 :    # idx가 1이상이면
            ctldata1.append(ctldata[idx[0]])    # 예측 데이터 추가
         else :    # 아니면 nan값 추가
            ctldata1.append(np.nan)
     # 예측 시계열 그리기
     plt.plot(N[i:], np.array(ctldata1), c = c, linestyle = '-', markeredgecolor = c, label = tydatelist[i], linewidth = 1.5, alpha = 0.8)

# plot정보로 부터 범례 표기
leg = plt.legend(loc = 'best', fontsize = 'small')

# color_legend_texts() 함수 정의 : plot 유형별 색과 라벨 색, 동일하게 표기
def color_legend_texts(leg) :
    for line, txt in zip (leg.get_lines(), leg.get_texts()): # leg 정보 이용 plot 유형과 라벨 할당
       txt.set_color(line.get_color())      # plot 유형의 색을 plot 라벨 색으로 지정

color_legend_texts(leg)     # plot 유형의 라벨과 색, 동일하게 지정

# 시계열 꾸미기 및 정보 표시
plt.xlim(0, n - 1)      # x 축 범위 지정
plt.ylim(np.min(tydata['o' + var]) - 20, np.max(tydata['o' + var]) + 10)     # y축 범위 지정
plt.xticks(N, tydata['odate'], rotation = 'vertical', fontsize = 13.)   # x축 눈금 라벨 지정
plt.yticks(fontsize = 16.)      # y축 눈금 라벨 표시
plt.xlabel('Time (UTC)', fontsize = 16.)     # x축 제목 지정 및 표시
plt.ylabel(varDict[var], fontsize = 16.)      # y축 제목 지정 및 표시
plt.title(tyname.rstrip() + " : " + nwpname + " vs. OBS", fontsize = 20.)    # 그림 제목 표시
plt.savefig("./Pics/fig_4_3.png")      # 그림 저장
plt.show()         # 그림 확인
