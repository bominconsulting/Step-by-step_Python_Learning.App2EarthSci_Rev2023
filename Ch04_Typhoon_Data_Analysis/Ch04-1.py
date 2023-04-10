#!/usr/bin/env python
# coding: utf-8

# -! 라이브러리 및 인터페이스 호출
import os
import numpy as np
import datetime
from datetime import timedelta
from numpy.lib.recfunctions import append_fields
import math, matplotlib.pyplot as plt

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

# -! narr_match_date() 함수 이용
# -! 리스트 ty_date 이용 분석 시작과 마지막 시간의 인덱스값 찾기
print(tysdate)
print(ty_date)
ns = narr_match_date(tysdate, ty_date)
ne = narr_match_date(tyedate, ty_date) + 1
print(ns, ne)

# -! 분석 시작과 마지막 시간의 인덱스 값 이용 태풍 관측 데이터 슬라이싱
print(len(ty_lon))
ty_lon = ty_lon[ns : ne]; ty_lat = ty_lat[ns : ne]; ty_pmin = ty_pmin[ns : ne]
ty_date = ty_date[ns : ne]; ty_vmax = ty_vmax[ns : ne]; ty_cat = ty_cat[ns : ne]
print(len(ty_lon))

a=[ np.array(ty_cat), np.array(ty_date), np.array(ty_lon), np.array(ty_lat), np.array(ty_pmin), np.array(ty_vmax) ]
names='ocat, odate, olon, olat, opmin, ovmax'
formats='U1, U6, f8, f8, f8, f8'
tydata=np.core.records.fromarrays(a, names=names, formats=formats)
print(tydata)

# -! 관측 시간 리스트 요소의 형태 변경 후 할당
tydate_list = []   # 리스트 초기화
for x in ty_date :    # mmddhh 형태 반복
   tydate_list.append(''.join( ['2020', x]))   # join() 함수 이용 yyyymmddhh  형태 변경 후 추가

print(tydate_list)
# -! 태풍 예측 자료에서 읽을 열의 위치, 이름 선언
usecols = (0, 1, 2, 3, 4)   # 읽을 열의 위치 지정
names = ['fcst', 'flon', 'flat', 'fvmax', 'fpmin']    # 읽을 열의 이름 지정

def fdata_append_odata_bydate(date, fdata, tydata) :
     # 예측 시간의 태풍 관측 데이터 추출
    idata = ('XXX','999999', -1., -1., -1., -1.)    # idata, 미씽값으로 초기화
    fdate_list, odata = [], []   # 리스트 초기화
    print(fdata)
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
#print(tydate_list)
ctldata_dict = dict(); expdata_dict = dict()   # 딕셔너리 초기화
for date in tydate_list :    # 관측 시간 리스트 만큼 반복
   ctlf = "./data/CTL_TC"+tynum+"."+date    # ctl실험 파일 지정
   expf = "./data/EXP_TC"+tynum+"."+date   # exp실험 파일 지정
   if os.path.exists(ctlf) :    # ctl 실험 파일이 존재하면 자료처리
     ctldata = np.genfromtxt(ctlf, skip_header = 1, names = names, usecols = usecols)    # np.genfromtxt()함수 이용 구조체 배열로 읽기
     ctldata1 = fdata_append_odata_bydate(date, ctldata, tydata)  # fdata_append_odata_bydate() 함수 이용 예측 데이터와 관측 데이터 합치기
     ctldata_dict[date] = ctldata1   # 딕셔너리에 날짜별 구조체 배열을 할당
   if os.path.exists(expf) : # exp 실험 파일이 존재하면 자료처리
     expdata = np.genfromtxt(expf, skip_header = 1, usecols = usecols, names = names)
     expdata1 = fdata_append_odata_bydate(date, expdata, tydata)
     expdata_dict[date] = expdata1

def distance(origin, destination) :
    lat1, lon1 = origin  # 시작점인 위·경도 선언
    lat2, lon2 = destination   # 끝점인 위·경도 선언
    radius = 6371    # 지구 반지름(km) 할당
    toRad = math.atan(1.) / 45.   # 라디안 단위 할당
    dlat = (lat2-lat1) * toRad     # 위도 차이, 라디안 단위 변환
    dlon = (lon2-lon1) * toRad    # 경도 차이, 라디안 단위 변환
    a = math.sin(dlat / 2.) * math.sin(dlat / 2)  \
    + math.cos(lat1 * toRad) * math.cos(lat2 * toRad) * math.sin(dlon / 2.) * math.sin(dlon / 2.)  # 거리 계산
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1. - a))
    d = radius * c       # 지구 반지름 곱하기
    return d    # 거리 값 반환

# -! cal_tyOmB_bydata() 함수 정의 : 시간순 예측 오차 계산
def cal_tyOmF_bydate(fdata_dict, tydate_list) :
   tyOmF = dict()    # 딕셔너리 초기화
   for date in tydate_list :
      date_list, fcst_list, dis_list, pmin_list, vmax_list =[], [], [], [], []  # 리스트 초기화
      for i in fdata_dict[date] :
         if not 'XXX' in i :    # 미씽값이 아니면 다음을 수행
           dis = distance((i['olat'], i['olon']), (i['flat'], i['flon']))   # 관측과 예측의 태풍 중심 위치 사이의 거리 계산
           pmin = i['opmin'] - i['fpmin']     # 중심기압 오차 계산
           vmax = i['ovmax'] - i['fvmax']    # 최대풍속 오차 계산
         else :    # 미씽값이면
           dis = np.nan; pmin = np.nan; vmax = np.nan    # NaN값 할당
         dis_list.append(dis); pmin_list.append(pmin); vmax_list.append(vmax);
         fcst_list.append(i['fcst']); date_list.append(i['fdate'])
      zipOmF = list(zip(np.array(date_list), np.array(fcst_list), np.array(dis_list), np.array(pmin_list), np.array(vmax_list)))  # 리스트, 배열 변경, 변수별 구분 후 다시 묶기
      dtype =[('date', 'U10'), ('fcst', '<i8'), ('dis', '<f8'), ('pmin', '<f8'), ('vmax', '<f8')]
      OmF = np.array(zipOmF, dtype = dtype)
      tyOmF[date] = OmF
   return tyOmF

# -! cal_tyOmF_bydate() 함수 이용 딕셔너리별 예측 오차 계산하기
ctltyOmF = cal_tyOmF_bydate(ctldata_dict, tydate_list)
exptyOmF = cal_tyOmF_bydate(expdata_dict, tydate_list)
ctltyOmF

# -! chkey_date2fcsthr_tyOmF() 함수 정의 : 오차 값, 예측 기간 순 정렬
def chkey_date2fcsthr_tyOmF(tyOmF, tydate_list) :
   fcsthr = tyOmF[tydate_list[0]]['fcst']     # 딕셔너리 이용 예측 기간 리스트 할당
   ftyOmF = dict()   # 딕셔너리 초기화
   for hr in fcsthr :    # 예측 기간 반복
      hr_date, hr_fcst,  hr_dis,  hr_pmin,  hr_vmax = [], [], [], [], []     # 리스트 초기화
      for date in tydate_list : # 관측 시간 반복
         idx = np.where(tyOmF[date]['fcst'] == hr)[0]  # 예측 기간 기준 인덱스 값 찾기
         if len(idx) >= 1:    # 인덱스 값이 있다면 리스트별 값 추가
           hr_date.append(tyOmF[date][idx]['date'][0])
           hr_fcst.append(tyOmF[date][idx]['fcst'][0])
           hr_dis.append(tyOmF[date][idx]['dis'][0])
           hr_pmin.append(tyOmF[date][idx]['pmin'][0])
           hr_vmax.append(tyOmF[date][idx]['vmax'][0])
      ziphr = list(zip(np.array(hr_date), np.array(hr_fcst), np.array(hr_dis), np.array(hr_pmin), np.array(hr_vmax)))
      dtype = [('date', 'U10'), ('fcst', '<i8'), ('dis', '<f8'), ('pmin', '<f8'), ('vmax', '<f8')]
      OmF = np.array(ziphr, dtype = dtype)
      ftyOmF[str(hr)] = OmF      # 딕셔너리에 예측 기간별 구조체 배열 할당
   return ftyOmF, fcsthr

# -! chkey_date2fcsthr_tyOmF() 함수 이용 예측 기간별 관측 시간순 정렬한 딕셔너리 할당
ctlftyOmF, fcsthr = chkey_date2fcsthr_tyOmF(ctltyOmF, tydate_list)
expftyOmF, fcsthr = chkey_date2fcsthr_tyOmF(exptyOmF, tydate_list)

# -! cal_mean_ftyOmF() 함수 정의 : 예측 기간별 오차 평균
def cal_mean_ftyOmF(ftyOmF, fcsthr) :
   str_fcsthr = map(str, fcsthr)     # map() 함수 이용 예측 기간 리스트, 문자열로 변환
   dis, pmin, vmax = [], [], []         # 리스트 초기화
   for hr in str_fcsthr :
     d = np.nanmean(ftyOmF[hr]['dis'])    # np.nanmean() 함수 이용 평균(NaN값 무시)
     p = np.nanmean(ftyOmF[hr]['pmin'])
     v = np.nanmean(ftyOmF[hr]['vmax'])
     dis.append(d); pmin.append(p); vmax.append(v)
   ziparr = list(zip(np.array(fcsthr), np.array(dis), np.array(pmin), np.array(vmax)))
   dtype = [('fcst', '<i8'),('dis', '<f8'), ('pmin', '<f8'), ('vmax', '<f8')]
   mean = np.array(ziparr, dtype = dtype)
   return mean

# -! cal_mean_ftyOmF() 함수 이용 예측 기간별 분석 기간 평균 오차 계산
ctl=cal_mean_ftyOmF(ctlftyOmF, fcsthr)
exp=cal_mean_ftyOmF(expftyOmF, fcsthr)

print(ctl)

# -! make_fcst_hr_labels () 함수 정의 : 예측 기간 라벨리스트 만들기
def make_fcst_hr_labels(shr, ehr, ihr) :
   fcst_hr = np.arange(shr, ehr, ihr)      # 예측 시작, 끝, 간격 이용 예측 기간 리스트 할당
   fcsthr_labels = []   # 리스트 초기화
   for i in np.arange(len(fcst_hr)) :
      if fcst_hr[i] < 10 :    # 예측 10시간 이하면
        fcsthr_label = "0" + str(fcst_hr[i]) + "h"    # 두 자리 문자열 할당
      else :    # 예측 10시간 이상이면 다음과 같이 할당
        fcsthr_label = str(fcst_hr[i]) + "h"
      fcsthr_labels.append(fcsthr_label)
   return fcst_hr, fcsthr_labels

# -! make_fcst_hr_labels () 함수 이용 예측 기간 라벨 리스트 할당
fcst_hr, xticks_labels = make_fcst_hr_labels(0, 121, 6)

# -! 선언한 x축 중심 기준 막대의 위치 선언
xticks_xpos = np.arange(1, len(fcsthr) + 1, 1)  # 예측 기간 수 만큼 x축 위치 지정
xposa = [x - 0.25 for x in xticks_xpos]

# -! 변수별 정보 선언
varlist = ['dis', 'pmin', 'vmax']
vlabellist = ['Direct position error (km)', 'Central pressure absolute error (hPa)', 'Maximum wind speed absolute error (m/s)']
dylist = [10, 5, 2]
ylist=[255, 8, 15]

# -! 막대 차트 그리기
for iv, var in enumerate(varlist) :
    # np.absolute () 함수 이용 평균오차, 절대값 변경
    ectl = np.absolute(ctl[var])
    eexp = np.absolute(exp[var])
    # min(), max() 함수 이용 절대평균오차의 최소, 최대값 구하기
    dmin = min([np.nanmin(ectl), np.nanmin(eexp)])
    dmax = max([np.nanmax(ectl), np.nanmax(eexp)])
    # 막대 그리기
    fig = plt.figure(figsize=(11,8))
    plt.bar(xposa, ectl, width = 0.25, color = 'skyblue', align = 'edge', label = 'CTL')
    plt.bar(xticks_xpos, eexp, width = 0.25, color = 'magenta', align = 'edge', label = 'EXP')
    # 막대 차트 꾸미기 및 정보 표시
    plt.xlim(0, len(fcsthr) + 1) ##, 1)    # x축 범위 지정
    plt.ylim(0, dmax + dylist[iv]) ##, 10)    # y 축 범위 지정
    plt.xticks(xticks_xpos[::2], xticks_labels[::2], fontsize = 22.)    # x축 눈금 라벨 표시
    plt.yticks(fontsize = 22.)    # y축 눈금 라벨 크기 지정
    plt.grid(True, linestyle = ':')    # 격자선 표시
    plt.xlabel('Lead time (hours)', fontsize=22.)    # x축 제목 지정 및 표시
    plt.ylabel(vlabellist[iv], fontsize=22.)    # y축 제목 지정 및 표시
    plt.title(tyname, fontsize = 25.)    # 그림 제목 지정 및 표시
    plt.legend(loc='best', fontsize = 'x-large')    # 범례 표시
    plt.savefig("./Pics/fig_4_1_"+var+".png")    # 그림 저장
    plt.show()   # 그림 확인
