# 4.OMPS_pre.py
#import pandas as pd
#from netCDF4 import Dataset
import numpy as np
import os


year='Y2018'
dys=365
# input 및 output 파일 경로는 각자 맞게 지정
idir='./Data_OMPS/'+year+'/'
#output 파일은 격자화된 1년 자료로 만들 예정
ofnam='TCO.'+year[1:5]+'0101_'+year[1:5]+'1231.bin'

f_list=os.listdir(idir)
# 확장자가 txt인 파일만 인식
f_list_txt=[file for file in f_list if file.endswith(".txt")]
buff4=np.empty((180,360),dtype='f4')
odb=np.empty((dys,180,360),dtype='f4')
nfile=len(f_list_txt)
print(nfile)

nff=0
for fn in f_list_txt:
        f=open(idir+fn,'r')
        print(fn)
# 헤더 정보 건너띄기
        data=f.readlines()[3:]
        k=0
        for i in range(14,len(data),15):
                buff="".join(data[i-14:i+1])
                buff1=buff.replace("\n ","").split('l')[0]
                m=0
                for j in range(1,len(buff1)-3,3):
                        buff4[k,m]=float(buff1[j:j+3])
                        m=m+1
                k=k+1
# -180~180 경도 자료를 0~360 경도자료로 변경
        odb[nff,:,0:180]=buff4[:,180:360]
        odb[nff,:,180:360]=buff4[:,0:180]
        nff=nff+1
        f.close()
# 0을 -999로 missing 처리
ix=odb==0
odb[ix]=-999
with open(idir+ofnam,"wb") as of:
        of.write(odb[:,:,:])
