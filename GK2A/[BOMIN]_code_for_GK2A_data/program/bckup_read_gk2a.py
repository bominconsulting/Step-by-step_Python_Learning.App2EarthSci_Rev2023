# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 22:42:25 2023

@author: ryujih
"""

import h5py
import numpy as np
import pandas as pd

# 파일 경로
path='../data/'
fname='gk2a_ami_le1b_ir105_fd020ge_202307170000.nc'
filename=path+fname


# 1. 엑셀파일의 Look-up table 읽고 int16 값을 Brightness temperatrue로 치환
with h5py.File(filename, 'r') as f:
	var  = '/image_pixel_values'           # Variable
	longitude  = '/dim_image_x'            # Longitude
	latitude   = '/dim_image_y'            # Latitude
	data = f[var][:]                       # Read variable(Near-surface rain rate)
	lon  = f[longitude][:]                 # Read longitude
	lat  = f[latitude][:]                  # Read latitude
    
# 엑셀파일 읽어오기
convert_table='20191115_gk-2a ami calibration table_v3.1_ir133_srf_shift.xlsx'
df = pd.read_excel(convert_table, sheet_name='Calibration Table_WN')

# 채널정보 확인
chan_info=fname[14:19] # 파일명에서의 채널정보 텍스트 위치

# 파일명에 사용되는 GK2A의 채널명들
chan_names=['vi004','vi005','vi006','vi008','nr013','nr016','sw038','wv063',
            'wv069','wv073','ir087','ir096','ir105','ir112','ir123','ir133']


#  엑셀파일로부터 index 산출 =====
# 파일명에서 얻은 채널정보를 바탕으로 몇번째 채널인지 index 추출
chan_num = chan_names.index(chan_info)
# 엑셀파일에서 각 채널별 column의 첫번째 index
col_idx=[1,3,5,7,9,11,13,16,19,22,25,28,31,34,37,40] #column index
chan_col=col_idx[chan_num] # 해당 채널은 엑셀파일에서 몇번째 column에 있는지 찾기
# ============================== 


nan=np.where(data == 32768) # data에서 nan 값 위치 index 
tr=np.where(data!=32768) # data에서 nan이 아닌 값 위치 index

bt=np.zeros((np.shape(data)[0],np.shape(data)[1]))

bt[nan]=np.nan # bt 배열에 nan값 덮기

nvalue=np.size(tr[0]) # nan이 아닌 값의 개수

# int16 값을 밝기온도로 변환
for i in np.arange(nvalue):
    x=tr[0][i]; y=tr[1][i]
    bt[x,y] = df.values[data[x,y]][chan_col+2]


# 2. 좌표파일 읽어오기
fname_lon='Lon_2km.bin'
with open(fname_lon,'r') as f:
    lon_a=np.fromfile(f,dtype=np.float32)
lon=lon_a.reshape(5500,5500)

fname_lat='Lat_2km.bin'
with open(fname_lat,'r') as f:
    lat_a=np.fromfile(f,dtype=np.float32)
lat=lat_a.reshape(5500,5500)


    
    
    