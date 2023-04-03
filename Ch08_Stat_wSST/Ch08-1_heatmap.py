### Load Package
from netCDF4 import Dataset
import numpy as np
import pandas as pd

### Load SST data
fsst = Dataset('./Data/sst.mnmean.nc')
lon = fsst.variables['lon'][:]
lat = fsst.variables['lat'][:]
sst_ = fsst.variables['sst'][:]
time_ = np.array(pd.date_range('1982-01', '2023-03', freq = 'M').strftime('%Y%m'))

time = time_[0:np.where(time_=='202212')[0][0]+1]
sst = sst_[0:np.where(time_=='202212')[0][0]+1]


### Replace SST value
sst = np.where(sst<=0,np.nan, sst)

### Check the dimension size
nt = 2022-1982+1; _, ny, nx = sst.shape

### Land mask(land=0, sea=1)
flm = Dataset('./Data/lsmask.nc')
lsmask = flm['mask'][0,:,:]
lsm = np.stack([lsmask]*(nt*12))
sst[lsm==0] = np.nan


# SST monthly climatology
sstClim = np.full((12,ny,nx), np.nan)
for i in range(12):
  sstClim[i,:,:] = np.nanmean(sst[i::12,:,:][:30],0)

# SST monthly anomaly
sstAno = sst.copy()
for i in range(12):
  sstAno[i::12,:,:] = sst[i::12,:,:]-sstClim[i,:,:]

sstAno_mean = np.nanmean(sstAno,(1,2))

import scipy.stats as stats

sstRank = sstAno_mean.copy()
for i in range(12):
  sstRank[i::12] = stats.rankdata(sstAno_mean[i::12]*(-1))

sstRank=sstRank.astype(int)

rankIdx= sstRank>3
sstRank= sstRank.astype(str)
sstRank[rankIdx]= ''
sstRank_re = sstRank.reshape([nt,12]).swapaxes(0,1)


#%% pandas
mon_name = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
#year = [time[i][0:4] for i in range(nt*12)]
#mon = [time[i][4:6] for i in range(nt*12)]
year = [t[0:4] for t in time]
mon = [t[4:6] for t in time]

data = pd.DataFrame({'SST' : sstAno_mean})
data.insert(0,'month',mon)
data.insert(0,'year',year)

df = data.pivot('month','year','SST')
df.index = mon_name
df.columns.name = None

import seaborn as sns
import matplotlib.pyplot as plt

#%% regrid figure
year = np.arange(1982,2023,5)
year_tick = np.arange(0.5,nt,5)

plt.figure(figsize=(12, 4))
sns.heatmap(df, annot = sstRank_re, fmt="", center=0, square = True, vmin = -.4, vmax = .4, cmap='RdBu_r', cbar_kws={"shrink": 0.7})

plt.xticks(year_tick,year,fontsize = 15,rotation=0)
plt.yticks(rotation=0, fontsize = 15)
plt.ylabel('')

plt.tight_layout()
plt.title('Global SST anomaly heatmap ($^o$C)', fontsize=20, fontweight = 'bold', pad = 10)
plt.savefig('./Figure/8-1_heatmap.png', dpi = 400)
