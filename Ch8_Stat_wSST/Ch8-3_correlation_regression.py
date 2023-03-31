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

### CPC ONI index
db = open('./Data/oni_index.txt', 'r')
lines = db.readlines()

oni = np.full((len(lines)-1),np.nan)
for i in range(1,len(lines)):
  temp = lines[i].split('\t');
  oni[i-1] = float(temp[7])

### Monthly anomaly
sstClim = np.full((12,ny,nx), np.nan)
for i in range(12):
  sstClim[i,:,:] = np.nanmean(sst[i::12,:,:][:30],0)

sstAno = sst - np.tile(sstClim, (nt, 1, 1))

### JJA mean
sstJJA = np.full((nt,ny,nx), np.nan)
for i in range(0,nt):
  sstJJA[i,:,:] = np.nanmean(sstAno[i*12+5:i*12+8,:,:],0)


import scipy.stats as stats

### Fuction of correlation between 1-dimension variable and 3-dimension variable
def corr_1d3d(var1d, var3d):
  nt,ny,nx = var3d.shape
  cor = np.full((ny,nx),np.nan)
  corp = np.full((ny,nx),np.nan)
  for x in range(nx):
    for y in range(ny):
      bad = ~np.logical_or(np.isnan(var1d),np.isnan(var3d[:,y,x]))
      mod1 = np.compress(bad, var1d)
      mod2 = np.compress(bad, var3d[:,y,x])
      try:
        temp = stats.pearsonr(mod1, mod2)
        cor [y,x] = temp[0]
        corp[y,x] = temp[1]
      except:
        continue
  return cor, corp

### Fuction of regression between 1-dimension variable and 3-dimension variable
def linregress_1d3d(var1d, var3d):
  nt,ny,nx = var3d.shape
  reg = np.full((ny,nx),np.nan)
  regp = np.full((ny,nx),np.nan)
  for x in range(nx):
    for y in range(ny):
      bad = ~np.logical_or(np.isnan(var1d),np.isnan(var3d[:,y,x]))
      mod1 = np.compress(bad, var1d)
      mod2 = np.compress(bad, var3d[:,y,x])
      try:
        reg[y,x], _, _, regp[y,x], _ = stats.linregress(mod1, mod2)
      except:
        continue
  return reg, regp

### Calculate correlation
cor, corp = corr_1d3d(oni, sstJJA)

### Calculate regression
reg, regp = linregress_1d3d(oni, sstJJA)

import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as colors

### figure plot
proj = ccrs.PlateCarree(central_longitude=180)

nrows = 2; ncols = 1
fig, ax = plt.subplots(nrows=nrows,ncols=ncols,figsize=(16,10),subplot_kw={'projection': proj})

clevs = np.arange(-1.,1.+0.01,0.1)

### color set to white
mycolors = plt.cm.seismic(np.linspace(0, 1, len(clevs) + 1))
mycolors[len(clevs) // 2] = [1, 1, 1, 1]
mycolors[len(clevs) // 2 + 1] = colors.to_rgba('white')

### Correlation figure
ax[0].coastlines()
ax[0].set_extent([0, 359.9, -60, 60])
ax[0].set_xticks(np.arange(-180,181,60), crs=proj)

ax[0].xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax[0].set_yticks(np.arange(-60,61,20), crs=proj)
ax[0].yaxis.set_major_formatter(cticker.LatitudeFormatter())
ax[0].tick_params(axis='both',labelsize = 15)
ax[0].tick_params(axis='y', left = False, which='major', pad=-10)

cs=ax[0].contourf(lon,lat,cor,transform= ccrs.PlateCarree(),
levels=clevs,colors=mycolors,extend='both')
ax[0].set_title('(a) Corr[ONI index, SST]', fontsize='15')

### Regression figure
ax[1].coastlines()
ax[1].set_extent([0, 359.9, -60, 60])
ax[1].set_xticks(np.arange(-180,181,60), crs=proj)

ax[1].xaxis.set_major_formatter(cticker.LongitudeFormatter())
ax[1].set_yticks(np.arange(-60,61,20), crs=proj)
ax[1].yaxis.set_major_formatter(cticker.LatitudeFormatter())
ax[1].tick_params(axis='both',labelsize = 15)
ax[1].tick_params(axis='y', left = False, which='major', pad=-10)

cs=ax[1].contourf(lon,lat,reg,transform= ccrs.PlateCarree(),
levels=clevs,colors=mycolors,extend='both')
ax[1].set_title('(b) Reg[ONI index, SST]', fontsize='15')

### Set of figure colorbar
cbar_ax = fig.add_axes([0.2, 0.12, 0.6, 0.02])
cbar=fig.colorbar(cs,cax=cbar_ax,orientation='horizontal')

plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9)
plt.savefig('./Figure/8-3_correlation_regression.png', bbox_inches='tight')
