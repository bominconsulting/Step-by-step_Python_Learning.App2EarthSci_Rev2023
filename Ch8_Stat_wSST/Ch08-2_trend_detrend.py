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

# Annual mean
sst_yearly = np.full((nt,ny,nx), np.nan)

k=0
for t in range(0,nt):
  sst_yearly[t,:,:] = np.nanmean(sst[k:k+12,:,:],0)
  k=k+12

sstClm = np.nanmean(sst_yearly[0:30,:,:],0)
sstAno= sst_yearly-sstClm[None,:,:]

import scipy.stats as stats
import scipy.signal as signal

# Function of trend calculation
def time_trend(var):
  ntt, nyy, nxx = var.shape
  var = var.reshape(ntt, nyy*nxx)
  vart = np.full((nyy*nxx), np.nan)
  varp = np.full((nyy*nxx), np.nan)
  for i in range(nyy*nxx):
    v = var[:,i]
    mask = ~np.isnan(v)
    v1 = v[mask]
    tt = np.arange(1,len(v1)+1,1)

    if len(v1) == 0:
      vart[i] = np.nan; varp[i] = np.nan
    else:
      vart[i], intercept, r_value, varp[i], std_err = stats.linregress(tt,v1)
  return vart, varp

# Trend per decade
vart, varp = time_trend(sstAno)

vart = vart.reshape((ny,nx))*10
varp = varp.reshape((ny,nx))

# De-Trend
sstAno_ = sstAno.reshape(nt, ny*nx)
sstAno_ = np.where(np.isnan(sstAno_), 0, sstAno_)
sst_dtr = signal.detrend(sstAno_, axis=0, type='linear', bp=0).reshape((nt,ny,nx))
sst_dtr = np.where(sst_dtr==0,np.nan,sst_dtr)

# Trend per decade (after De-Trend)
vart_dtr, varp_dtr = time_trend(sst_dtr)

vart_dtr = vart_dtr.reshape((ny,nx))*10
varp_dtr = varp_dtr.reshape((ny,nx))

import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


# Function of figure plot
def plot_global(i,lon,lat,var,clevs,colormap,title):
  ax[i].coastlines()
  ax[i].set_extent([0, 359.9, -60, 60])

  ax[i].set_xticks(np.arange(-180,181,60), crs=proj)
  ax[i].xaxis.set_major_formatter(cticker.LongitudeFormatter())
  ax[i].set_yticks(np.arange(-60,61,20), crs=proj)
  ax[i].yaxis.set_major_formatter(cticker.LatitudeFormatter())

  ax[i].tick_params(axis='both',labelsize = 15)
  ax[i].tick_params(axis='y', left = False, which='major', pad=-10)

  cs=ax[i].contourf(lon,lat,var,transform= ccrs.PlateCarree(),
  levels=clevs,cmap=colormap,extend='both')
  ax[i].set_title(title, fontsize=20)
  fig.colorbar(cs, ax=ax[i], pad=0.01)

proj = ccrs.PlateCarree(central_longitude=180)

nrows = 2; ncols = 1
fig, ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=(16,10),
subplot_kw={'projection': proj})

# Trend color bar level
clevs1 = np.arange(-1.,1.+0.01,0.1)

# Detrend color bar level
clevs2 = np.arange(-.01,.01+0.0001,0.001)

plot_global(0, lon, lat, vart, clevs1, 'seismic','(a) SST trend ($^oC$/decade)')
plot_global(1, lon, lat, vart_dtr, clevs2, 'seismic', '(b) SST Detrend ($^oC$/decade)')

plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1,right=0.9)
plt.savefig('./Figure/8-2_trend_detrend.png', bbox_inches='tight')
