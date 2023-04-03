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


### Monthly anomaly
sstClim = np.full((12,ny,nx), np.nan)
for i in range(12):
  sstClim[i,:,:] = np.nanmean(sst[i::12,:,:][:30],0)

sstAno = sst - np.tile(sstClim, (nt, 1, 1))


import scipy.signal as signal

### Detrend
sstAno_ = np.where(np.isnan(sstAno), 0, sstAno)
sstDtr = signal.detrend(sstAno_, axis=0, type='linear', bp=0)
sstDtr = np.where(np.isnan(sstAno),np.nan,sstDtr)


from eofs.standard import Eof

### EOF calculation
### Create an EOF solver to do the EOF analysis.
### Square-root of cosine of latitude weights are applied.
coslat = np.cos(np.deg2rad(lat))
wgts = np.sqrt(coslat)[..., np.newaxis]
solver = Eof(sstDtr, weights=wgts)

### Retrieve the leading EOF, expressed as the correlation between the leading
### PC time series and the input SST anomalies at each grid point, and the leading PC time series itself.

eof = solver.eofs(neofs=3, eofscaling=2) # 2: EOFs are multiplied by the square-root of their eigenvalues
pc  = solver.pcs(npcs=3, pcscaling=1) # 1:  PCs are scaled to unit variance (divided by the square-root of their eigenvalue)

varfrac = solver.varianceFraction()*100
lambdas = solver.eigenvalues()

import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.colors as colors

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

  cs=ax[i].contourf(lon,lat,var,transform=ccrs.PlateCarree(),levels=clevs,cmap=colormap,extend='both')
  ax[i].set_title(title, fontsize=20)
  fig.colorbar(cs, ax=ax[i], pad=0.01)

proj = ccrs.PlateCarree(central_longitude=180)

nrows = 3; ncols = 1
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30,10), subplot_kw={'projection': proj})

# EOF map
for i in range(3):
  clevs = np.arange(-1.,1.+0.01,0.1)
  title = 'EOF'+str(i+1)+' '+str(round(varfrac[i],1))+'%'
  plot_global(i,lon,lat,eof[i,:,:],clevs,'seismic',title)

plt.tight_layout()
plt.savefig('./Figure/8-5_EOF.png', bbox_inches='tight')


# PC time sereis
year = np.arange(1982,2023,5)
year_tick = np.arange(0,len(pc),60)

fig, ax2 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15,10))

for j in range(0,3):
  plt.subplot(nrows,ncols,j+1)
  plt.plot(np.arange(0,len(pc),1),pc[:,j],'b',linewidth=1.5)
  plt.axhline(0,color='k')
  plt.xlabel('Year',fontsize=14)
  plt.ylabel('PC'+str(j+1)+' Amplitude',fontsize=14)
  plt.xticks(year_tick,year,fontsize=12)
  plt.xlim(0,len(pc))
  plt.ylim(np.min(pc.squeeze()), np.max(pc.squeeze()))

plt.tight_layout()
plt.savefig('./Figure/8-5_EOF_ts.png', bbox_inches='tight')
