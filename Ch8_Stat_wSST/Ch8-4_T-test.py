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

# Annual mean
sst_yearly = np.full((nt,ny,nx), np.nan)

k=0
for t in range(0,nt):
  sst_yearly[t,:,:] = np.nanmean(sst[k:k+12,:,:],0)
  k=k+12

sstClm = np.nanmean(sst_yearly[0:30,:,:],0)
sstAno_yearly= sst_yearly-sstClm[None,:,:]


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
vart, varp = time_trend(sstAno_yearly)

vart = vart.reshape((ny,nx))*10
varp = varp.reshape((ny,nx))


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

### Function of trend figure including P-value
def plot_global(i,lon,lat,var,varp,clevs,color,title):
  ax[i].coastlines()
  ax[i].set_extent([0, 359.9, -60, 60])

  ax[i].set_xticks(np.arange(-180,181,60), crs=proj)
  ax[i].xaxis.set_major_formatter(cticker.LongitudeFormatter())
  ax[i].set_yticks(np.arange(-60,61,20), crs=proj)
  ax[i].yaxis.set_major_formatter(cticker.LatitudeFormatter())

  ax[i].tick_params(axis='both',labelsize = 15)
  ax[i].tick_params(axis='y', left = False, which='major', pad=-10)

  cs=ax[i].contourf(lon,lat,var,transform= ccrs.PlateCarree(),
  levels=clevs, colors =color,extend='both')

    ### P-value plot
  plevs = [0, 0.05, 0.1]
  ps=ax[i].contourf(lon,lat,varp,transform = ccrs.PlateCarree(),levels=plevs,hatches=['..',''],alpha=0)
  ax[i].set_title(title, fontsize=16)
  fig.colorbar(cs, ax=ax[i], pad=0.01)

proj = ccrs.PlateCarree(central_longitude=180)
nrows = 3; ncols = 1
fig, ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=(26,15),
subplot_kw={'projection': proj})

clevs = np.arange(-1.,1.+0.01,0.1)

### color set to white
mycolors = plt.cm.seismic(np.linspace(0, 1, len(clevs) + 1))
mycolors[len(clevs) // 2] = [1, 1, 1, 1]
mycolors[len(clevs) // 2 + 1] = [1, 1, 1, 1]

plot_global(0, lon, lat, vart, varp, clevs, mycolors,'SST trend ($^oC$/decade)')
plot_global(1, lon, lat, cor, corp, clevs, mycolors,'Corr[ONI index, SST]')
plot_global(2, lon, lat, reg, regp, clevs, mycolors,'Reg[ONI index, SST]')

#plt.tight_layout()
plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1,right=0.9)
plt.savefig('./Figure/8-4_T-test.png', bbox_inches='tight')
