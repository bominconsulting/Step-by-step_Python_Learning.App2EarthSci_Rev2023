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
  intr = np.full((nyy*nxx), np.nan)
  varp = np.full((nyy*nxx), np.nan)
  for i in range(nyy*nxx):
    v = var[:,i]
    mask = ~np.isnan(v)
    v1 = v[mask]
    tt = np.arange(1,len(v1)+1,1)

    if len(v1) == 0:
      vart[i] = np.nan; varp[i] = np.nan
    else:
      vart[i], intr[i], r_value, varp[i], std_err = stats.linregress(tt,v1)
  return vart, intr, varp

# Trend per decade
vart, intr, varp = time_trend(sstAno)

vart = vart.reshape((ny,nx))
intr = intr.reshape((ny,nx))
varp = varp.reshape((ny,nx))

### This is for the regression over time (i.e., trend)
def get_new_dof_one_tseries(ts1):
    r= np.corrcoef(ts1[:-1],ts1[1:])[1,0]
    N= len(ts1)
    Neff= N*(1-r)/(1+r) if r>0 else N
    return Neff

### Calculation regression p-value
def get_pval_regr_slope(x,y,slope,intercept,Neff=None):
    if Neff==None:
        Neff=len(y)
    var_residual=np.sum((y-slope*x-intercept)**2,axis=0)/(Neff-2)
    t=slope/np.sqrt(var_residual/np.sum((x - np.mean(x))**2))

### two-tailed, 1-p_val
    sf_level=1-stats.t.sf(np.absolute(t),df=Neff-2)*2

    p_val= 1-sf_level
    if t<0: sf_level*=-1
    return p_val, sf_level

Neff = np.full((ny,nx),np.nan)
for j in range(nx):
  for i in range(ny):
    Neff[i,j] = get_new_dof_one_tseries(sstAno[:,i,j])

pval_mod = np.full((ny,nx), np.nan)
for i in range(nx):
  for j in range(ny):
    y = sstAno[:,j,i]
    x = np.arange(1,len(y)+1,1)
    pval_mod[j,i], _ = get_pval_regr_slope(x,y,vart[j,i],intr[j,i],Neff=Neff[j,i])


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

  cs=ax[i].contourf(lon,lat,var,transform= ccrs.PlateCarree(),levels=clevs,colors=color,extend='both')

  ### P-value plot
  plevs = [0, 0.05, 0.1]
  ps=ax[i].contourf(lon,lat,varp,transform = ccrs.PlateCarree(),levels=plevs,hatches=['..',''],alpha=0)
  ax[i].set_title(title, fontsize=20)
  fig.colorbar(cs, ax=ax[i], pad=0.01)

proj = ccrs.PlateCarree(central_longitude=180)
nrows = 2; ncols = 1
fig, ax = plt.subplots(nrows=nrows, ncols=ncols,figsize=(26,10),subplot_kw={'projection': proj})

clevs = np.arange(-1.,1.+0.01,0.1)

### color set to white
mycolors = plt.cm.seismic(np.linspace(0, 1, len(clevs) + 1))
mycolors[len(clevs) // 2] = [1, 1, 1, 1]
mycolors[len(clevs) // 2 + 1] = [1, 1, 1, 1]

plot_global(0,lon,lat,vart*10,varp,clevs,mycolors,'Original p-value of SST trend ($^oC$/decade)')
plot_global(1,lon,lat,vart*10,pval_mod,clevs,mycolors,'Modified p-value of SST trend ($^oC$/decade)')

plt.subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9)
plt.savefig('./Figure/8-4_T-test_effective_freedom_trend.png', bbox_inches='tight')
