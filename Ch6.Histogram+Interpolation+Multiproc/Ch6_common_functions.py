"""
A collection of functions to be called by codes in the same directory.

By Daeho Jin, 2023.01.30
"""

import sys
import os.path
import numpy as np
from datetime import date, timedelta
import math

def yield_date_range(start_date, end_date, tdelta=1, include_end_date=True):
    dt=1 if include_end_date else 0
    for n in range(0, int((end_date - start_date).days)+dt, tdelta):
        yield start_date + timedelta(n)


def lon_deg2x(lon, lon0, dlon):
    """
    Transform Lon in degree into array index
      based on pre-known resolution characteristics,
      lon0(starting longitude) and dlon(lon resolution)
    """
    x= math.ceil((lon-lon0)/dlon)
    nx= int(360/dlon)
    if x<0:
        while(x<0):
            x+= nx
    elif x>=nx: x=x%nx
    return x
lat_deg2y = lambda lat,lat0,dlat: math.ceil((lat-lat0)/dlat)

def lon_formatter(x,pos):
    """
    This function can be used with 'matplotlib.ticker.FuncFormatter()'
    to represent longitudes on map.
    """
    if x>0 and x<180:
        return "{:.0f}\u00B0E".format(x)
    elif x>180 and x<360:
        return "{:.0f}\u00B0W".format(360-x)
    elif x>-180 and x<0:
        return "{:.0f}\u00B0W".format(-x)
    else:
        return "{:.0f}\u00B0".format(x)

def bin_file_read2arr(fname, dtype=np.float32):
    """ Open a binary file, read data, and return as numpy 1-D array
        fname : file name
        dtype   : data type; np.float32 or np.float64, etc.
    """
    if not os.path.isfile(fname):
        #print( "File does not exist:"+fname); sys.exit()
        sys.exit("File does not exist:"+fname)

    with open(fname,'rb') as f:
        bin_arr = np.fromfile(file=f, dtype=dtype)
    return bin_arr

def read_rmm_text(fname, date_range=[]):
    """
    Read RMM Index Text file
    fname: include directory
    date_range: start and end dates, including both end dates, optional
    """
    if not os.path.isfile(fname):
        #print( "File does not exist:"+fname); sys.exit()
        sys.exit("File does not exist: "+fname)

    if len(date_range)==0:
        print("For all data records in RMM file")
    elif len(date_range)==2:
        date_txt= [dd.strftime('%Y.%m.%d')
                    for dd in date_range]
        print("From {} to {}".format(*date_txt))
    else:
        print("date_range should be [] or [ini_date,end_date]")
        sys.exit()

    months, pc, phs=[], [], []
    with open(fname,'r') as f:
        for i,line in enumerate(f):
            if i>=2:  ### Skip header (2 lines)
                ww=line.strip().split() #
                onedate=date(*map(int,ww[0:3])) ### "map()": Apply "int()" function to each member of ww[0:3]
                if len(date_range)==0 or (len(date_range)==2
                  and onedate>=date_range[0]
                  and onedate<=date_range[1]):
                    pc.append([float(ww[3]),float(ww[4])]) ### RMM PC1 and PC2
                    phs.append(int(ww[5]))  ### MJO Phase
                    months.append(onedate.month)  ### Save month only

    print("Total RMM data record=",len(phs))
    return np.asarray(months),np.asarray(pc),np.asarray(phs) ### Return as Numpy array

def running_mean_1d(x, N):
    """
    Calculate running mean with "Cumulative Sum" function, asuming no missings.
    Ref: https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    Input x: 1-d numpy array of time series
    Input N: Running Mean period
    Return: Same dimension with x; end points are averaged for less than N values
    """
    cumsum= np.cumsum(np.insert(x, 0, 0))
    new_x= (cumsum[N:] - cumsum[:-N]) / float(N)  ## Now it's running mean of [dim(x)-N] size
    pd0= N//2; pd1= N-1-pd0  ## Padding before and after. If N=5: pd0=2, pd1=2
    head=[]; tail=[]
    for i in range(pd0):
        head.append(x[:i+N-pd0].mean())
        tail.append(x[-i-1-pd1:].mean())
    new_x= np.concatenate((np.asarray(head),new_x,np.asarray(tail)[::-1][:pd1]))
    return new_x

def running_mean_2d(arr, N):
    """
    Calculate running mean with "Cumulative Sum" function, asuming no missings,
       and it's a version of calcuating a set of time series together.
    Ref: https://stackoverflow.com/questions/13728392/moving-average-or-running-mean
    Input arr: 2-d numpy array of time series, [variables, time_series]
    Input N: Running Mean period
    Return: Same dimension with arr; end points are averaged for less than N values
    """
    if len(arr.shape)!=2:
        print("Input variable should be in the form of [variables, time_series].")
        print("Current input data shape = ",arr.shape)
        sys.exit()
    cumsum = np.cumsum(np.pad(arr, ((0,0),(1,0)),'constant', constant_values= 0),axis=1)
    new_arr= (cumsum[:,N:] - cumsum[:,:-N]) / float(N)
    pd0= N//2; pd1= N-1-pd0  ### Padding before and after. N=5: pd0=2, pd1=2
    head=[]; tail=[]
    for i in range(pd0):
        head.append(arr[:,:i+N-pd0].mean(axis=1))
        tail.append(arr[:,-i-1-pd1:].mean(axis=1))
    head, tail= np.asarray(head).T, np.asarray(tail).T
    new_arr= np.concatenate((head,new_arr,tail[:,::-1][:,:pd1]),axis=1)
    return new_arr

def running_mean_2d_general(arr, N, undef=-999., crt=0.5, wt=[]):
    """
    Calculate running mean, a set of time series together.
    It can deal with time series having missings.
    Input arr: 2-d numpy array of time series, [variables, time_series]
    Input N: Running Mean period
    Input undef: lower values than undef will be masked out.
    Input crt: cut-off criterion. If valid ratio < crt: set undef.
    Input wt: Weights for running mean. Ex) 1-2-1 filter: wt=[1,2,1]
            If empty, wt=1 (equal weight).
    Return: Same dimension with arr; undef value represents missing
    ----
    Rev.1: undef can deal with np.nan
    """
    if len(arr.shape)!=2:
        print("Input variable should be in the form of [variables, time_series].")
        print("Current input data shape = ",arr.shape)
        sys.exit()

    if len(wt)==0:
        wt=np.ones([N,],dtype=float)
    else:
        if len(wt)!=N:
            print("Length of wt should be 0 or N.")
            print("Len(wt), N= {}, {}".format(len(wt),N))
            sys.exit()
        else:
            wt=np.asarray(wt)

    pd0= N//2; pd1= N-1-pd0  ### Padding before and after. N=5: pd0=2, pd1=2
    arr_padded= np.pad(arr, ((0,0),(pd0,pd1)), mode='constant', constant_values=undef)

    if np.isnan(undef):  ## In the case of undef==NaN
        miss_idx= np.isnan(arr_padded)
        arr_padded[miss_idx]=0.
        no_miss_idx= np.logical_not(miss_idx)
    else:
        no_miss_idx= arr_padded>undef ## "undef" is supposed to negative value

    wt_arr= np.zeros_like(arr, dtype=float)  ### Track weights, Same dim as arr
    work_arr= np.zeros_like(arr, dtype=float)  ### Work space
    count_arr= np.zeros_like(arr, dtype=int)  ### Count non-missings

    ### Sum for N period
    nt= arr.shape[1]
    for ishift in range(-pd0,pd1+1,1):
        it0, it1= pd0+ishift, nt+pd0+ishift
        work_arr+= arr_padded[:,it0:it1]*no_miss_idx[:,it0:it1]*wt[ishift+pd0]
        wt_arr+= no_miss_idx[:,it0:it1]*wt[ishift+pd0]
        count_arr+= no_miss_idx[:,it0:it1]

    ### Decide missings and average them
    miss_idx= count_arr<crt*N
    work_arr[miss_idx]= undef
    work_arr[~miss_idx]/= wt_arr[~miss_idx]
    return work_arr

def get_lanczos_lp_weights(N):
    '''
    N= low-pass cut-off, unit= number of data points
    co= decide how many data to be used to apply time filter
       (sugeested as 0.66 or 1.09, which makes sum of wgt close to 1.0)
    Return: 1-d numpy array
    '''
    co= 0.66  #1.09
    fc1= 1/N
    nn= int(N*co)+1
    n1= np.arange(-nn,nn+1,1)
    wgt= (np.sinc(2*fc1*n1)*2*fc1)*np.sinc(n1/nn)
    print("Length of weight coefficients, and wgt_sum= {}, {}".format(2*nn+1,wgt.sum()))
    return wgt

def interp2d_fine2coarse_simple(arr, x_scaler=2, y_scaler=2, include_missing=False, undef=-999., crt=0.5):
    """
    Interpolation: Simple Case
    - Fine to Coarse grid (Should be multiple of fine grid size)
    - In the case of missing existing, if ratio of valid grids is less than crt, then it will be set as missing.
    - Input arr: Multi-dimensional numpy array (at least 2D), and last two axes= [y,x]
    - Output: numpy array if no missings or masked array with missings
    """
    arr_shape0= arr.shape
    if len(arr_shape0)<2:
        print("Error: Need at least 2-D array as an input")
        print("Current array shape= ",arr_shape0)
    else:
        ### Make the input array 3-D
        ny,nx= arr_shape0[-2:]
        if len(arr_shape0)>3:
            arr= arr.reshape([-1,ny,nx])
        elif len(arr_shape0)==2:
            arr= arr.reshape([1,ny,nx])
        nz= arr.shape[0]

    if ny%y_scaler!=0 or nx%x_scaler!=0:
        print("Coarse grid size is not a multiple of fine grid size", ny, y_scaler, nx, x_scaler)
        sys.exit()
    else:
        ny2,nx2= ny//y_scaler, nx//x_scaler

    if include_missing:
        new_arr= arr.reshape([nz,ny2,y_scaler,nx2,x_scaler]).swapaxes(2,3).reshape([nz,ny2,nx2,-1])
        if np.isnan(undef):  ## In the case of undef==NaN
            new_arr= np.ma.masked_invalid(new_arr)
        else:
            new_arr= np.ma.masked_less_equal(new_arr,undef)

        new_arr, wsum= np.ma.average(new_arr,axis=3,returned=True)
        ms_idx= wsum < crt*x_scaler*y_scaler  ### Mask out if missings are dominant
        new_arr.mask= ms_idx
    else:
        new_arr= arr.reshape([nz,ny2,y_scaler,nx2,x_scaler]
                    ).mean(axis=(2,4))
    ### Restore input array's dimension
    if len(arr_shape0)==2:
        new_arr= new_arr.squeeze()
    elif len(arr_shape0)>3:
        new_arr= new_arr.reshape([*arr_shape0[:-2],ny2,nx2])
    return new_arr

def bar_x_locator(width, data_dim=[1,10]):
    """
    Depending on width and number of bars,
    return bar location on x axis
    Input width: (0,1) range
    Input data_dim: [# of vars, # of bins]
    Output locs: list of 1-D array(s)
    """
    xx=np.arange(data_dim[1])
    shifter= -width/2*(data_dim[0]-1)
    locs=[]
    for x1 in range(data_dim[0]):
        locs.append(xx+(shifter+width*x1))
    return locs

def write_val(ax, values, xlocs, ylocs, crt=0, ha='center', va='center', dformat='{:.0f}%'):
    """
    Show values on designated location if val>crt.
    Input values, xloc, and yloc should be of same dimension
    """
    ### Show data values
    for val,xl,yl in zip(values,xlocs,ylocs):
        if val>crt: # Write large enough numbers only
            pctxt=dformat.format(val)
            ax.text(xl,yl,pctxt,ha=ha,va=va,stretch='semi-condensed',fontsize=10)
    return

def plot_horizontal_step(ax, xx, yy, label='', props=dict(color='k',)):
    '''
    Draw horizontal step plot
    Input xx: values
    Input yy: location of bin boundaries, dim(yy)= dim(xx)+1
    Input props: line property
    '''
    nn=yy.shape[0]
    for i in range(nn-1):
        ax.plot([xx[i],xx[i]],[yy[i],yy[i+1]],**props)

    for i in range(1,nn-1,1):
        ax.plot([xx[i-1],xx[i]],[yy[i],yy[i]],**props)
    ax.plot([0,xx[0]],[yy[0],yy[0]],**props)
    l1=ax.plot([0,xx[-1]],[yy[-1],yy[-1]],label=label,**props)
    return l1

def draw_colorbar(fig, ax, pic1, type='vertical', size='panel', gap=0.06, width=0.02, extend='neither', tick_labelsize=10):
    '''
    Draw colorbar
    Type: 'horizontal' or 'vertical'
    Size: 'page' or 'panel'
    Gap: gap between panel(ax) and colorbar, ratio to total page size
    Width: how thick the colorbar is, ratio to total page size
    Extend: End style of color bar, 'both', 'min', 'max', 'neither'
    Tick_labelsize: Font size of tick label
    '''
    pos1=ax.get_position().bounds  ##<= (left,bottom,width,height)
    if type.lower()=='vertical' and size.lower()=='page':
        cb_ax =fig.add_axes([pos1[0]+pos1[2]+gap,0.1,width,0.8])  ##<= (left,bottom,width,height)
    elif type.lower()=='vertical' and size.lower()=='panel':
        cb_ax =fig.add_axes([pos1[0]+pos1[2]+gap,pos1[1],width,pos1[3]])  ##<= (left,bottom,width,height)
    elif type.lower()=='horizontal' and size.lower()=='page':
        cb_ax =fig.add_axes([0.1,pos1[1]-gap,0.8,width])  ##<= (left,bottom,width,height)
    elif type.lower()=='horizontal' and size.lower()=='panel':
        cb_ax =fig.add_axes([pos1[0],pos1[1]-gap,pos1[2],width])  ##<= (left,bottom,width,height)
    else:
        print('Error: Options are incorrect:',type,size)
        return

    cbar=fig.colorbar(pic1,cax=cb_ax,extend=extend,orientation=type)  #,ticks=[0.01,0.1,1],format='%.2f')
    cbar.ax.tick_params(labelsize=tick_labelsize)
    return cbar

class Area_weighted_interpolator(object):
    """
    Modules interpolate from fine grid to coarse grid using "Area Weight"

    Parameters
    ----------
    old_lons: 1-d array(len>3) of lons or [nlon, lon0, dlon] of originaal data
    old_lats: 1-d array(len>3) of lats or [nlat, lat0, dlat] of originaal data
    new_lons: 1-d array(len>3) of lons or [nlon, lon0, dlon] of target resolution
    new_lats: 1-d array(len>3) of lats or [nlat, lat0, dlat] of target resolution
    lat_wt:   if apply latitude weight, sin(northen bound)-sin(sourthern bound)
    method: 'average', 'max', or 'min'
    ud:    number indicating missings
    crt:   Minimum ratio, 0 to 1, of valid data to calculate interpolation on a grid cell.

    Attributes
    ----------
    arr2d: Input data, presumably [nlat, nlon]
    arr3d: Input data, presumably [nvar or nt, nlat, nlon]


    Option(s) in interpolate2d()/interpolate3d()
    ----------------------------


    """

    def __init__(self,old_lons=[],old_lats=[],new_lons=[],new_lats=[],
                lat_wt=True,method='average',ud=np.nan,crt=0.5):
        self.old_lons= old_lons
        self.old_lats= old_lats
        self.new_lons= new_lons
        self.new_lats= new_lats
        self.lat_wt= lat_wt
        self.method= method
        self.ud= ud
        self.crt= crt
        return

    def _get_boundaries(self):
        nlen= len(self.old_lons)
        if nlen<3:
            print('Error: old_lons is not given', self.old_lons)
            sys.exit()
        else:
            self.old_lonb= self._calc_bound(self.old_lons)

        nlen= len(self.old_lats)
        if nlen<3:
            print('Error: old_lats is not given', self.old_lats)
            sys.exit()
        else:
            self.old_latb= self._calc_bound(self.old_lats)

        nlen= len(self.new_lons)
        if nlen<3:
            print('Error: new_lons is not given', self.new_lons)
            sys.exit()
        else:
            self.new_lonb= self._calc_bound(self.new_lons)

        nlen= len(self.new_lats)
        if nlen<3:
            print('Error: new_lats is not given', self.new_lats)
            sys.exit()
        else:
            self.new_latb= self._calc_bound(self.new_lats)

    def _calc_bound(self, arr1):
        if len(arr1)==3:
            ni,i0,di= arr1
            bound= np.arange(ni+1)*di-i0-di/2.
        else:
            bound=[]
            for i in range(len(arr1)-1):
                di= arr1[i+1]-arr1[i]
                if i==0:
                    bound.append(arr1[i]-di/2.)
                bound.append(arr1[i]+di/2.)
            bound.append(arr1[-1]+di/2.)
            bound= np.asarray(bound)
        return bound

    def get_weights(self):
        ### Get boundaries of grids
        self._get_boundaries()
        ### X-axis (longitude)
        xinfo= []
        xold_id= 0
        while(self.old_lonb[xold_id] < self.new_lonb[0]):
            xold_id+= 1

        breaker=False
        for iix in range(len(self.new_lonb)-1):
            xold0= xold_id-1
            while(self.old_lonb[xold_id] < self.new_lonb[iix+1]):
                xold_id+=1
                if xold_id>=len(self.old_lonb):
                    breaker=True
                    xold_id=len(self.old_lonb)-1
                    break

            xold1= xold_id
            if xold0<0:
                xold0, alpha= 0, 1.
            else:
                alpha= self._get_ratio(self.old_lonb[xold0], self.new_lonb[iix], self.old_lonb[xold0+1], wh='right')

            beta= self._get_ratio(self.old_lonb[xold1-1], self.new_lonb[iix+1], self.old_lonb[xold1], wh='left')
            xwt= np.ones([xold1-xold0,], dtype=float)
            xwt[0]=alpha; xwt[-1]=beta
            xwtsum= xwt.sum()
            if breaker:
                xwt[-1]= 1.

            xinfo.append([iix,xold0,xold1,xwt/xwtsum])
            if breaker:
                break

        ### Y-axis (latitude)
        yinfo=[]
        yold_id=0
        while(self.old_latb[yold_id] < self.new_latb[0]):
            yold_id+=1

        breaker=False
        for iiy in range(len(self.new_latb)-1):
            yold0= yold_id-1
            while(self.old_latb[yold_id] < self.new_latb[iiy+1]):
                yold_id+= 1
                if yold_id>=len(self.old_latb):
                    breaker= True
                    yold_id= len(self.old_latb)-1
                    break

            yold1= yold_id
            if yold0<0:
                yold0=0; alpha=1.
            else:
                alpha= self._get_ratio(self.old_latb[yold0], self.new_latb[iiy], self.old_latb[yold0+1], wh='right')

            beta= self._get_ratio(self.old_latb[yold1-1], self.new_latb[iiy+1], self.old_latb[yold1], wh='left')

            if self.lat_wt:
                ywt=[]; pi_coef= math.pi/180.
                for yy in range(yold0,yold1):
                    ### Lat_weight= sin(northern bound)-sin(southern bound)
                    ywt.append(math.sin(self.old_latb[yy+1]*pi_coef)-math.sin(self.old_latb[yy]*pi_coef))
                ywt= np.asarray(ywt)
            else:
                ywt= np.ones([yold1-yold0,], dtype=float)
            ywt[0]*= alpha; ywt[-1]*= beta
            ywtsum= ywt.sum()
            if breaker:
                ywt[-1]= ywt[-1]/beta

            yinfo.append([iiy,yold0,yold1,ywt/ywtsum]);
            if breaker:
                break

        self.xinfo, self.yinfo= xinfo, yinfo

        ### Build weight-matrix
        n,m= self.yinfo[-1][0]+1, self.xinfo[-1][0]+1
        wtmtx_arr=[]
        for iy in range(n):
            ywt= self.yinfo[iy][-1]
            wtmtx_arr.append([])
            for ix in range(m):
                xwt= self.xinfo[ix][-1]
                wtmtx= np.outer(ywt,xwt)
                wtmtx_arr[-1].append(wtmtx)
        self.wtmtx_arr= wtmtx_arr
        return

    def _get_ratio(self, x1, y, x2, wh='left'):
        if wh=='left':
            return (y-x1)/(x2-x1)
        elif wh=='right':
            return (x2-y)/(x2-x1)
        else:
            sys.exit('_get_ratio '+wh)

    def interpolate2d(self, arr2d, ud=None, method=None, crt=None):
        if method!=None:
            self.method= method
        if ud!=None:
            self.ud= ud
        if crt!=None:
            self.crt=crt

        if np.isnan(self.ud):
            arr2d= np.ma.masked_invalid(arr2d)
        elif self.ud<0:
            arr2d= np.ma.masked_less(arr2d, self.ud+1)
        else:
            arr2d= np.ma.masked_greater(arr2d,self.ud-1)

        n,m= self.yinfo[-1][0]+1, self.xinfo[-1][0]+1
        outarr= np.empty([n,m], dtype=float)

        if self.method.lower()=='average':
            for iy in range(n):
                yold0, yold1= self.yinfo[iy][1:3]
                for ix in range(m):
                    xold0, xold1= self.xinfo[ix][1:3]
                    indata= arr2d[yold0:yold1, xold0:xold1]
                    wtmtx= self.wtmtx_arr[iy][ix]

                    if (~indata.mask).sum()>0:
                        val, wtsum= np.ma.average(indata,weights=wtmtx,returned=True)
                    else:
                        wtsum=0.

                    if wtsum>=self.crt:
                        outarr[iy,ix]= val
                    else:
                        outarr[iy,ix]= self.ud

        elif self.method.lower()=='max':
            for iy in range(n):
                yold0, yold1= self.yinfo[iy][1:3]
                for ix in range(m):
                    xold0, xold1= self.xinfo[ix][1:3]
                    indata= arr2d[yold0:yold1, xold0:xold1]
                    wtmtx= self.wtmtx_arr[iy][ix]
                    if (~indata.mask).sum()>0 and (wtmtx*indata.mask).sum()<self.crt:
                        outarr[iy,ix]= indata.max()
                    else:
                        outarr[iy,ix]= self.ud

        elif self.method.lower()=='min':
            for iy in range(n):
                yold0, yold1= self.yinfo[iy][1:3]
                for ix in range(m):
                    xold0, xold1= self.xinfo[ix][1:3]
                    indata= arr2d[yold0:yold1, xold0:xold1]
                    wtmtx= self.wtmtx_arr[iy][ix]
                    if (~indata.mask).sum()>0 and (wtmtx*indata.mask).sum()<self.crt:
                        outarr[iy,ix]= indata.min()
                    else:
                        outarr[iy,ix]= self.ud

        else:
            print('{} is not in the supported method list: ["average", "max", "min"]'.format(method))
            sys.exit()

        return outarr

    def interpolate3d(self, arr3d, ud=None, method=None, crt=None):
        if arr3d.ndim==2:
            return self.interpolate2d(arr3d,ud=ud,method=method)
        elif arr3d.ndim==3:
            outarr=[]
            for k in range(arr3d.shape[0]):
                outarr.append(self.interpolate2d(arr3d[k,:], ud=ud, method=method, crt=crt))
            return np.stack(outarr, axis=0)
        else:
            sys.exit("Not supporting more than 3 in dim.")
