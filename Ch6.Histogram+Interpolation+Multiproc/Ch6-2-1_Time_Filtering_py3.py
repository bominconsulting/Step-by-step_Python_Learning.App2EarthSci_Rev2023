"""
Read RMM Index from text file,
and perform running mean and Lanczos filtering

1. Running Mean, case of no missings
2. Running Mean with missings
 - Lanczos filtering is a special case of running mean with specific weights.
 - cf. running mean uses equal weights.

By Daeho Jin, 2020.03.04
---

Make compatible to object-oriented concept.

By Daeho Jin, 2023.01.30
"""
import sys
import os.path
import numpy as np
from datetime import date
import Code_6_common_functions as cf

def main_test():
    ###-------------------------
    ### Test with simple example
    ###-------------------------
    a= (np.arange(6)-2).reshape([2,3]).swapaxes(0,1).reshape(-1)
    print("Input =\n",a)
    N=3
    print("Method1 =\n",cf.running_mean_1d(a,N))
    print("Method2 =\n",
          cf.running_mean_2d(a.reshape([1,-1]),N).squeeze())
    print("Method3 =\n",
          cf.running_mean_2d_general(
          a.reshape([1,-1]),N).squeeze())

    wt=[1,2,1]
    print("Method3 with weight {} =\n".format(wt),
          cf.running_mean_2d_general(
          a.reshape([1,-1]),N,wt=wt)[0,:])
    print("\n")
    return

def main():
    ###--------------------
    ### Test with RMM Index
    ###--------------------
    ### Parameters
    rmm_fname='./Data/rmm.74toRealtime.txt'
    tgt_dates=(date(2000,11,1),date(2001,3,31))
    N= 21  ## Period for running mean (days)

    ### Read RMM data
    _,pcs,phs= cf.read_rmm_text(rmm_fname,tgt_dates)
    print(pcs.shape) ### Check the dimension

    ### Calculate strength from PCs
    strs= np.sqrt((pcs**2).sum(axis=1))
    print(strs.min(),strs.max()) ### Check the range of strength
    ### PC1, PC2, and Strength time-series together
    data= np.concatenate((pcs,strs.reshape([-1,1])),axis=1) ## Now shape=[days, 3 vars]
    data= data.T ### Now shape=[3 vars, days]
    print(data.shape)

    ### Running Mean
    rm_data= cf.running_mean_2d(data,N)

    ### In order to test running mean with time-series having missings,
    ### artificially adding missings at random locations
    idx= np.arange(data.shape[1])
    RS= np.random.RandomState(seed=1234)  ## Set seed in order to obtain the same result every time.
    RS.shuffle(idx)
    data_ms= np.copy(data)
    data_ms[:,idx[:30]]=-999.9  ## Values at random locations are replaced by -999.9

    ### Lanczos Filter
    lz_wgt= cf.get_lanczos_lp_weights(N)
    lz_data= cf.running_mean_2d_general(data_ms, len(lz_wgt), undef=-999.9, wt=lz_wgt)
    ### The results are changed to "masked array" in order to display properly
    data_ms= np.ma.masked_less(data_ms,-999.)
    lz_data= np.ma.masked_less(lz_data,-999.)

    ### Produce a list of time info
    times= list(cf.yield_date_range(*tgt_dates))

    ### Prepare data for plotting
    outdir= "./Pics/"
    fnout= outdir+"Code6-2-1_Time_Filter_example.png"
    suptit= "Time Filter Example with MJO RMM Index"
    var_names= ['PC1','PC2','Str']
    data_set= [
         ('{}-day Running Mean'.format(N),data,rm_data),
         ('{}-day Lanczos Filtered'.format(N),data_ms,lz_data)
    ]
    pic_data= dict(data_set= data_set,
                   xt= times, var_names= var_names,
                   suptit=suptit, fnout=fnout,
    )
    plot_main(pic_data)
    return

###--------------------
### Display time series
###--------------------
### Plan to draw six panels
### 1. Time-series and running mean (left)
### 2. Time-series with some missings and Lanczos filtering (right)
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.dates import DateFormatter, MonthLocator

def plot_main(pdata):
    data_set= pdata['data_set']
    xt= pdata['xt']
    var_names= pdata['var_names']
    abc, ai= 'abcdefghijklmn', 0
    ncol, nrow= len(data_set), len(var_names)

    ###---
    fig=plt.figure()
    fig.set_size_inches(8.5,6)  ## (xsize,ysize)
    ### Page Title
    fig.suptitle(pdata['suptit'], fontsize=16, y=0.975, va='bottom') #stretch='semi-condensed'

    ### Parameters for subplot area
    left,right,top,bottom= 0.05, 0.95, 0.92, 0.05
    npnx,gapx,npny,gapy= ncol, 0.06, nrow, 0.11
    lx= (right-left-gapx*(npnx-1))/npnx
    ly= (top-bottom-gapy*(npny-1))/npny
    ix,iy= left, top

    ###---
    cc= ['orange', 'RoyalBlue']
    for i, (tit, data1, data2) in enumerate(data_set):
        for k, vn in enumerate(var_names): ## Each panel for one time-series
            ax1=fig.add_axes([ix, iy-ly, lx, ly])
            ax1.plot(xt, data1[k,:], c= '0.4')  ### Original time-series
            ax1.plot(xt, data2[k,:], c= cc[i])  ### After running mean
            subtit='({}) {} {}'.format(abc[ai],vn,tit); ai+=1
            plot_common(ax1,subtit)
            iy=iy-ly-gapy
        iy= top; ix+= lx+gapx
    ### As an example to change time format...
    xt_loc= [dd for dd in xt if dd.day==15]  ## Select only if day==15
    ax1.set_xticks(xt_loc)
    #ax1.xaxis.set_major_locator(MonthLocator(bymonthday=15))  ## Same effect
    ax1.xaxis.set_major_formatter(DateFormatter("%d%b\n%Y")) ## Change time format

    ###--- Save or Show
    plt.savefig(pdata['fnout'], bbox_inches='tight', dpi=150)
    print(pdata['fnout'])
    #plt.show()
    return

def plot_common(ax,subtit):
    """
    Decorating time-series plot
    """
    ### Title
    ax.set_title(subtit,fontsize=13, ha='left', x=0.0)

    ### Ticks and Grid
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ymin,ymax= ax.get_ylim()
    if ymin*ymax<0:  ### if y_range include 0, draw a horizontal line
        ym=max(-ymin,ymax)
        ax.set_ylim(-ym,ym)
        ax.axhline(y=0.,ls='--',c='0.3',lw=1)
    else:
        ax.set_ylim(0,ymax)
    ax.grid(axis='y',color='0.7', linestyle=':', linewidth=1)
    ax.yaxis.set_ticks_position('both')
    ax.tick_params(axis='both',which='major',labelsize=10)
    return

if __name__=="__main__":
    #main_test()
    main()
