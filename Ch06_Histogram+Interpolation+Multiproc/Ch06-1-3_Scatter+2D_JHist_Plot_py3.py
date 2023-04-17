"""
Read RMM Index from text file,
and display joint distribution of PC1 and PC2.

1. Scatter + Density Estimation
2. 2D-Joint Histogram (pcolormesh)

By Daeho Jin, 2020.03.04
---

Make compatible to object-oriented concept.

By Daeho Jin, 2023.01.30
"""
import sys
import os.path
import numpy as np
from datetime import date
import Ch06_common_functions as cf

def main():
    ### Parameters
    rmm_fname='./Data/rmm.74toRealtime.txt'
    tgt_dates=(date(2000,11,1),date(2020,3,31))

    ### Read RMM data
    mons,pcs,_= cf.read_rmm_text(rmm_fname,tgt_dates)
    print(pcs.shape) ## Check the dimension

    ### Filtering only for Nov-Mar
    tidx= np.logical_or(mons>=11,mons<=3)
    pcs=pcs[tidx,:]
    print(pcs.shape) ## Check the dimension after filtering

    ### Filtering for both PCs positive
    idx= np.logical_and(pcs[:,0]>0,pcs[:,1]>0)
    pcs=pcs[idx,:]
    print(pcs.shape) ## Check the dimension after filtering

    ### Calculate strength from PCs
    strs= np.sqrt((pcs**2).sum(axis=1))
    print(strs.min(),strs.max()) ## Check the range of strength

    ### Build 2-D joint histogram
    bin_bounds= [0.,0.2,0.4,0.7,1.1,1.6,2.2,3.] ## Non-linear boundaries
    X,Y= np.meshgrid(bin_bounds,bin_bounds) ## Boundary grid prepared for pcolormesh
    H, xedges, yedges = np.histogram2d(pcs[:,0], pcs[:,1], bins=(bin_bounds,bin_bounds))
    H= (H/H.sum()*100.).T  ## Normalized. Now it is in percent(%). Transpose is necessary.
    print(X.shape, H.shape, H.min(), H.max()) ## Check dimension and values

    ### Prepare data for plotting
    outdir= "./Pics/"
    fnout= outdir+"Code6-1-3_Scatter+JHist_example.png"
    suptit= "Scatter and 2D-Joint Histogram Example with MJO RMM Index"
    pic_data= dict(pcs_strs=(pcs,strs), hist2d_data=(X,Y,H),
                   suptit=suptit, fnout=fnout, )
    plot_main(pic_data)
    return

###------------------------------------
### Scatter and 2D-Joint Histogram plot
###------------------------------------
### Plan to draw two axes
### 1. Scatter plot
### 2. 2D-Joint Histogram plot
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as cls
from matplotlib.ticker import MultipleLocator,AutoMinorLocator

def plot_main(pdata):
    X,Y,H= pdata['hist2d_data']
    pcs,strs= pdata['pcs_strs']
    abc, ai= 'abcdefghijklmn', 0
    ncol, nrow= 2, 1
    #nbins= hist_data.shape[1]

    ###---
    fig=plt.figure()
    fig.set_size_inches(8.5,4.2)  ## (xsize,ysize)
    ### Page Title
    fig.suptitle(pdata['suptit'], fontsize=16, y=0.98, va='bottom') #stretch='semi-condensed'

    ### Parameters for subplot area
    left,right,top,bottom= 0.05, 0.95, 0.90, 0.05
    npnx,gapx,npny,gapy= ncol, 0.03, nrow, 0.1
    lx= (right-left-gapx*(npnx-1))/npnx
    ly= (top-bottom-gapy*(npny-1))/npny
    ix,iy= left, top

    ###-- Panel1: Scatter plot
    ax1=fig.add_axes([ix,iy-ly,lx,ly])

    ### Set properties
    #cm=plt.cm.get_cmap('jet'); ### Define ColorMap
    props = dict(edgecolor='none',alpha=0.8,vmin=0.,vmax=3.,cmap='jet')

    ### Draw scatter plot
    pic1=ax1.scatter(pcs[:,0],pcs[:,1],c=strs,s=15,marker='o',**props)

    ### Fine tuning and decorating
    tit= '({}) Positive PC1 vs. Positive PC2'.format(abc[ai]); ai+=1
    ax1.set_title(tit, x=0., ha='left', fontsize=12, stretch='semi-condensed')
    xtloc=np.arange(0,3.1,0.5)
    myscatter_common(ax1,xtloc,xtloc,True)
    ax1.set_xlabel('PC1',fontsize=11,labelpad=1)
    ax1.set_ylabel('PC2',fontsize=11)

    ### Draw Colorbar
    cb= cf.draw_colorbar(fig, ax1, pic1, type='horizontal',size='panel',
                         extend='max', gap=0.14, width=0.03, tick_labelsize=10)
    cb.ax.set_xlabel('MJO Strength',fontsize=11)

    ### Add gaussian density estimation over the scatter plot
    from scipy.stats import gaussian_kde
    k= gaussian_kde([pcs[:,0], pcs[:,1]]) ## Estimate density
    xi,yi= np.mgrid[pcs[:,0].min():pcs[:,0].max():160j,
                    pcs[:,1].min():pcs[:,1].max():160j ] ## New Grid to show density contour
    zi= k(np.vstack([xi.flatten(),yi.flatten()])) ## Calculate Density values for each grid point.

    ### Draw contour plot and insert contour label
    cs= ax1.contour(xi, yi, zi.reshape(xi.shape),
                    levels=5, colors='k', linewidths=1.3 )
    ax1.clabel(cs, inline=True, fontsize=10, fmt='%.1f') #; print(cs.levels)

    ix= ix+lx+gapx
    ###-- Panel2: 2D Joint Histogram plot
    ax2= fig.add_axes([ix,iy-ly,lx,ly])

    ### Set Properties
    newmax= 5 ## Set max percentage value, slightly larger than H.max()
    #cm= plt.cm.get_cmap('viridis',50)
    cm= mpl.colormaps['viridis'].resampled(50)
    cmnew= cm(np.arange(50))
    cmnew= np.concatenate((np.array([1,1,1,1]).reshape([1,-1]),cmnew[1:,:]))  ## Add white at the end
    newcm = cls.LinearSegmentedColormap.from_list("newcm",cmnew)
    props = dict(edgecolor='none', alpha=0.8, vmin=0., vmax=newmax, cmap=newcm)

    ### Draw 2-D joint histogram
    pic2= ax2.pcolormesh(X, Y, H, **props) #; print(H.min(), H.max())

    ### Fine tuning and decorating
    subtit= '({}) Joint Histogram'.format(abc[ai]); ai+=1
    ax2.set_title(subtit, x=0., ha='left', fontsize=12, stretch='semi-condensed')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_xlabel('PC1', fontsize=11, labelpad=1)
    ax2.set_ylabel('PC2', fontsize=11, rotation=270, va='bottom')
    ax2.yaxis.set_ticks_position('both')

    ### Add number to show notable values
    for j in range(H.shape[0]):
        x_center= (X[j,1:]+X[j,:-1])/2. ### Center location of histogram bins
        y_center= (Y[j+1,:-1]+Y[j,:-1])/2.
        cf.write_val(ax2, values=H[j,:], xlocs=x_center, ylocs=y_center,
                     crt=3.95, dformat='{:.1f}%' )

    ### Draw Colorbar
    cb= cf.draw_colorbar(fig, ax2, pic2, type='horizontal', size='panel',
                         extend='max', gap=0.14, width=0.03, tick_labelsize=10)
    cb.ax.set_xlabel('Population Fraction (%)',fontsize=11)

    ###--- Save or Show
    plt.savefig(pdata['fnout'], bbox_inches='tight', dpi=150)
    print(pdata['fnout'])
    #plt.show()
    return

def myscatter_common(ax, xtloc, ytloc, ylidx=True):
    """
    Decorating scatter plot
    Input xtloc, ytloc: major tick location for x and y axis, respectively.
    - tick location values are also used for tick label
    - xlim and ylim are also determined by xtloc and ytloc
    Input ylidx: if False, remove y-tick label.
    """
    ax.set_xticks(xtloc)
    ax.set_yticks(ytloc)
    ax.axis([xtloc[0], xtloc[-1], ytloc[0], ytloc[-1]])
    ax.tick_params(axis='both',which='major',labelsize=10)
    ax.grid(ls=':')
    ax.yaxis.set_ticks_position('both')
    if not ylidx:
        ax.set_yticklabels('')
    return

if __name__=="__main__":
    main()
