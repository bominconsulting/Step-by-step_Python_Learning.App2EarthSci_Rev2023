"""
Read RMM Index from text file,
and display MJO strength distribution by MJO Phases.

1. Horizontal Bar Plot
2. Corresponding Step Plot (Lined Bar)
3. Box + Violin Plot
4. Violin plot, half and half

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
    rmm_fname= './Data/rmm.74toRealtime.txt'
    tgt_dates= (date(2000,11,1),date(2020,3,31))
    ph2draw=[3,6] ## Display MJO Phases 3 and 6

    ### Read RMM data
    mons,pcs,phs= cf.read_rmm_text(rmm_fname,tgt_dates)
    print(pcs.shape) ### Check the dimension

    ### Filtering only for Nov-Mar
    tidx= np.logical_or(mons>=11,mons<=3)
    pcs= pcs[tidx,:]
    phs= phs[tidx]
    print(pcs.shape) ### Check the dimension after filtering

    ### Calculate strength from PCs
    strs= np.sqrt(pcs[:,0]**2+pcs[:,1]**2)
    print(strs.min(),strs.max()) ### Check the range of strength if any weird value exists

    ### Build histogram of strength by MJO phase
    bin_bounds=np.arange(0.,4.,0.5) ### Produce 7 histogram bins
    bin_bounds[-1]=9.9 ### Extend to extremely large values
    hists=[]
    for ph in range(1,9,1): ### For each MJO phase
        phidx= phs==ph
        hist= np.histogram(strs[phidx],bin_bounds)[0]
        hists.append(hist)
    hists=np.asarray(hists)
    print(hists.shape) ### Check the dimension of histogram results
    hists=hists/phs.shape[0]*100. ### Normalized by total population, now in percent

    ### Prepare data for plotting
    outdir= "./Pics/"
    fnout= outdir+"Code6-1-2_barh+Violin_example.png"
    suptit="Horizontal Bar Plot Example with MJO RMM Index"
    pic_data= dict(hist_data=hists, phs_strs=(phs,strs),
                   ph2draw=ph2draw, bin_bounds=bin_bounds,
                   suptit=suptit, fnout=fnout,
    )
    plot_main(pic_data)
    return

###---------
### Bar plot
###---------
### Plan to draw four axes
### 1. Horizontal Bar plot
### 2. Lined Bar(Step) plot
### 3. Box plot over violin plot
### 4. Violin plot, half and half
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,AutoMinorLocator,FormatStrFormatter
from itertools import repeat

def plot_main(pdata):
    hist_data= pdata['hist_data']
    phs, strs= pdata['phs_strs']
    bin_bounds= pdata['bin_bounds']
    ph2draw= pdata['ph2draw']
    abc, ai= 'abcdefghijklmn', 0
    ncol, nrow= 4, 1
    nbins= hist_data.shape[1]

    ###---
    fig= plt.figure()
    fig.set_size_inches(8.5,6)  ## (xsize,ysize)
    ### Page Title
    fig.suptitle(pdata['suptit'], fontsize=16, y=0.975, va='bottom') #stretch='semi-condensed'

    ### Parameters for subplot area
    left,right,top,bottom= 0.05, 0.95, 0.92, 0.05
    npnx,gapx,npny,gapy= ncol, 0.03, nrow, 0.1
    lx= (right-left-gapx*(npnx-1))/npnx
    ly= (top-bottom-gapy*(npny-1))/npny
    ix,iy= left, top

    ###-- Panel1: Compare strength distribution of phase 3 and 6
    ax1=fig.add_axes([ix,iy-ly,lx,ly])

    ### Parameters for this panel
    nvars= len(ph2draw)
    cc=['DarkBlue','IndianRed'] ## Preset colors for each bars

    h_gap= 0.3
    wd= (1-h_gap)/nvars ## Width of bar
    ylocs= cf.bar_x_locator(wd, data_dim=[nvars, nbins])

    ### Draw Bar Plot
    for i,ph in enumerate(ph2draw):
        pbar= ax1.barh(ylocs[i], hist_data[ph-1,:], height=wd,
                       color=cc[i], alpha=0.7,
                       label='Phase{}'.format(ph)
        )
        cf.write_val(ax1, values=hist_data[ph-1,:],
                 xlocs=repeat(0.1), ylocs=ylocs[i],
                 crt=1.45, ha='left', dformat='{:.1f}'
    )  ## itertools.repeat: make constant iterable

    ### Fine tuning and decorating
    x_range=[0,6]
    subtit='({}) Bar_H'.format(abc[ai]); ai+=1
    barh_common(ax1, subtit, y_dim=nbins,
                yt_labs=bin_bounds, x_range=x_range )
    ax1.set_xlabel('Percent', fontsize=11)
    ax1.set_ylabel('MJO Strength', fontsize=11) #,labelpad=0)
    ax1.legend(loc='upper right', bbox_to_anchor=(0.99,0.995),
               fontsize=10, framealpha=0.75 )
    ix=ix+lx+gapx

    ###-- Panel2: Lined Bar(Step) plot
    ax2=fig.add_axes([ix,iy-ly,lx,ly])

    ### Parameters for this panel
    yy=np.arange(len(bin_bounds))-0.5

    ### Draw lined bar plot
    for i,ph in enumerate(ph2draw):
        data1= np.pad(hist_data[ph-1,:], (1,1),
                      mode='constant', constant_values=0
        )  ## Add zero values at both sides
        pbar2= cf.plot_horizontal_step(ax2, xx=hist_data[ph-1],
                                yy=yy,label='Phase{}'.format(ph),
                                props=dict(color=cc[i],lw=2,alpha=0.9)
        )

    ### Fine tuning and decorating
    subtit='({}) Step_H'.format(abc[ai]); ai+=1
    barh_common(ax2, subtit, y_dim=nbins, yt_labs=[], x_range=x_range)
    ax2.set_xlabel('Percent', fontsize=11)
    #ax2.set_ylabel('MJO Strength', fontsize=12) # Not enough room for ylabel
    ax2.legend(bbox_to_anchor=(.99,0.995), loc='upper right',
               fontsize=10, framealpha=0.75 )
    ix=ix+lx+gapx

    ###--- Panel3: Box plot over Violin plot
    ax3=fig.add_axes([ix,iy-ly,lx,ly])

    ### Parameters for this panel
    wd_box= 0.65 ## Width of box
    wd_vio= 0.85 ## Width of violin
    xtm= np.arange(0,nvars)  ## x-location for box/violin plot

    ### Collect data for box and violin plot
    data=[]
    for i,ph in enumerate(ph2draw):
        data.append(strs[phs==ph])

    ### There are several properties to change for the box plot
    flierprops= dict(marker='.', markerfacecolor='gray',
                      markeredgecolor='none', markersize=3,
                      linestyle='none' )
    meanprops= dict(marker='x', markeredgecolor='k',
                     markerfacecolor='k', markersize=9,
                     markeredgewidth=2.5 )

    box1=ax3.boxplot(
             data, whis=[5,95], widths=wd_box, positions=xtm,
             showfliers=True, meanline=False, showmeans=True,
             boxprops=dict(linewidth=1.5, color='k'),
             medianprops=dict(color='k', linewidth=1.5),
             meanprops=meanprops, flierprops=flierprops,
             capprops=dict(linewidth=1.5,color='k'),
             whiskerprops=dict(linewidth=1.5,linestyle='--')
    )

    ### Draw violin plot
    vio1= ax3.violinplot(data, positions=xtm,
                         showextrema=False, widths=wd_vio)
    for b1 in vio1['bodies']:
        b1.set_color('PaleGoldenrod')
        b1.set_alpha(0.9)

    ### Fine tuning and decorating
    subtit='({}) Box+Violin'.format(abc[ai]); ai+=1
    ax3.set_title(subtit, fontsize=13, ha='left', x=0.0)
    ax3.set_yticklabels('')
    ax3.yaxis.set_ticks_position('both')
    ax3.set_xlim(xtm[0]-0.7,xtm[1]+0.7)
    ax3.set_xticklabels(['Phase{}'.format(ph) for ph in ph2draw])
    ax3.grid(axis='y', color='0.7', linestyle=':', linewidth=1)
    ix=ix+lx+gapx

    ###--- Panel4: Violin plot, half and half
    ax4=fig.add_axes([ix,iy-ly,lx,ly])

    ### Draw violin plot first
    vio1=ax4.violinplot([data[0],], positions=[0,],
                        showextrema=False, widths=1 )
    vio2=ax4.violinplot([data[1],], positions=[0,],
                        showextrema=False, widths=1 )

    ### Change properties of violin plot
    for b1,b2 in zip(vio1['bodies'], vio2['bodies']):
        b1.set_color(cc[0]); b1.set_alpha(0.8)
        m= np.mean(b1.get_paths()[0].vertices[:, 0])
        b1.get_paths()[0].vertices[:, 0]= np.clip(b1.get_paths()[0].vertices[:, 0], -np.inf, m)

        b2.set_color(cc[1]); b2.set_alpha(0.8)
        m= np.mean(b2.get_paths()[0].vertices[:, 0])
        b2.get_paths()[0].vertices[:, 0]= np.clip(b2.get_paths()[0].vertices[:, 0], m, np.inf)

    ### Need to draw manual legend
    import matplotlib.patches as mpatches
    patch1= mpatches.Patch(color=cc[0])
    patch2= mpatches.Patch(color=cc[1])
    ax4.legend([patch1,patch2], ['Phase{}'.format(ph) for ph in ph2draw],
                bbox_to_anchor=(0.02,0.995), loc='upper left',
                fontsize=10, framealpha=0.6, borderaxespad=0.
    )

    ### Fine tuning and decorating
    subtit='({}) Violin vs. Violin'.format(abc[ai]); ai+=1
    ax4.set_title(subtit,fontsize=13,ha='left',x=0.0)
    ax4.set_xlim(-1.1,1.1)
    ax4.set_xticks([0,])
    ax4.set_xticklabels('')
    ax4.axvline(x=0, linestyle='--', lw=1, c='k')
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    ax4.yaxis.set_ticks_position('both')
    ax4.set_ylabel('MJO Strength',fontsize=11,rotation=-90, va='bottom') #,labelpad=0)
    ax4.grid(axis='y',color='0.7', linestyle=':', linewidth=1)

    ###--- Save or Show
    plt.savefig(pdata['fnout'], bbox_inches='tight', dpi=150)
    print(pdata['fnout'])
    #plt.show()
    return

def barh_common(ax,subtit,y_dim=10,yt_labs=[],x_range=[]):
    """
    Decorating Barh plot
    """
    ### Title
    ax.set_title(subtit,fontsize=13,ha='left',x=0.0)

    ### Axis Control
    yy=np.arange(y_dim+1)
    ax.set_ylim(yy[0]-0.6,yy[-2]+0.6)
    ax.set_yticks(yy-0.5)
    ax.set_yticklabels(yt_labs) #,rotation=35,ha='right')
    if len(x_range)==2:
        ax.set_xlim(x_range)

    ### Ticks and Grid
    ax.tick_params(axis='both',which='major',labelsize=10)
    #ax.xaxis.set_major_formatter(FormatStrFormatter("%d%%"))
    ax.xaxis.set_major_formatter("{x:.0f}%")
    if x_range[1]-x_range[0]<=5:
        ax.xaxis.set_major_locator(MultipleLocator(1))
    elif x_range[1]-x_range[0]<=10:
        ax.xaxis.set_major_locator(MultipleLocator(2))
    elif x_range[1]-x_range[0]<=30:
        ax.xaxis.set_major_locator(MultipleLocator(5))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_ticks_position('both')
    ax.grid(axis='both',color='0.7', linestyle=':', lw=1)
    return

if __name__=="__main__":
    main()
