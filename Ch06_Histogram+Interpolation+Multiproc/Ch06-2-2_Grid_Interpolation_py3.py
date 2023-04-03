"""
Read IMERG Precipitation from binary file,
and perform horizontal grid interpolation

1. Simple Method: if target_resol is multiple of org_resol, both w/ or w/o missings.
2. Using "scipy.interpolate.interp2d"
3. Area-weighted interpolation
4. Using "scipy.interpolate.griddata," which build grid from non-grid structure.

By Daeho Jin, 2020.03.04
---

Make compatible to object-oriented concept.
"scipy.interpolate.interp2d" would be depreciated after Scipy 1.10.0, so
it is replaced by "RegularGridInterpolator"

By Daeho Jin, 2023.01.30
"""
import numpy as np
import sys
import os.path
import Ch06_common_functions as cf

def main():
    ### Precip File Info
    ### This is a binary file, so this information should be known already.
    nlon, lon0, dlon= 3600, -179.95, 0.1
    nlat, lat0, dlat= 1800, -89.95, 0.1
    ### Build Lons and Lats based on above information
    lon= np.arange(nlon)*dlon+lon0
    lat= np.arange(nlat)*dlat+lat0

    ### Limit region for simpler example
    tgt_lats= [-15,5] ## in degree
    tgt_lons= [35,85] ## in degree
    latidx= [cf.lat_deg2y(y,lat0,dlat) for y in tgt_lats] ## index to match lat in degree
    lonidx= [cf.lon_deg2x(x,lon0,dlon) for x in tgt_lons] ## index to match lon in degree
    lat, lon= lat[latidx[0]:latidx[1]], lon[lonidx[0]:lonidx[1]]
    print("Lon:",lon.shape,lon[[0,-1]])
    print("Lat:",lat.shape,lat[[0,-1]])

    ### Read Precipitation data
    indir= './Data/'
    infn= indir+'IMERG_precipitationCal_V06B.20180101-0000.{}x{}.f32dat'.format(nlat,nlon)
    ### Two methods to read binary. Screen one method, and use the other.
    ### (1) In the case of small sized binary file, it is convenient to read as a whole
    in_dims= [nlat, nlon]
    pr= cf.bin_file_read2arr(infn, dtype=np.float32).reshape(in_dims)
    pr_ref= pr[latidx[0]:latidx[1],lonidx[0]:lonidx[1]]
    ### (2) In the case of large sized binary file
    ### and only a part of the large file is needed:
    offset= nlon*latidx[0]*4  ## Starting point to read
    ##<-- Last "4" means float_32bit = 4 Bytes
    pr= np.memmap(infn, mode='r', dtype=np.float32, offset=offset, shape=(latidx[1]-latidx[0],nlon))
    ## After slicing, memmap object is transformed to numpy array, so loaded to memory.
    pr_ref= np.array(pr[:,lonidx[0]:lonidx[1]])

    pr= 0  ## Initialize to flush memory space
    ### Check the read result
    print("Precip:", pr_ref.shape, pr_ref.min(), pr_ref.max())

    ### Build grid boundaries to be used for "pcolormesh"
    lon_b= np.insert(lon, 0, lon[0]-dlon)+dlon/2
    lat_b= np.insert(lat, 0, lat[0]-dlat)+dlat/2
    print("Lon_b:",lon_b.shape,lon_b[[0,-1]])
    print("Lat_b:",lat_b.shape,lat_b[[0,-1]])
    #x_ref, y_ref= np.meshgrid(lon_b,lat_b)
    y_ref, x_ref= np.meshgrid(lat_b,lon_b,indexing='ij')
    ### Reference data to be displayed
    display_data= [dict(x=x_ref, y=y_ref, data=pr_ref, title='Original Data (0.1-deg)'),]

    ### Method1: Simple interpolation
    ### Interpolate from (0.1-deg, 0.1-deg) to (0.5-deg, 1.0-deg)
    x_scaler= 10 ## 1-deg / 0.1-deg
    y_scaler= 5 ## 0.5-deg / 0.1-deg
    pr_intpl1= cf.interp2d_fine2coarse_simple(pr_ref, x_scaler=x_scaler, y_scaler=y_scaler)
    ### Build grid boundaries to be used for pcolormesh
    lon_b_intpl1= lon_b[::x_scaler]
    lat_b_intpl1= lat_b[::y_scaler]
    y_intpl1, x_intpl1= np.meshgrid(lat_b_intpl1, lon_b_intpl1, indexing='ij')
    ### Simple interpolation data to be displayed
    display_data.append(dict(x=x_intpl1, y=y_intpl1, data=pr_intpl1, title='To 0.5d & 1d'))

    ### Method2: Using "RegularGridInterpolator"
    ### Interpolate from 0.1-deg to 0.25-deg.
    from scipy.interpolate import RegularGridInterpolator
    ### First, set-up Interpolation object with input data
    intpl_f0= RegularGridInterpolator((lat,lon),pr_ref,bounds_error=False)

    ### Second, interpolate to new grid
    new_resol=0.25
    lon2= np.arange(lon_b[0]+new_resol/2,lon_b[-1],new_resol)
    lat2= np.arange(lat_b[0]+new_resol/2,lat_b[-1],new_resol)
    ## Need a tuple of 2D grid info of resulting grid
    Y2,X2= np.meshgrid(lat2,lon2,indexing='ij')
    pr_intpl_lin= intpl_f0((Y2,X2),method='linear')  ## Linear
    pr_intpl_cu= intpl_f0((Y2,X2),method='cubic')  ## Cubic

    ### Build grid boundaries to be used for pcolormesh
    #lon_b_intpl2= np.insert(lon2, 0, lon2[0]-new_resol)+new_resol/2
    #lat_b_intpl2= np.insert(lat2, 0, lat2[0]-new_resol)+new_resol/2
    lon_b_intpl2= np.arange(lon_b[0],lon_b[-1]+0.01,new_resol)
    lat_b_intpl2= np.arange(lat_b[0],lat_b[-1]+0.01,new_resol)
    y_intpl2, x_intpl2= np.meshgrid(lat_b_intpl2, lon_b_intpl2, indexing='ij')
    ### "RGI" interpolation data to be displayed
    display_data.append(dict(x=x_intpl2, y=y_intpl2, data=pr_intpl_lin, title='To 0.25d (RGI; linear)'))
    display_data.append(dict(x=x_intpl2, y=y_intpl2, data=pr_intpl_cu, title='To 0.25d (RGI; cubic)'))

    #--
    ### Interpolation with data having some missings
    #--
    ### Artificially adding missings
    pr_ms= np.copy(pr_ref)
    yy= np.arange(len(lat))
    for dx in range(25):
        xx= 180+dx+(yy/1.4).astype(int)
        pr_ms[yy,xx]= -999.9
    #pr_ms[95:105,:]= -999.9  ### Values at some locations are replaced by -999.9
    #pr_ms[:,240:260]= -999.9
    ### Reference data with missings to be displayed
    display_data2=[dict(x=x_ref, y=y_ref, data=np.ma.masked_less(pr_ms, -999.),
                   title='Original Data with missings'),]

    ### Method1b: Simple interpolation with Missings
    ### Interpolate from (0.1-deg, 0.1-deg) to (0.5-deg, 1.0-deg)
    x_scaler=10 ## 1-deg / 0.1-deg
    y_scaler=5 ## 0.5-deg / 0.1-deg
    pr_ms_intpl1= cf.interp2d_fine2coarse_simple(pr_ms, x_scaler=x_scaler, y_scaler=y_scaler,
                  include_missing=True, undef=-999.9, crt=0.5)
    ### Simple method interpolation data to be displayed
    display_data2.append(dict(x=x_intpl1, y=y_intpl1, data=pr_ms_intpl1, title='To 0.5d & 1d'))

    ### Method3: Area-weighted Interpolation
    ### Interpolate from 0.1-deg to 0.25-deg.
    ## Define an object and make it ready
    awi= cf.Area_weighted_interpolator(old_lons=lon, old_lats=lat, new_lons=lon2, new_lats=lat2)
    awi.get_weights()
    ## Perform interpolation
    pr_ms_intpl_aw= awi.interpolate2d(pr_ms, ud=-999.9, method='average', crt=0.5)
    ### Area-weighted interpolation data to be displayed
    display_data2.append(dict(x=x_intpl2, y=y_intpl2, data=np.ma.masked_less(pr_ms_intpl_aw, -999.),
                        title='To 0.25d (area-weighted)'))

    ### Method4: Using "griddata"
    ### Interpolate from no-grid data to 0.25-deg.
    from scipy.interpolate import griddata
    ### Input data in the form of (x,y,val), all 1-D, same length
    valid_idx= pr_ms>=0.
    grid_x, grid_y= np.meshgrid(lon,lat)
    new_grid_x, new_grid_y= np.meshgrid(lon2,lat2) ### 0.25-deg meshed grid
    pr_ms_gridded= griddata((grid_x[valid_idx],grid_y[valid_idx]), pr_ms[valid_idx],
                   (new_grid_x,new_grid_y), method='cubic', fill_value=-999.9) ## Cubic interpolation
    ### "griddata" interpolation data to be displayed
    display_data2.append(dict(x=x_intpl2, y=y_intpl2, data=np.ma.masked_less(pr_ms_gridded, -999.),
                        title='To 0.25d (griddata; cubic)'))

    ### Prepare data for plotting
    outdir= "./Pics/"
    fnout= outdir+"Code6-2-2_Grid_Interpolation_example.png"
    suptit="IMERG Precip [2018.01.01 00h-00m UTC]"
    pic_data= dict(data=(display_data, display_data2),
                   tgt_latlon= (tgt_lats, tgt_lons),
                   suptit=suptit, fnout=fnout,
    )
    plot_main(pic_data)
    return

###--------------------
### Display Data on Map
###--------------------
### Plan to draw two columns, four panels each.
### 1. (left) no-missing data
### 2. (right) with-missing data
import matplotlib.pyplot as plt
import matplotlib.colors as cls
from matplotlib.ticker import FixedLocator, MultipleLocator
import cartopy.crs as ccrs

def plot_main(pdata):
    data= pdata['data']
    tgt_lats, tgt_lons= pdata['tgt_latlon']
    abc, ai= 'abcdefghijklmn', 0
    ncol, nrow= len(data), len(data[0])

    ###--
    fig = plt.figure()
    fig.set_size_inches(6, 8.5)    ## (lx,ly)
    ### Page Title
    fig.suptitle(pdata['suptit'],fontsize=16,y=0.97,va='bottom') #stretch='semi-condensed'

    ### Parameters for subplot area
    left,right,top,bottom= 0.05, 0.95, 0.93, 0.05
    npnx,gapx,npny,gapy= ncol, 0.02, nrow, 0.07
    lx= (right-left-gapx*(npnx-1))/npnx
    ly= (top-bottom-gapy*(npny-1))/npny
    ix,iy= left, top

    ### Precip values vary exponentially, hence decide to use non-linear levels
    p_lev= [0.,0.1,0.2,0.5,1,2,5,10,20,50,100]
    cm= plt.cm.get_cmap('terrain_r'); ## Define ColorMap
    cm.set_bad('r')  ## Set for missing data (masked grid cells)
    cm.set_under('0.8')  ## Set for values below 0.
    norm= cls.BoundaryNorm(p_lev, ncolors=cm.N, clip=False) ## Transform continuous colormap into distinct one.
    props = dict(edgecolor='none',alpha=0.8,cmap=cm,norm=norm) ## Properties for pcolormesh map
    map_proj, data_proj= ccrs.PlateCarree(), ccrs.PlateCarree()

    ###-- Draw by panel
    for j, display_data in enumerate(data):
        for i, dp_data in enumerate(display_data):
            ax1= fig.add_axes([ix,iy-ly,lx,ly], projection=map_proj)
            ax1.set_extent([*tgt_lons,*tgt_lats], crs=data_proj) ### Limit the map region
            pic1= ax1.pcolormesh(dp_data['x'], dp_data['y'], dp_data['data'], **props)
            subtit='({}) {}'.format(abc[ai],dp_data['title']); ai+=1
            if j==0: gl_lab_locator=[False,True,True,False]
            else:    gl_lab_locator=[False,True,False,True]  ##[Top,Bottom,Left,Right]
            map_common(ax1, subtit, data_proj, gl_lab_locator)
            txt= 'Max={:.1f} mm/h'.format(dp_data['data'].max())
            ax1.text(0.02, 0.97, txt, ha='left',va='top', fontsize=10,
                     color='0.2', weight='bold', transform=ax1.transAxes)
            iy=iy-ly-gapy; print(subtit)
        iy=top; ix=ix+lx+gapx

    ### Draw Colorbar
    cb= cf.draw_colorbar(fig, ax1, pic1, type='horizontal', size='page', extend='both', width=0.02)
    cb.ax.set_xlabel('Precipitation Rate (mm/h)',fontsize=11)
    cb.set_ticks(p_lev) ### Specify tick location. Otherwise ticks are one in two bins.

    ###--- Save or Show
    plt.savefig(pdata['fnout'], bbox_inches='tight', dpi=150)
    print(pdata['fnout'])
    #plt.show()
    return

def map_common(ax, subtit, crs, gl_lab_locator=[False,True,True,False]):
    """ Decorating Cartopy Map
    """
    ### Title
    ax.set_title(subtit, fontsize=13, ha='left', x=0.0)
    ### Coast Lines
    ax.coastlines(color='silver', linewidth=1.)
    ### Grid Lines
    gl= ax.gridlines(crs=crs, draw_labels=True,
                    linewidth=0.6, color='gray', alpha=0.5, linestyle='--')
    ### x and y-axis tick labels
    gl.top_labels, gl.bottom_labels, gl.left_labels, gl.right_labels = gl_lab_locator
    gl.xlocator = MultipleLocator(10) #FixedLocator(range(30,100,10))
    gl.ylocator = MultipleLocator(5)
    gl.xlabel_style = {'size': 10, 'color': 'k'}
    gl.ylabel_style = {'size': 10, 'color': 'k'}
    ### Aspect ratio of map
    ax.set_aspect('auto') ### 'auto' allows the map to be distorted and fill the defined axes
    return

if __name__=="__main__":
    main()
