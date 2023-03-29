import numpy as np
import sys
import os.path
from datetime import timedelta, date
import h5py

def main():
    indir= './Data/'
    '''
    ### Below commented block is an example for reading several IMERG Lv3 files together
    today_files=[]
    todate= date(2018,1,1)
    for hh in np.arange(0,24,0.5):
        fn=indir+get_imerg_lv3_hdf_file_name(todate,hh)
        today_files.append(open_hdf5(fn))
    '''

    ### This time we need just one file
    infn= indir+'3B-HHR.MS.MRG.3IMERG.20180101-S000000-E002959.0000.V06B.HDF5'
    hdf_f=open_hdf5(infn)

    '''
    ### If need to check the details of HDF5 file
    h5keys=[]
    hdf_f.visit(h5keys.append)
    for i,key in enumerate(h5keys):
        print("{} Key: {}".format(i,key))
        for (name,val) in hdf_f[key].attrs.items():
            print("\t{}: {}".format(name,val))
    '''

    ### Assume we already know "keys"
    key0='Grid'
    tgt_keys=['lon','lat','precipitationCal']

    data=[]
    for key1 in tgt_keys:
        data.append(hdf_f[key0][key1][:].astype(float))  # Save as Numpy array, dtype=float
        if key1[:2]=='pr':
            print("{}: {}, {:.1f}, {:.1f}".format(key1,data[-1].shape,data[-1].min(),data[-1].max()))  # Check the range of data values
        else:
            print("{}: {}, {:.1f}, {:.1f}".format(key1,data[-1].shape,data[-1][0],data[-1][-1])) # Check if Lon or Lat is flipped

    lon,lat,pr= data
    ### Correct dimension from [1,Lon,Lat] to [Lat,Lon]
    pr=pr.squeeze().T

    ### Gather basic info of Lat and Lon
    nlon, lon0, dlon= lon.shape[0], lon[0], lon[1]-lon[0]
    nlat, lat0, dlat= lat.shape[0], lat[0], lat[1]-lat[0]

    outfn= indir+'IMERG_precipitationCal_V06B.20180101-0000.{}x{}.f32dat'.format(nlat,nlon)
    with open(outfn, 'wb') as f:
        pr.astype(np.float32).tofile(f)  # Save as 4-Byte float binary

    return

def open_hdf5(fname):
    if not os.path.isfile(fname):
        print("File does not exist:"+fname)
        sys.exit()
    #print("Open:",fname)
    return h5py.File(fname,'r')

def get_imerg_lv3_hdf_file_name(tday,hh):
    '''
    Return IMERG Level3 HDf file name from given time information
    [year]/[file name]
    '''
    ver='V06B'
    dd=tday.strftime('%Y%m%d')

    hr=int(hh); mm=hh-int(hh)
    h1=hr*10000+mm*6000
    h2=h1+2959
    h3=hh*60
    h1=str(int(h1)).zfill(6)
    h2=str(int(h2)).zfill(6)
    h3=str(int(h3)).zfill(4)
    fn='{}/3B-HHR.MS.MRG.3IMERG.{}-S{}-E{}.{}.{}.HDF5'.format(dd[:4],dd,h1,h2,h3,ver)
    return fn

if __name__=="__main__":
    main()
