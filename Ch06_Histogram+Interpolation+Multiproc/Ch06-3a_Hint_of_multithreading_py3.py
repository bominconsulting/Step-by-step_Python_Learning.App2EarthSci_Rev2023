"""
Test the performance of multi-threading.
Read IMERG Precipitation from binary file,
and perform area-weighted interpolation.
1. ThreadPoolExecutor
2. ProcessPoolExecutor

By Daeho Jin, 2020.03.04
---

Make compatible to object-oriented concept.

By Daeho Jin, 2023.01.30

"""
import sys
import os.path
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import repeat
import Ch6_common_functions as cf

def awi_thread_executor(func, indata, ud=-999., nThreads=1, chunk_size=0):
    """
    ThreadPoolExecutor
    - The asynchronous execution can be performed with threads
    - https://docs.python.org/3.8/library/concurrent.futures.html

    Function "chunks" provides parts of indata.
    Usually large chunks rather than small chunks perform faster.
    For the constant input required for given function,
    it is needed to "repeat" it for each thread/process.
    """
    if chunk_size<=0:
        chunk_size= indata.shape[0]//nThreads
        if indata.shape[0]%nThreads!=0:
            chunk_size+=1
    with ThreadPoolExecutor(max_workers=nThreads) as executor:
        results=executor.map(func,chunks(indata,chunk_size),repeat(ud))
    return results

def awi_process_executor(func, indata, ud=-999., nThreads=1, chunk_size=0):
    """
    ProcessPoolExecutor
    - The asynchronous execution can be performed with separate processes
    - https://docs.python.org/3.8/library/concurrent.futures.html

    Function "chunks" provides parts of indata.
    Usually large chunks rather than small chunks perform faster.
    For the constant input required for given function,
    it is needed to "repeat" it for each thread/process.
    """
    if chunk_size<=0:
        chunk_size= indata.shape[0]//nThreads
        if indata.shape[0]%nThreads!=0:
            chunk_size+=1
    with ProcessPoolExecutor(max_workers=nThreads) as executor:
        results=executor.map(func,chunks(indata,chunk_size),repeat(ud))
    return results

def chunks(data, chunk_size=1):
    """
    Overly-simple chunker...
    Basic idea here is that, assumed that given input data is 3-D,
    "yield" chunks of data by dividing the first axis of input data
    into a few sub-groups (chunks).
    """
    intervals = list(range(0, data.shape[0], chunk_size)) + [None,]
    for start, stop in zip(intervals[:-1], intervals[1:]):
        yield data[start:stop,:]

def main(nThreads):
    ### Precip File Info
    ### This is binary file, so this information should be known already.
    nlon, lon0, dlon= 3600, -179.95, 0.1
    nlat, lat0, dlat= 1800, -89.95, 0.1
    ### Build Lons and Lats based on above information
    lon= np.arange(nlon)*dlon+lon0
    lat= np.arange(nlat)*dlat+lat0

    ### Read Precipitation data
    indir= './Data/'
    infn= indir+'IMERG_precipitationCal_V06B.20180101-0000.{}x{}.f32dat'.format(nlat,nlon)
    ### In the case of small sized binary file and better to read as a whole
    pr= cf.bin_file_read2arr(infn,dtype=np.float32).reshape([nlat,nlon])
    pr= pr[300:1500,:] ## Cut the area of missings, so now 60S-60N
    nlat= 1200

    ### Transform current 2-D precip array into artificial 3-D array
    lat_scaler= 4; nlat2= nlat//lat_scaler ## Every 30 degrees
    lon_scaler= 6; nlon2= nlon//lon_scaler ## Every 60 degrees
    pr= pr.reshape([lat_scaler,nlat2,lon_scaler,nlon2]).swapaxes(1,2)
    pr= pr.reshape([lat_scaler*lon_scaler,nlat2,nlon2])
    ### Now the array represents 24 sub-regions of 45-deg x 60-deg size
    ### Check the read result
    print("Precip:",pr.shape)
    lon_sub= lon[:nlon//lon_scaler]
    lat_sub= lat[:nlat//lat_scaler]

    ### Interpolate from 0.1-deg to 0.25-deg.
    new_resol=0.25
    lon_target= np.arange(lon_sub[0]-dlon/2+new_resol/2,lon_sub[-1]+dlon/2,new_resol)
    lat_target= np.arange(lat_sub[0]-dlat/2+new_resol/2,lat_sub[-1]+dlat/2,new_resol)
    ### Area-weighted Interpolation
    awi= cf.Area_weighted_interpolator(old_lons=lon_sub, old_lats=lat_sub,
                        new_lons=lon_target, new_lats=lat_target) ### Define an object
    awi.get_weights()

    ### Multi-threading Setting
    if len(sys.argv) >= 3 and int(sys.argv[2]) <= np.ceil(pr.shape[0]/nThreads):
        chunk_size= int(sys.argv[2])
    else:
        chunk_size= int(np.ceil(pr.shape[0]/nThreads))
        print("chunk_size is changed to optimal number, {}".format(chunk_size))


    ### Run1: ProcessPool
    time0= time.time()
    results= awi_process_executor(awi.interpolate3d,pr, ud=-9999.9,
                                nThreads=nThreads, chunk_size= chunk_size)
    ### The output results are a set of numpy arrays
    pr_intpl_aw_p=np.concatenate(list(results))
    time1= time.time()
    print("Interpolated to:",pr_intpl_aw_p.shape)
    print("Process_Pool_executor with {} threads and {} chunk_size: {:.3f} sec".format(nThreads, chunk_size, time1-time0))

    ### Run2: ThreadPool
    time0= time.time()
    results= awi_thread_executor(awi.interpolate3d, pr, ud=-9999.9,
                                nThreads=nThreads, chunk_size= chunk_size)
    ### The output results are a set of numpy arrays
    pr_intpl_aw_t= np.concatenate(list(results))
    time1= time.time()
    print("Interpolated to:", pr_intpl_aw_t.shape)
    print("Thread_Pool_executor with {} threads and {} chunk_size: {:.3f} sec".format(nThreads, chunk_size, time1-time0))

    ### Run3: Single Thread
    time0= time.time()  ### Record the starting time
    pr_intpl_aw= awi.interpolate3d(pr,ud=-9999.9)
    time1= time.time()  ### Record the ending time
    print("Interpolated:",pr_intpl_aw.shape)
    print("Single Thread: {:.3f} sec".format(time1-time0))

    ### Test the results if they are equivalent.
    print("Test if each value is same between single vs. multi-thread")
    print(np.array_equal(pr_intpl_aw,pr_intpl_aw_t))
    print(np.array_equal(pr_intpl_aw,pr_intpl_aw_p))

    return

if __name__=="__main__":
    ### Multi-threading Setting
    try:
        nThreads= int(sys.argv[1])
    except:
        sys.exit("Number of Thread(s) is necessary")

    main(nThreads)
