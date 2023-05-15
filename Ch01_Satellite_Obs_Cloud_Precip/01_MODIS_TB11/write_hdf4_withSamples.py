import sys
import os.path
import numpy as np
import os
from pyhdf.SD import * #SD, SDC,setattr

def main():
    data_flist = os.listdir('./data/radiance') # MOD021KM (Radiance data)
    #geo_flist = os.listdir('./data/geo') 		 # MOD03 (Geolocation data)

    # Variable name from MOD021KM header file
    var_name = 'EV_1KM_Emissive'
    lat_name = 'Latitude'
    lon_name = 'Longitude'

    for fn in np.arange(np.size(data_flist)):
        data_fname = './data/radiance/'+data_flist[fn]
        #geo_fname = './data/geo/'+geo_flist[fn]

        fname= data_fname
        if fname.strip().split('.')[-1].lower() != 'hdf':
            continue
        else:
            out_fn= fname.replace('hdf','sample.hdf')

        ### Read original data
        hdf = SD(data_fname, SDC.READ)
        data_2d = hdf.select(var_name)
        #print(type(data_2d), data_2d); sys.exit()
        attr= data_2d.attributes()
        raw_data = data_2d[:,:]
        print(type(raw_data), raw_data.dtype,raw_data.shape)
        ndim= len(raw_data.shape)
        dim_names=[]
        for i in range(ndim):
            print(data_2d.dim(i))
            dim1= data_2d.dim(i)
            print(dim1.info()) #, dim1.units)
            dim_names.append(dim1.info()[0])

        #for item in attr.keys():
        #    print('{}: {}'.format(item,attr[item]))

        # Create an HDF file
        sd = SD(out_fn, SDC.WRITE | SDC.CREATE)

        # Create a dataset
        sds = sd.create(var_name, SDC.UINT16, raw_data.shape)
        #print(type(sds)) #; sys.exit()

        # Set dimension names
        for i in range(ndim):
            dim1 = sds.dim(i)
            dim1.setname(dim_names[i])

        # Assign an attribute to the dataset
        for item in attr.keys():
            setattr(sds,item,attr[item])


        # Write data
        sds[:] = raw_data

        # Close the dataset
        sds.endaccess()

        # Flush and close the HDF file
        sd.end()

        '''
        ##-- Open hdf4 file
        hdf_f = open_hdf4("hello.hdf")


        ##-- Print variable names
        var_names= print_hdf4_details(hdf_f)

        ##-- Select a variable to see the details
        while True:
            answer= input("\nIf want to attribute details, type the number of variable.\n")
            if answer.isnumeric() and (int(answer)>0 and int(answer)<=len(var_names)):
                vnm= var_names[int(answer)-1]
                print('\nAttributes of {}'.format(vnm))
                attr= hdf_f.select(vnm).attributes()
                for item in attr.keys():
                    print('{}: {}'.format(item,attr[item]))

                for i in range(ndim):
                    dim1= data_2d.dim(i)
                    print(dim1.info()) #, dim1.units)

            else:
                break
        hdf_f.end()  # Close hdf4 file
        sys.exit()
        '''
    return


def open_hdf4(fname):
    if not os.path.isfile(fname):
        print("File does not exist:"+fname)
        sys.exit()

    hid = SD(fname, SDC.READ)
    print("Open:",fname)
    return hid

def print_hdf4_details(hdf_fid):
    dsets= hdf_fid.datasets()
    vnames=[]
    for i,dd in enumerate(dsets.keys()):
        print("{:3d} Name: {}".format(i+1,dd))
        print("   Values: {}".format(dsets[dd]))
        vnames.append(dd)
    return vnames

if __name__ == "__main__":
    main()
