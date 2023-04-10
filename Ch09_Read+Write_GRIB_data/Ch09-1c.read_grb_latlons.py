#1-2.read_grb_latlons.py 
import pygrib as pg
idir='./Data/'
ifnam='U.19500101.grb'
fi=pg.open(idir+ifnam)
var=fi.select(name='U component of wind')[0]

lats,lons=var.latlons()

print('Latitude = ',lats.min(), lats.max())
print('Longitude = ',lons.min(), lons.max())

