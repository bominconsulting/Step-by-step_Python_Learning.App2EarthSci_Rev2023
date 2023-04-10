# 1-1.read_grb_select.py 
import pygrib as pg

idir='./Data/'
ifnam='U.19500101.grb'
fi=pg.open(idir+ifnam)
var=fi.select(name='U component of wind')

for grb in fi:
	   print(grb)

