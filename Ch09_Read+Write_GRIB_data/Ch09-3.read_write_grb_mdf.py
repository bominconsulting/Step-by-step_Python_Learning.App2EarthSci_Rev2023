# 2.read_write_grb.py 
import pygrib as pg

idir='./Data/'
odir=idir

ifnam='U.19500101.grb'
ofnam='U.19500101.mdf.grb'

fi=pg.open(idir+ifnam)
ofi=open(odir+ofnam,'wb')

for grb in fi:
# 기존 값에 10배를 한 뒤 저장합니다.
	grb.values=10*grb.values
	msg=grb.tostring()
	ofi.write(msg)

ofi.close()

