# 2.read_write_grb.py 
import pygrib as pg

idir='./Data/'
# odir은 output 파일 디렉토리 위치입니다.
odir=idir

ifnam='U.19500101.grb'
# ofnam은 output 파일 이름입니다.
ofnam='U.19500101.test.grb'

fi=pg.open(idir+ifnam)
ofi=open(odir+ofnam,'wb')

for grb in fi:
	msg=grb.tostring()
	ofi.write(msg)

ofi.close()

