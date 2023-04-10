# 1.read_grb.py 
import pygrib as pg
# idir은 input 파일 디렉토리 위치입니다.
# ifnam은 input 파일 이름입니다.
idir='./Data/'
ifnam='U.19500101.grb'
fi=pg.open(idir+ifnam)

for grb in fi:
	# 자료는 wgrib 때와 같이 2차원 자료를 하나의 단위로 읽습니다.
	   print(grb)

