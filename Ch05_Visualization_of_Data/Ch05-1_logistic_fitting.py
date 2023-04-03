import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt    

def logistic(xx, a0,a1,a2,a3):
    yy = ( (-a2) / (1.+np.exp(-a0*(1.-xx*a1))) ) + a3
    return yy
## 로지스틱 커브 함수. curve_fit 함수에 적용하기 위하여 선언되어야 합니다.
## xx는 x축 데이터를 의미. a0부터 a3는 로지스틱 함수의 개형을 결정하는 인자.
## xx와 a0, a1, a2, a3 값이 주어졌을 때, 주어진 x값과 해당 인자에 따른 로지스틱 커브의 y값을 출력

def residuals(a0,a1,a2,a3,yy,xx):
    err = np.absolute(yy - logistic(xx, a0,a1,a2,a3))
    return err
## 로지스틱 커브 접합을 수행한 이후, 계산된 최적 인자 (a0,a1,a2,a3)를 대입한 로지스틱 커브와 실제 관측값과의 잔차 (fitting error)를 계산하기 위한 함수

##우선 연습을 위한 랜덤 데이터셋을 생성합니다.
np.random.seed(1) ## 난수 생성 과정에 필요한 시드값 (seed)를 1로 설정
noise = np.random.randn(24)/20. ## y축 데이터 노이즈
julian = 15. * np.arange(24) +15. ## x축 데이터값
ndvi = (1. - np.cos((julian)*(2*np.pi/365.)))/2. + noise ## 임의로 생성된 y축 데이터
## 봄/겨울에 최소를 나타내고 여름에 최대를 보이는 계절변화 데이터를 생성.

maxp = np.argmax(ndvi) ## ndvi가 최댓값을 보이는 지점의 위치
pinit = [5.,0.01,0.5,0.5] ## 곡선 접합과정에서 사용될 각 인자 (a0,a1,a2,a3)의 초기값.

if maxp > 3: # 주어진 y값의 최댓값이 4번째 이후에 존재하는 경우,
    try: # try except 구문; curve_fit 실패를 대비.
        xdat = julian[:maxp+1] # 0번 데이터부터 maxp, 즉 최댓값이 나타나는 시점까지의 x값
        ydat = ndvi[:maxp+1] # 0번 데이터부터 maxp, 즉 최댓값이 나타나는 시점까지의 y값

        popt, pcov = curve_fit(logistic, xdat, ydat,
                p0=pinit, bounds=([0,0,0,0], [100,2,2,2])) # curve_fit은 이하의 표 참조
        a0,a1,a2,a3 = popt 
        sos = 1./a1 # 생장계절시작일 (NDVI 계절진폭의 50%에 도달하는 날짜)
        err = np.sum(residuals(a0,a1,a2,a3, ydat, xdat))
        print ('SOS:',format(sos, '3.1f'), ' Err:', format(err, '1.3f'))

    except RuntimeError: 
        sos = np.nan 
        print ('Fitting failure! Input or bounding problem?')
        raise SystemError
# curve_fit을 수행하기에 입력자료가 부족한 경우, 혹은 계산이 수렴하지 않아 곡선접합이 실패한 경우에는 Runtime Error가 발생할 수 있음.
# try except 구문을 이용하여 에러를 출력시키고 계산을 중단시키는 역할.

if ~np.isnan(sos): # curve_fit이 성공한 경우, 이를 간략히 시각화하는 과정.
    plt.figure(figsize = (6,4))
    plt.plot(xdat,ydat,'o',label = 'NDVI', color = 'k')
    plt.plot(xdat,logistic(xdat,a0,a1,a2,a3),color ='b'
            ,label = 'Logistic curve')
    plt.axvline(x = sos, color = 'b', linestyle = ':'
                ,label = 'SOS') # 특정 x 위치에 연직선을 그리는 옵션
    plt.xlabel('Julian day')
    plt.ylabel('NDVI')
    plt.legend()
    plt.title('Logistic curve fitting - SOS calculation')
    plt.tight_layout()
    plt.show()
