import numpy as np
import matplotlib.pyplot as plt    
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

def mullinreg3(x,a,b,c,d):
    return a + b*x[0] + c*x[1] + d*x[2]
 # 로지스틱 커브에서와 마찬가지로 curve_fit에 적용할 3변수 다항식의 형태를 만들어줍니다.

# 예제를 위한 시계열을 생성합니다.
ntime = 100
np.random.seed(1)
xvar1 = np.random.randn(ntime) + .5 # 랜덤 요소
xvar2 = np.sin(np.arange(ntime)/10.) + np.random.randn(ntime)/4. # 주기 요소
xvar3 = np.random.randn(ntime)/2. - np.arange(ntime)/100. # 장기추세 요소
yvar = 3.*(xvar1**2.) + 5*(xvar2**3.) + 7*(xvar3**2.) + np.random.randn(ntime)/2.
yvar  = yvar - np.mean(yvar) 
# 세 x값을 이용하여 비선형적으로 y값을 생성하고, 편의상 y값의 평균을 제거하여 편차값으로 바꾸어줍니다.

xvarset = np.stack((xvar1,xvar2,xvar3)) # 세 x 변수를 합쳐서 (ntime, 변수갯수)의 형태로 쌓아줍니다.
popt, pcov = curve_fit(mullinreg3, xvarset, yvar) # 선언된 3변수 다항식에 x변수 세트 및 y 변수 세트를 입력, 다변수 회귀분석을 수행하여 popt (즉, a0, a1, a2, a3)를 계산합니다. 

predicted_y = mullinreg3(xvarset, popt[0],popt[1],popt[2],popt[3])
# 선언된 3변수 다항식에 x변수 세트 및 계산된 a0,a1,a2,a3를 입력하여 fitted y를 계산합니다.

con_x1 = xvar1 * popt[1] #x1과 a1을 곱하여 x1의 기여도 (a1*x1) 시계열 생성 
con_x2 = xvar2 * popt[2] #x2과 a2을 곱하여 x2의 기여도 (a2*x2) 시계열 생성
con_x3 = xvar3 * popt[3] #x3과 a3을 곱하여 x3의 기여도 (a3*x3) 시계열 생성
intercept = popt[0] * np.ones(ntime) #계산된 y 절편을 시계열로 변환
fig = plt.figure(figsize = (8,6), dpi = 100)
gs = gridspec.GridSpec(4,1, left=0.1, right=0.9 , top = 0.9, bottom = 0.1,
                        height_ratios=[3,1,1,1], wspace = 0.05, hspace =0.05)
ax0 = plt.subplot(gs[0])
 # 앞서 설명된 gridspec을 이용하여 서브플롯 위치를 조정해줍니다.

times = np.arange(ntime)  #x축 생성
ax0.plot(times, yvar, 'o-',color = 'k', linewidth = 1,zorder = 5, label = 'y') # y값 표시
ax0.plot(times, predicted_y, marker='o', color = 'grey', linewidth = 1, linestyle = '--', zorder = 4, label = 'predicted y') 
# 다변수 회귀분석을 통하여 예측된 y값 (fitted y)를 표시
ax0.tick_params(axis='x', labelbottom=False)
# x tick을 생략하도록 설정.
colorset = {'x1':'r', 'x2':'g', 'x3':'b', 'x0':'lightgrey'} # 각 변수별 색깔을 딕셔너리(dictionary)로 지정
dataset = {'x1':xvar1, 'x2':xvar2, 'x3':xvar3} 
# 시계열로 그려질 각 변수 데이터를 각 변수의 이름을 이용하여 딕셔너리(dictionary)로 선언
contset =  {'x1':con_x1, 'x2':con_x2, 'x3':con_x3, 'x0':intercept} 
# 막대그래프로 그려질 기여도 데이터를 각 변수의 이름을 이용하여 딕셔너리(dictionary)로 선언

for vv, vname in enumerate(['x1','x2','x3']): #enumerate 설명은 아래 참조
    ax = plt.subplot(gs[vv+1], sharex=ax0) 
    # 각 변수의 시계열 및 기여도가 그려질 ax를 선언하고, ax0와 x축 특성을 공유하도록 설정
    ax.plot(times, dataset[vname], color =colorset[vname]) # 변수 시계열 표출
    ax.set_xlim([0,ntime])

    subax = ax.twinx()  # 오른쪽 y축을 추가로 생성.
    subax.bar(times, contset[vname], color =colorset[vname], alpha = 0.5)
    # 앞서 계산된 변수별 기여도를 plt.bar를 이용하여 그리기
    minr,maxr = contset[vname].min(), contset[vname].max()
    ax.set_ylabel(vname)
    subax.set_ylim([np.floor(minr),np.ceil(maxr)]) #minr 및 maxr을 이용하여 y축 범위 설정
    subax.set_ylabel(vname +'\n cont.')
    if vname != 'x3':
        ax.tick_params(axis='x', labelbottom=False)
    #x3가 아닌 경우, 즉 맨 밑의 서브플롯이 아닌 경우 틱 라벨을 생략해줍니다.
ax0.legend()
plt.show()

