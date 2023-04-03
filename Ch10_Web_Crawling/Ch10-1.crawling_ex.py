# 1.crawling_ex.py 파이썬 크롤링 연습
import requests as rq
from bs4 import BeautifulSoup

url='https://ozonewatch.gsfc.nasa.gov/data/omps/Y2018/'
test=rq.get(url)
# rq.get만 이용할 시, 전체 텍스트로 인식하여 자료 가공이 어려움
# html 자료 가공을 위해 BeautifulSoup 함수 사용

test_html=BeautifulSoup(test.content,'html.parser')

print(test_html)

