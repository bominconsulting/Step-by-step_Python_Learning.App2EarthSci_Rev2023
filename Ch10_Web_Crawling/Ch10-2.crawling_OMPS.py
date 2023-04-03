# 2.crawling_OMPS.py OMPS 오존전량 자료 파일명 크롤링
import requests as rq
from bs4 import BeautifulSoup

url='https://ozonewatch.gsfc.nasa.gov/data/omps/Y2018/'
test=rq.get(url)

test_html=BeautifulSoup(test.content,'html.parser')
atag=test_html.find_all('a')
omps_list=[]

for i in atag:
        href=i.attrs['href']
        if href.endswith(".txt"):
             omps_list.append(href)
             print(href)

