# 3.OMPS_download.py OMPS 오존전량 자료 파일명 크롤링
import requests as rq
import wget
from bs4 import BeautifulSoup
import os

year='Y2018'
url='https://ozonewatch.gsfc.nasa.gov/data/omps/'+year+'/'
odir='./Data/OMPS/'+year+'/'
if not os.path.isdir(odir):  # Check if the directory already exists
    os.makedirs(odir)  # Create a directory

test=rq.get(url)
test_html=BeautifulSoup(test.content,'html.parser')
atag=test_html.find_all('a')

for i in atag:
    href=i.attrs['href']
    if href.endswith(".txt"):
        print(href)
        wget.download(url+href,odir+href)
