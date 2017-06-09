#!/usr/bin/python

import urllib
from numpy import genfromtxt

url='http://www.google.com/finance/historical?q={}&startdate=Jan+01%2C+2000&output=csv'

if __name__ == '__main__':
    tickers = ['NASDAQ:AAPL', 'NASDAQ:AAL']
    for t in tickers:
        destination=t.split(':')[-1]+'.csv'
        p = urllib.urlretrieve(url.format(t), destination)
        print destination
        print(p)
        data = genfromtxt(destination)
        print data
        
    
 
    
