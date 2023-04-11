import numpy as np
import matplotlib.pyplot as plt

from src.linear_regression.regression import(
        multi_regress)

def plotting(x,y,xlab,ylab, type = 'g.'):
    plt.plot(x,y,type)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.autoscale(enable=True, tight=True)
    
        
def main():
    filename = 'M_data.txt'
    
    time, magnitude = np.loadtxt(filename, unpack=True)
    
    plotting(time, magnitude, 'time (hours)', 'magnitude')
    plt.show()
    
    mintime = 45
    maxtime = 120
    subtime = []
    submag = []
    
    count = 0
    for row in time:
        if row >= mintime and row <= maxtime:
            subtime.append(row)
            submag.append(magnitude[count])
        count += 1
    
    plotting(subtime, submag, 'time (hours)', 'magnitude') 
    plt.show()
    
    minm = 0
    maxm = max(submag)-0.5
    total = len(submag)
    interval = 0.05
    num_dots = int((maxm-minm)/interval)
    
    M = np.linspace(minm, maxm, num_dots)
    num = np.zeros(len(M))
    
    for i in range(len(submag)):
        for j in range(num_dots):
            if submag[i]>M[j]:
                num[j] += 1
    
    plotting(M, num, 'magnitude', 'number of earthquakes','o')
    plt.show()
    
    lin = np.zeros(len(M))
    for i in range(len(num)):
        if num[i]<=0:
            None
        else:
            lin[i] = np.log10(num[i])
    
    Z = np.zeros((len(num),2))
    for i in range(len(num)):
        Z[i,0]=1
        Z[i,1]=M[i]
    
    a, r, rsq = multi_regress(lin, Z)
    
    plotting(M,lin,'magnitude', 'log(number of earthquakes)','o')
    
    M = np.linspace(min(Z[:,1]),max(Z[:,1]),100)
    y = a[0] + a[1]*M
    
    plt.plot(M,y,'r-')
    plt.plot(M,y,'r.')
    plt.show()
    


main()