import numpy as np
import matplotlib.pyplot as plt

from src.linear_regression.regression import(
        multi_regress)

def plotting(x,y,xlab,ylab, type = 'g.'):
    """Plots data.
    
    Parameters
    ----------
    x:  array_like, shape = (n, ) or (n,1)
        The list of x values
    y:  array_like, shape = (n, ) or (n,1)
        The list of y values
    xlab:   string
        label for the x axis
    ylab:   string
        label for the y axis
    type:   string
        command for indicator on the graph
    
    """
    plt.plot(x,y,type)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.autoscale(enable=True, tight=True)
    
        
def main():
    filename = 'M_data.txt'
    
    time, magnitude = np.loadtxt(filename, unpack=True)
    
    plotting(time, magnitude, 'time (hours)', 'magnitude')
    plt.show()
    
    # creating smaller lists to reduce time range
    timerange = np.linspace(0,120,12)
    b = []
    inty = []
    for i in range(len(timerange)-1):
        subtime = []
        submag = []
        count = 0
        intt =(timerange[i]+timerange[i+1])/2
        inty.append(intt)
        for row in time:
            if row >= timerange[i] and row <= timerange[i+1]:
                subtime.append(row)
                submag.append(magnitude[count])
            count += 1
        
        # for selected range
        minm = 0
        maxm = max(submag)-0.5
        total = len(submag)
        interval = 0.05
        num_dots = int((maxm-minm)/interval)
        
        M = np.linspace(minm, maxm, num_dots)
        num = np.zeros(len(M))
        
        # counting number of earthquakes over a given magnitude
        for i in range(len(submag)):
            for j in range(num_dots):
                if submag[i]>M[j]:
                    num[j] += 1
        
        # linearizing using log relationship. skips locations where num = 0
        lin = np.zeros(len(M))
        for i in range(len(num)):
            if num[i]<=0:
                None
            else:
                lin[i] = np.log10(num[i])
        
        plotting(M, lin, 'magnitude', 'number of earthquakes','o')
        
        # making Z matrix
        Z = np.zeros((len(num),2))
        for i in range(len(num)):
            Z[i,0]=1
            Z[i,1]=M[i]
        a, r, rsq = multi_regress(lin, Z)
        y = a[0] + a[1]*M
        plt.plot(M, y, '-')
        plt.title(intt)
        plt.show()
        b.append(abs(a[1]))
    plt.plot(inty,b,'o')
    plt.ylabel('b')
    plt.xlabel('middle of interval (hours)')
    plt.autoscale(enable=True, tight=True)
    plt.show()
    
    
    ## for whole original data
    minm = 0
    maxm = max(magnitude)-0.5
    total = len(magnitude)
    interval = 0.05
    num_dots = int((maxm-minm)/interval)
    
    M = np.linspace(minm, maxm, num_dots)
    num = np.zeros(len(M))
    
    # counting number of earthquakes over a given magnitude
    for i in range(len(magnitude)):
        for j in range(num_dots):
            if magnitude[i]>M[j]:
                num[j] += 1
    
    plotting(M, num, 'magnitude', 'number of earthquakes','o')
    plt.show()
    
    # linearizing using log relationship. skips locations where num = 0
    lin = np.zeros(len(M))
    for i in range(len(num)):
        if num[i]<=0:
            None
        else:
            lin[i] = np.log10(num[i])
    
    # making Z matrix
    Z = np.zeros((len(num),2))
    for i in range(len(num)):
        Z[i,0]=1
        Z[i,1]=M[i]
    a, r, rsq = multi_regress(lin, Z)
    
    plotting(M,lin,'magnitude', 'log(number of earthquakes)','o')
    
    # making a line of best fit from linear regression
    M = np.linspace(min(Z[:,1]),max(Z[:,1]),100)
    y = a[0] + a[1]*M
    
    plt.plot(M,y,'r-')
    plt.plot(M,y,'r.')
    plt.show()
    


main()