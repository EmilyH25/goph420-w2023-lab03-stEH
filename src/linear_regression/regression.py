import numpy as np

def multi_regress(y, Z):
    """Perform multiple linear regression.
    
    Parameters
    ----------
    y:  array_like, shape = (n, ) or (n,1)
        The vector of dependent variable data
    Z:  array_like, shape = (n,m)
        The matrix of independent variable data
    
    Returns
    -------
    numpy.ndarray, shape = (m, ) or (m,1)
        The vector of model coefficients
    numpy.ndarray, shape = (n, ) or (n,1)
        The vector of residuals
    float
        The coefficient of determination, r^2
    """
    n=np.size(Z,0)
    m=np.size(Z,1)
    X=np.zeros([m,m])
    Y=np.zeros([m,1])
    r=np.zeros([n,1])
    
    ZT = np.transpose(Z)
    et = np.dot(ZT,Z)
    nt = np.dot(ZT,y)
    inet = np.linalg.inv(et)
    a = np.dot(inet,nt)
    # building Z matrix
    #for k in range(m):
     #   for j in range (m):
      #      X[k,j]=0
       #     for i in range (n):
        #        X[k,j]=X[k,j]+Z[i,k]*Z[i,j]
    
    # building Y column vector
    #for k in range (m):
     #   Y[k]=0
      #  for i in range (n):
       #     Y[k]=Y[k]+Z[i,k]*y[i]
    
    # Inverse Z matrix
    #inverseX = np.linalg.pinv(X)
    # Coefficients column vector
    #a = np.dot(inverseX, Y)
    
    for i in range (n):
        r[i]=0
        for j in range (m):
            r[i] = r[i]+a[j]*Z[i,j]
    
    sumy = 0
    for i in range (n):
        sumy = sumy + y[i]
    mean = sumy/n
    
    St = 0
    for i in range (n):
        St = St + ((y[i]-mean)**2)
    Sr = 0
    for i in range(n):
        Sr = Sr + (r[i]**2)
    r2=(St-Sr)/St
    
    return a, r, r2