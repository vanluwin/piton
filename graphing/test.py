import numpy as np
import mpmath as mp
import scipy
import scipy.stats as sp
import matplotlib.pyplot as plt
import subprocess
import cmath as cm
#import scipy.fftpack as sf

#x1 is a two dimensional list with one row as its value and
#other row as its domain point n of that value i.e. x1=[x,n]
def dtft(x1,N):
    
    x=x1[0]
    j=cm.sqrt(-1)
    n=x1[1]
    X=[]
    
    w=np.linspace(-np.pi,np.pi,N)
    for i in range(0,N):
        w_tmp=w[i]
        X_tmp=0
        for k in range(0,len(x)):
            X_tmp+=(x[k]*np.exp(-n[k]*w_tmp*j))

       
        X.append(abs(X_tmp))
    
    plt.plot(w,X)
    plt.show()
    
#Example
x=[1/2,1/2]
n=[0,1]
x1=[x,n]
dtft(x1,100)