import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab

#%% create signal
NFFT= 1024  
f1 = 4
f2 = 10
dt  = 0.01
Fs = int(1.0 / dt)
t  = np.arange(0, 10, dt)
s = np.sin(2 * np.pi *f1* t)+ 0.5*np.sin(2 * np.pi * f2*t)

#%% plot values
plt.subplot(311)
plt.plot(t, s)
plt.subplot(312)
plt.psd(s, 512, 1 / dt)
plt.subplot(313)
plt.specgram(s, NFFT=NFFT, Fs=Fs, noverlap=100)

plt.show()