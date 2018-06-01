from collections import deque

import numpy as np
from numpy import sin, cos, pi

from scipy.signal import stft, butter, lfilter

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import time 

class TimePlot:
    def __init__(self, axis, max_entries = 1000):
        self.x_axis = deque(maxlen=max_entries)
        self.y_axis = deque(maxlen=max_entries)

        self.axis = axis

        self.lineplot, = axis.plot([], [], "b-")
        self.axis.grid()
        self.axis.set_autoscaley_on(True)
    
    def add_point(self, x, y):
        self.x_axis.append(x)
        self.y_axis.append(y)

        self.lineplot.set_data(self.x_axis, self.y_axis)

        self.axis.set_xlim(self.x_axis[0], self.x_axis[-1] + 1e-15)
        self.axis.relim(); self.axis.autoscale_view() # rescale the y-axis

    def add_array(self, x, y):
        self.x_axis = deque(x, maxlen=1000)
        self.y_axis = deque(y, maxlen=1000)

        self.lineplot.set_data(self.x_axis, self.y_axis)

        self.axis.set_xlim(self.x_axis[0], self.x_axis[-1] + 1e-15)
        self.axis.relim(); self.axis.autoscale_view() # rescale the y-axis

class FourierPlot:
    def __init__(self, axis, spacing, f_range):
        self.axis = axis
        self.T = spacing
        self.f_range = f_range * 10

        self.lineplot, = axis.plot([], [], "b-")
        self.axis.grid()
        self.axis.set_autoscaley_on(True)
        
    def plot(self, time, signal):
        freqs = np.fft.fftfreq(1000, self.T)
        freqs = freqs[1:self.f_range]

        fourier = np.abs(np.fft.fft(signal))
        fourier = 2.0/1e3 * np.abs(fourier[:1000//2])[1:self.f_range]

        self.lineplot.set_data(freqs, fourier)
        self.axis.relim(); self.axis.autoscale_view() # rescale the y-axis

class SpectrogramPlot:
    def __init__(self, fig, axis, spacing, f_range):
        self.fig = fig
        self.axis = axis
        self.T = spacing
        self.f_range = f_range

        self.cbar = False

    def remap(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

    def colorbar(self, mappable):
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        self.cbar = True
        self.colorb = fig.colorbar(mappable, cax=cax)

    def delete_cbar(self):
        self.colorb.remove()

    def plot(self, time, signal):
       
        (f, t, Zxx) = stft(signal, fs=int(1.0 / self.T), window='hamming', nfft=256)
        Zxx = np.abs(Zxx)
        t = np.vectorize(self.remap)(t, t[0], t[-1], time[0], time[-1])


        self.axis.clear()

        self.axis.set(title='Spectrogram', xlabel='Time', ylabel='Frequency', ylim=(0, self.f_range))

        im = self.axis.pcolor(t, f, Zxx)

        if self.cbar:
            self.delete_cbar()
 
        self.colorbar(im)

        

def get_filter(lowcut, highcut, fs, order=4):
    # Cutoff frequencies expressed as a fraction of the Nyquist frequency
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')

    return b, a

def f(k):
    noise = 2 * sin(2 * pi * 60 * k) 
    noise += 3.3 * cos(2 * pi * 45 * k)
    noise += 3.3 * cos(2 * pi * 50 * k)
    return 4 * sin(2 * pi * 8 * k) + noise 

def f2(k):
    noise = 2 * sin(2 * pi * 60 * k) 
    noise += 3.3 * cos(2 * pi * 45 * k)
    noise += 3.3 * cos(2 * pi * 50 * k)
    return 6 * sin(2* pi * 6 * k) + noise
  
def f3(k):
    noise = 2 * sin(2 * pi * 60 * k) 
    noise += 3.3 * cos(2 * pi * 45 * k)
    noise += 3.3 * cos(2 * pi * 50 * k)
    return 8 * sin(2* pi * 4 * k) + noise

def main():
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 5))
    fig.tight_layout(h_pad=4.0)
    plt.subplots_adjust(top=0.929, bottom=0.117, left=0.042, right=0.964, hspace=0.48, wspace=0.124)

    axes[0, 0].set(title='Noisy signal', xlabel='Time')
    axes[1, 0].set(title='Filtered Signal', xlabel='Time')
    axes[1, 1].set(title='FFT', xlabel='Frequency', ylabel='Amplitute')

    k = 10
    T = 0.01
    b, a = get_filter(1, 32, fs=int(1.0 / T), order=6)

    start_time = np.arange(0, 10, 0.01)
    start_signal = f(start_time)
    filtered_start_signal = lfilter(b, a, start_signal)

    noisyPlot = TimePlot(axes[0, 0])
    noisyPlot.add_array(start_time, start_signal)

    filteredPlot = TimePlot(axes[1,0])
    filteredPlot.add_array(start_time, filtered_start_signal)

    specgramPlot = SpectrogramPlot(fig, axes[0, 1], T, 30)

    fourierPlot = FourierPlot(axes[1, 1], T, 30)

    specBg = fig.canvas.copy_from_bbox(axes[0, 1].bbox)

    while True:
        
        k = round(k + T, 2)
        if(k < 12):
            s = f2(k)
        else:
            s = f3(k)

        noisyPlot.add_point(k, s)

        filteredPlot.add_array(noisyPlot.x_axis, lfilter(b, a, noisyPlot.y_axis))

        fig.canvas.restore_region(specBg)

        specgramPlot.plot(filteredPlot.x_axis, filteredPlot.y_axis)

        fourierPlot.plot(filteredPlot.x_axis, filteredPlot.y_axis)

        plt.pause(0.1)


    
    plt.show()


if __name__ == "__main__": 
    main()