from collections import deque

import numpy as np
from numpy import sin, cos, pi

from scipy.signal import spectrogram, butter, lfilter

class Plot():
    def __init__(self, axis, max_entries = 1000):
        self.x_axis = deque(maxlen=max_entries)
        self.y_axis = deque(maxlen=max_entries)

        self.axis = axis
        self.max_entries = max_entries
        
        self.lineplot, = axis.plot([], [], "b-")
        self.axis.grid()
        self.axis.set_autoscaley_on(True)

    def add(self, x, y):
        self.x_axis.append(x)
        self.y_axis.append(y)

        self.lineplot.set_data(self.x_axis, self.y_axis)
        self.axis.set_xlim(self.x_axis[0], self.x_axis[-1] + 1e-15)
        self.axis.relim(); self.axis.autoscale_view() # rescale the y-axis

    def animate(self, figure, callback, interval = 50):
        import matplotlib.animation as animation

        def wrapper(frame_index):
            self.add(*callback(frame_index))
            self.axis.relim(); self.axis.autoscale_view() # rescale the y-axis
            return self.lineplot

        animation.FuncAnimation(figure, wrapper, interval=interval)

def butter_bandpass_filter(plot, k, s, lowcut, highcut, fs, order=4):
    # Cutoff frequencies expressed as a fraction of the Nyquist frequency
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')

    filtered_s = lfilter(b, a, s)

    plot.clear()

    plot.set(title='Filtered signal', xlabel='Time', xlim=(k[0], k[-1] + 1e-15))

    plot.grid(True)

    plot.plot(k, filtered_s)

    return filtered_s

def plotFourier(plot, T, k, s, f_range):

    N = len(k)

    #freq = np.linspace(0.0, 1.0/(2.0*T), N/2)

    freqs = np.fft.fftfreq(N, d=T)

    fourier = np.abs(np.fft.fft(s))

    plot.cla()

    plot.set(title='FFT', xlabel='Frequency')   

    plot.plot(freqs[1:f_range], 2.0/N * np.abs(fourier[:N//2])[1:f_range])

def f(k):
    noise = sin(2 * pi * 60 * k) 
    noise += 0.5 * cos(2 * pi * 35 * k + 0.1)
    noise += 0.3 * cos(2 * pi * 40 * k)
    noise += 0.3 * cos(2 * pi * 45 * k)
    noise += 0.3 * cos(2 * pi * 50 * k)
    return sin(2 * pi * 7 * k) + noise 

def f2(k):
    noise = sin(2 * pi * 60 * k) 
    noise += 0.5 * cos(2 * pi * 35 * k + 0.1)
    noise += 0.3 * cos(2 * pi * 40 * k)
    noise += 0.3 * cos(2 * pi * 45 * k)
    noise += 0.3 * cos(2 * pi * 50 * k)
    return sin(2* pi * 4 * k) + noise
  
def f3(k):
    noise = sin(2 * pi * 60 * k) 
    noise += 0.5 * cos(2 * pi * 35 * k + 0.1)
    noise += 0.3 * cos(2 * pi * 40 * k)
    noise += 0.3 * cos(2 * pi * 45 * k)
    noise += 0.3 * cos(2 * pi * 50 * k)
    return  2 * sin(2* pi * 1 * k) + noise

def main():
    from matplotlib import pyplot as plt
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 5))
    fig.tight_layout(h_pad=4.0)
    plt.subplots_adjust(top=0.929, bottom=0.117, left=0.051, right=0.98, hspace=0.48, wspace=0.123)
        
    axes[0, 0].set(title='Noisy signal', xlabel='Time')

    timePlot = Plot(axes[0, 0])

    start_time = np.arange(0, 10, 0.01)
    start_signal = f(start_time)

    timePlot.x_axis = deque(start_time, maxlen=1000)
    timePlot.y_axis = deque(start_signal, maxlen=1000)
    
    k = 10
    T = 0.01

    while True: 
        k = round(k + T, 2)

        if(k < 14):
            s = f2(k)
        else:
            s = f3(k)

        timePlot.add(k, s)
    
        
        plotFourier(axes[0, 1], T, timePlot.x_axis, timePlot.y_axis, 500)

        filtered_s = butter_bandpass_filter(axes[1, 0], timePlot.x_axis, timePlot.y_axis, 1, 15, 100, order=6)

        plotFourier(axes[1, 1], T, timePlot.x_axis, filtered_s, 300)

        plt.pause(0.0083)
    
    plt.show()


if __name__ == "__main__": 
    main()