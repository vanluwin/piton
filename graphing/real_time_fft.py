from collections import deque

import numpy as np
from numpy import sin, pi

from scipy.signal import spectrogram

import random

class Plot():
    def __init__(self, axes, max_entries = 100):
        self.x_axis = deque(maxlen=max_entries)
        self.y_axis = deque(maxlen=max_entries)

        self.full_x = deque()
        self.full_y = deque()

        self.axes = axes
        self.max_entries = max_entries
        
        self.lineplot, = axes.plot([], [], "b-")
        self.axes.grid()
        self.axes.set_autoscaley_on(True)

    def add(self, x, y):
        self.x_axis.append(x)
        self.y_axis.append(y)

        self.full_x.append(x)
        self.full_y.append(y)

        self.lineplot.set_data(self.x_axis, self.y_axis)
        self.axes.set_xlim(self.x_axis[0], self.x_axis[-1] + 1e-15)
        self.axes.relim(); self.axes.autoscale_view() # rescale the y-axis

    def animate(self, figure, callback, interval = 50):
        import matplotlib.animation as animation

        def wrapper(frame_index):
            self.add(*callback(frame_index))
            self.axes.relim(); self.axes.autoscale_view() # rescale the y-axis
            return self.lineplot

        animation.FuncAnimation(figure, wrapper, interval=interval)

def plotFourier(plot, T, k, s):

    N = len(k)

    freq = np.linspace(0.0, 1.0/(2.0*T), N/2)

    fourier = np.abs(np.fft.fft(s, N))

    plot.clear()

    plot.set(title='FFT', xlabel='frequency', ylabel='PSD')

    plot.plot(freq, 2.0/N * np.abs(fourier[:N//2]))

def plotSpecgram(plot, T, k, s):

    N = len(k)

    #f, t, Sxx = spectrogram(s, T)

    plot.clear()

    plot.set(title='Spectrogram', xlabel='time', ylabel='Frequency[Hz]')
    
    plot.specgram(s, NFFT=N, Fs=1/T, noverlap=N//2)
    
def main():
    from matplotlib import pyplot as plt
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)
    fig.tight_layout(h_pad=4.0)

    ax1.set(title='Time signal', xlabel='time', ylabel='f')

    timePlot = Plot(ax1)

    k = 0
    T = 0.01

    while True:
        k = k + T
        #noise = 0.75 * random.random() - 1.65 * random.random()
        s = sin(2 * pi * 10 * k) + 0.75 * sin(2 * pi * 30 * k) 

        timePlot.add(k, s)

        # FFT with oly the points showing in the display (100)
        plotFourier(ax2, T, timePlot.x_axis, timePlot.y_axis)

        # FFT with all the points until now
        plotFourier(ax3, T, timePlot.full_x, timePlot.full_y)

        plotSpecgram(ax4, T, timePlot.full_x, np.asarray(timePlot.full_y))

        plt.pause(0.01)
    
    plt.show()

if __name__ == "__main__": 
    main()