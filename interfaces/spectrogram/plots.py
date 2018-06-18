from collections import deque

import numpy as np
from numpy import sin, cos, pi

from scipy.signal import stft, butter, lfilter

from mpl_toolkits.axes_grid1 import make_axes_locatable

class TimePlot:
    """
    Creates a canvas for plotting a signal in time 
    """
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
    """
    Creates a canvas for plotting a signal in frequency
    """
    def __init__(self, axis, spacing, f_range_low, f_range_high):
        self.axis = axis
        self.T = spacing
        self.f_range_low = f_range_low
        self.f_range_high = f_range_high * 10

        self.lineplot, = axis.plot([], [], "b-")
        self.axis.grid()
        self.axis.set_autoscaley_on(True)
        
    def plot(self, time, signal):
        freqs = np.fft.fftfreq(1000, self.T)
        freqs = freqs[self.f_range_low:self.f_range_high]

        fourier = np.abs(np.fft.fft(signal))
        fourier = 2.0/1e3 * np.abs(fourier[:1000//2])[self.f_range_low:self.f_range_high]

        self.lineplot.set_data(freqs, fourier)
        self.axis.relim(); self.axis.autoscale_view() # rescale the y-axis
 
class SpectrogramPlot:
    """
    Creates a canvas for plotting the spectrogram of a signal
    """
    def __init__(self, fig, axis, spacing, f_range_low, f_range_high):
        self.fig = fig
        self.axis = axis
        self.T = spacing
        self.f_range_high = f_range_high
        self.f_range_low = f_range_low

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

        self.axis.set(title='Spectrogram', xlabel='Time', ylabel='Frequency', ylim=(self.f_range_low, self.f_range_high))

        im = self.axis.pcolor(t, f, Zxx)

        if self.cbar:
            self.delete_cbar()
 
        self.colorbar(im)

def get_filter(lowcut, highcut, fs, order=4):
    """
    Return the coeficients of a bandpass butterworth filter
    """
    # Cutoff frequencies expressed as a fraction of the Nyquist frequency
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')

    return b, a

