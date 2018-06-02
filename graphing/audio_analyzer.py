"""
Install pyaudio: pip intall pyaudio
In case of error you migth need: sudo apt-get install portaudio19-dev
"""
from collections import deque

import numpy as np
from numpy import sin, cos, pi

from scipy.signal import stft, butter, lfilter
from scipy.fftpack import fft

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pyaudio
import struct

class WaveformPlot(object):
    def __init__(self, axis):
        self.axis = axis

        # stream constants
        self.CHUNK = 1024 * 2
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.pause = False

        # stream object
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            frames_per_buffer=self.CHUNK,
        )

        self.x_axis = np.arange(0, 2 * self.CHUNK, 2)
        
        self.lineplot, = axis.plot(self.x_axis, np.random.rand(2048), "b-")
        self.axis.set_autoscaley_on(True)

    def draw(self):
        data = self.stream.read(self.CHUNK)
        data_int = struct.unpack(str(2 * self.CHUNK) + 'B', data)
        self.signal = np.array(data_int, dtype='b')[::2] + 128

        self.lineplot.set_ydata(self.signal)

        self.axis.set_xlim(self.x_axis[0], self.x_axis[-1] + 1e-15)


class FilteredPlot:
    def __init__(self, axis, max_entries = 1000):
        self.axis = axis

        self.x_axis = deque()
        self.y_axis = deque()

        self.lineplot, = axis.plot([], [], "b-")
        self.axis.grid()
        self.axis.set_autoscaley_on(True)
    
    def plot_array(self, x, y):
        self.x_axis = deque(x)
        self.y_axis = deque(y)

        self.lineplot.set_data(self.x_axis, self.y_axis)

        self.axis.set_xlim(self.x_axis[0], self.x_axis[-1] + 1e-15)
        self.axis.relim(); self.axis.autoscale_view() # rescale the y-axis

class FourierPlot:
    def __init__(self, axis, fs):
        self.axis = axis
        self.fs = fs

        self.x_axis = np.linspace(0, 44100, 2048)

        self.lineplot, = axis.semilogx(self.x_axis, np.random.rand(2048), "b-")
        self.axis.grid()
        self.axis.set_autoscaley_on(True)
        

    def plot(self, time, signal):
        
        fourier = fft(signal)

        fourier = np.abs(fourier[0:2048]) / (128 * 2048)

        self.lineplot.set_ydata(fourier)
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

def main():
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 5))
    fig.tight_layout(h_pad=4.0)
    plt.subplots_adjust(top=0.929, bottom=0.117, left=0.042, right=0.964, hspace=0.48, wspace=0.124)

    axes[0, 0].set(title='Audio Waveform', xlabel='Samples', yticks=[0, 128, 255], ylim=(0, 255))
    axes[1, 0].set(title='Filtered waveform', xlabel='Time')
    axes[1, 1].set(title='FFT', xlabel='Frequency', ylabel='Amplitute')

    b, a = get_filter(100, 600, fs=44100, order=4)

    waveform = WaveformPlot(axes[0, 0])

    filteredPlot = FilteredPlot(axes[1,0])

    specgramPlot = SpectrogramPlot(fig, axes[0, 1], 1/44100, 1000)

    fourierPlot = FourierPlot(axes[1, 1], 44100)

    while True:

        waveform.draw()

        filteredPlot.plot_array(waveform.x_axis, lfilter(b, a, waveform.signal))

        specgramPlot.plot(filteredPlot.x_axis, filteredPlot.y_axis)

        fourierPlot.plot(filteredPlot.x_axis, filteredPlot.y_axis)


        plt.pause(0.1)
    
    plt.show()


if __name__ == "__main__": 
    main()