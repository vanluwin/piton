# -*- coding: utf-8 -*-
from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
from numpy import sin, pi
import pyqtgraph as pg
import sys
from scipy.fftpack import fft

import time

start = time.time()

class Plot2D():
    def __init__(self):
        self.traces = dict()

        pg.setConfigOptions(antialias=True)

        self.app = QtGui.QApplication(sys.argv)
        self.win = pg.GraphicsWindow(title='Basic plotting')
        self.win.resize(1000, 600)
        self.win.setWindowTitle('Plotting')

        self.waveform = self.win.addPlot(title='Wave', row=1, col=1)
        #self.spectrum = self.win.addPlot(title='Sepctrum', row=2, col=1)

        self.RATE = 100
        self.N = 1024 

        self.t = np.arange(0,3.0, 3 / self.N)
        self.phase = 0

        self.f = np.linspace(0, 300, self.N / 2)

    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def set_plotData(self, name, data_x, data_y):
        if name in self.traces:
            self.traces[name].setData(data_x, data_y)
        else:
            if name == 'waveform':
                self.traces[name] = self.waveform.plot(pen='c', width=3)
            if name == 'spectrum':
                self.traces[name] = self.spectrum.plot(pen='m', width=3)
                self.spectrum.setLogMode(x=True, y=True)

    def update(self):
        s = sin(2 * pi * 1 * self.t + self.phase) + sin(2 * pi * 10 * self.t + self.phase)

        self.phase += 0.01

        self.set_plotData(name='waveform', data_x=self.t, data_y=s)

        """ sp_data = fft(s)
        sp_data = np.abs(sp_data)
        self.set_plotData(name='spectrum', data_x=self.f, data_y=sp_data) """


         
    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(10)
        self.start()

if __name__ == '__main__':
    p = Plot2D()
    p.animation()