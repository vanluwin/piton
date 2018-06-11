import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from plots import TimePlot, FourierPlot, SpectrogramPlot, get_filter

import numpy as np
from numpy import sin, cos, pi
from scipy.signal import lfilter

import random

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setupUi(self)

        self.show()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1250, 540)
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        
        self.playBtn = QtWidgets.QPushButton(self.centralwidget)
        self.playBtn.setObjectName("playBtn")
        self.playBtn.clicked.connect(self.playPause)
        self.horizontalLayout_3.addWidget(self.playBtn)
        
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        
        self.signal1 = QtWidgets.QPushButton(self.centralwidget)
        self.signal1.setObjectName("signal1")
        self.signal1.clicked.connect(self.sig1Btn)
        self.horizontalLayout_3.addWidget(self.signal1)
        
        self.signal2 = QtWidgets.QPushButton(self.centralwidget)
        self.signal2.setObjectName("signal2")
        self.signal2.clicked.connect(self.sig2Btn)
        self.horizontalLayout_3.addWidget(self.signal2)
        
        self.signal3 = QtWidgets.QPushButton(self.centralwidget)
        self.signal3.setObjectName("signal3")
        self.signal3.clicked.connect(self.sig3Btn)
        self.horizontalLayout_3.addWidget(self.signal3)
        
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem1)
        
        self.exitBtn = QtWidgets.QPushButton(self.centralwidget)
        self.exitBtn.setObjectName("exitBtn")
        self.exitBtn.clicked.connect(self.exit)
        self.horizontalLayout_3.addWidget(self.exitBtn)
        self.gridLayout.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)
        
        self.plot = PlotCanvas()
        self.gridLayout.addWidget(self.plot, 1, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 913, 22))
        self.menubar.setObjectName("menubar")
        
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionExit.triggered.connect(self.exit)
        
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Singals Magic"))
        self.playBtn.setText(_translate("MainWindow", "Start/Stop"))
        self.signal1.setText(_translate("MainWindow", "1Hz Signal"))
        self.signal2.setText(_translate("MainWindow", "2Hz Signal"))
        self.signal3.setText(_translate("MainWindow", "3Hz Signal"))
        self.exitBtn.setText(_translate("MainWindow", "Exit"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionSave.setText(_translate("MainWindow", "Save "))
        self.actionSave.setToolTip(_translate("MainWindow", "Save signals"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionExit.setToolTip(_translate("MainWindow", "Close application"))

    def playPause(self):
        print('Play/Pause')
    
    def sig1Btn(self):
        self.plot.changeSignal(1)
    
    def sig2Btn(self):
        self.plot.changeSignal(2)

    def sig3Btn(self):
        self.plot.changeSignal(3)

    def exit(self):
        choice = QtWidgets.QMessageBox.question(self, 'Exit App', 'Are you sure ?', QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)

        if choice == QtWidgets.QMessageBox.Yes:
            sys.exit()

class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig, self.axes = plt.subplots(nrows=2, ncols=2, figsize=(13, 5))
        self.fig.tight_layout(h_pad=4.0)
        plt.subplots_adjust(top=0.934, bottom=0.109, left=0.043, right=0.962, hspace=0.435, wspace=0.122)

        self.axes[0, 0].set(title='Noisy signal', xlabel='Time')
        self.axes[1, 0].set(title='Filtered Signal', xlabel='Time')
        
        self.axes[0, 1].set(title='Spectrogram', xlabel='Time')
        self.axes[1, 1].set(title='FFT', xlabel='Frequency')

        FigureCanvas.__init__(self, self.fig)
        
        FigureCanvas.setSizePolicy(
            self,
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )

        FigureCanvas.updateGeometry(self)

        self.initPlots()
    
    def initPlots(self):
        
        self.f_range = 15
        self.frequency = 4
        self.k = 10
        self.T = 0.01
        self.b, self.a = get_filter(1, 32, fs=int(1.0 / self.T), order=6)

        start_time = np.arange(0, 10, 0.01)
        start_signal = f(start_time)
        filtered_start_signal = lfilter(self.b, self.a, start_signal)

        self.noisyPlot = TimePlot(self.axes[0, 0])
        self.noisyPlot.add_array(start_time, start_signal)

        self.filteredPlot = TimePlot(self.axes[1, 0])
        self.filteredPlot.add_array(start_time, filtered_start_signal)

        self.fourierPlot = FourierPlot(self.axes[1, 1], self.T, self.f_range)
        self.fourierPlot.plot(self.filteredPlot.x_axis, self.filteredPlot.y_axis)

        self.specgramPlot = SpectrogramPlot(self.fig, self.axes[0, 1], self.T, self.f_range)
        self.specgramPlot.plot(self.filteredPlot.x_axis, self.filteredPlot.y_axis)

    def plot(self):
        self.k = round(self.k + self.T, 2)
        s = self.signal(self.k)

        self.noisyPlot.add_point(self.k, s)

        self.filteredPlot.add_array(self.noisyPlot.x_axis, lfilter(self.b, self.a, self.noisyPlot.y_axis))

        self.specgramPlot.plot(self.filteredPlot.x_axis, self.filteredPlot.y_axis)

        self.fourierPlot.plot(self.filteredPlot.x_axis, self.filteredPlot.y_axis)

        self.draw()
    
    def signal(self, time):
        noise = 2 * sin(2 * pi * 60 * time) 
        noise += 3.3 * cos(2 * pi * 45 * time)
        noise += 3.3 * cos(2 * pi * 50 * time)
        return 7 * sin(2 * pi * self.frequency * time) + noise

    def changeSignal(self, frequency):
        if(frequency == 1):
            self.frequency = 1
        elif(frequency == 2):
            self.frequency = 2
        elif(frequency == 3):
            self.frequency = 3

def f(k):
    noise = 2 * sin(2 * pi * 60 * k) 
    noise += 3.3 * cos(2 * pi * 45 * k)
    noise += 3.3 * cos(2 * pi * 50 * k)
    return 4 * sin(2 * pi * 3 * k) + noise 

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MainWindow()

    def update():
        ui.plot.plot()

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(0.01)
        
    sys.exit(app.exec_())
