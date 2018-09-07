import sys

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np
from numpy import cos, pi, sin

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QSizePolicy

from scipy.signal import lfilter

from plots import FourierPlot, SpectrogramPlot, TimePlot, get_filter

from datetime import datetime

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setupUi(self)

        self.show()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1250, 540)
        MainWindow.setWindowIcon(QtGui.QIcon('icons/chip.png'))

        self.allFather = QtWidgets.QWidget(MainWindow)
        self.allFather.setObjectName("allFather")
        
        self.gridLayout = QtWidgets.QGridLayout(self.allFather)
        self.gridLayout.setObjectName("gridLayout")
        
        self.sideBar = QtWidgets.QVBoxLayout()
        self.sideBar.setObjectName("sideBar")
        
        # Play Button
        self.paused = False
        self.playBtn = QtWidgets.QPushButton(self.allFather)
        self.playBtn.setObjectName("playBtn")
        self.playBtn.clicked.connect(self.playPause)
        self.sideBar.addWidget(self.playBtn)
        
        # Record checkbox
        self.record = QtWidgets.QCheckBox(self.allFather)
        self.record.setObjectName("record")
        self.record.stateChanged.connect(self.recordSignal)
        self.sideBar.addWidget(self.record)
        
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.sideBar.addItem(spacerItem)
        
        # Signal Settings
        self.signalSettings = QtWidgets.QGridLayout()
        self.signalSettings.setObjectName("signalSettings")
        self.signalFreq = QtWidgets.QSpinBox(self.allFather)
        self.signalFreq.setValue(4)
        self.signalFreq.setMaximum(99999)
        self.signalFreq.setObjectName("signalFreq")
        self.signalSettings.addWidget(self.signalFreq, 4, 0, 1, 1)
        self.signalBtn = QtWidgets.QPushButton(self.allFather)
        self.signalBtn.setObjectName("signalBtn")
        self.signalBtn.clicked.connect(self.applySignalFreq)
        self.signalSettings.addWidget(self.signalBtn, 5, 0, 1, 1)
        self.label_1 = QtWidgets.QLabel(self.allFather)
        self.label_1.setObjectName("label_1")
        self.signalSettings.addWidget(self.label_1, 3, 0, 1, 1)
        self.sideBar.addLayout(self.signalSettings)
        
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.sideBar.addItem(spacerItem1)
        
        # Filter Settings
        self.filterSettings = QtWidgets.QGridLayout()
        self.filterSettings.setObjectName("filterSettings")
        # Apply Button
        self.filterBtn = QtWidgets.QPushButton(self.allFather)
        self.filterBtn.setObjectName("filterBtn")
        self.filterBtn.clicked.connect(self.applyFilter)
        self.filterSettings.addWidget(self.filterBtn, 3, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.allFather)
        self.label_2.setObjectName("label_2")
        self.filterSettings.addWidget(self.label_2, 1, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lowCut = QtWidgets.QSpinBox(self.allFather)
        self.lowCut.setMaximum(99999)
        self.lowCut.setValue(1)
        self.lowCut.setObjectName("lowCut")
        self.horizontalLayout.addWidget(self.lowCut)
        self.highCut = QtWidgets.QSpinBox(self.allFather)
        self.highCut.setMaximum(99999)
        self.highCut.setValue(32)
        self.highCut.setObjectName("highCut")
        self.horizontalLayout.addWidget(self.highCut)
        self.filterSettings.addLayout(self.horizontalLayout, 2, 0, 1, 1)
        self.sideBar.addLayout(self.filterSettings)
        
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.sideBar.addItem(spacerItem2)
        
        # Spectrogram settings
        self.specgramSettings = QtWidgets.QGridLayout()
        self.specgramSettings.setObjectName("specgramSettings")
        self.label_3 = QtWidgets.QLabel(self.allFather)
        self.label_3.setObjectName("label_3")
        self.specgramSettings.addWidget(self.label_3, 1, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.fLowLim = QtWidgets.QSpinBox(self.allFather)
        self.fLowLim.setMaximum(99999)
        self.fLowLim.setValue(1)
        self.fLowLim.setObjectName("fLowLim")
        self.horizontalLayout_2.addWidget(self.fLowLim)
        self.fHighLim = QtWidgets.QSpinBox(self.allFather)
        self.fHighLim.setMaximum(99999)
        self.fHighLim.setValue(15)
        self.fHighLim.setObjectName("fHighLim")
        self.horizontalLayout_2.addWidget(self.fHighLim)
        self.specgramSettings.addLayout(self.horizontalLayout_2, 2, 0, 1, 1)
        # Apply Button
        self.fRangeBtn = QtWidgets.QPushButton(self.allFather)
        self.fRangeBtn.setObjectName("fRangeBtn")
        self.fRangeBtn.clicked.connect(self.applyFRange)
        self.specgramSettings.addWidget(self.fRangeBtn, 3, 0, 1, 1)
        self.sideBar.addLayout(self.specgramSettings)

        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.sideBar.addItem(spacerItem3)
        
        # Exit Button
        self.exitBtn = QtWidgets.QPushButton(self.allFather)
        self.exitBtn.setObjectName("exitBtn")
        self.exitBtn.clicked.connect(self.exit)
        self.sideBar.addWidget(self.exitBtn)
        
        self.gridLayout.addLayout(self.sideBar, 1, 1, 1, 1)
        
        # Graphs
        self.graphs = PlotCanvas()
        self.gridLayout.addWidget(self.graphs, 1, 0, 1, 1)
        
        MainWindow.setCentralWidget(self.allFather)
        
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1032, 22))
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
        self.record.setText(_translate("MainWindow", "Record"))
        self.signalBtn.setText(_translate("MainWindow", "Apply"))
        self.label_1.setText(_translate("MainWindow", "Signal Frequency"))
        self.filterBtn.setText(_translate("MainWindow", "Apply"))
        self.label_2.setText(_translate("MainWindow", "Filter"))
        self.label_3.setText(_translate("MainWindow", "Frequency Range"))
        self.fRangeBtn.setText(_translate("MainWindow", "Apply"))
        self.exitBtn.setText(_translate("MainWindow", "Exit"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionSave.setText(_translate("MainWindow", "Save "))
        self.actionSave.setToolTip(_translate("MainWindow", "Save signals"))
        self.actionSave.setStatusTip(_translate("MainWindow", "Save signals"))
        self.actionSave.setWhatsThis(_translate("MainWindow", "Save signals"))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionExit.setToolTip(_translate("MainWindow", "Close application"))
        self.actionExit.setStatusTip(_translate("MainWindow", "Close application"))
        self.actionExit.setWhatsThis(_translate("MainWindow", "Close application"))
        self.actionExit.setShortcut(_translate("MainWindow", "Ctrl+Q"))

    def applyFRange(self):
        lowLim = self.fLowLim.value()
        highLim = self.fHighLim.value()

        if(lowLim > highLim or lowLim == highLim):
            print('Wrong arguments! Keeping the default ones')
            return
        
        self.graphs.f_range_low = lowLim
        self.graphs.f_range_high = highLim

        self.graphs.updatePlots()

    def applyFilter(self):
        lowCut = self.lowCut.value()
        highCut = self.highCut.value()

        if(lowCut > highCut or lowCut == highCut):
            print('Wrong arguments! Keeping the default ones')
            return

        self.graphs.filterLowCut = lowCut
        self.graphs.filterHighCut = highCut

        self.graphs.updateFilter()
        
    def applySignalFreq(self):
        freq = self.signalFreq.value()

        self.graphs.frequency = freq

    def playPause(self):
        if(self.paused):
            timer.start(0.01)
            self.paused = False
        else:
            timer.stop()
            self.paused = True
    
    def recordSignal(self):
        if(self.record.isChecked()):

            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    
            fileName = 'recordings/input_signal_%s.txt' % now
            self.graphs.openFile(fileName)
            
            self.graphs.record = True
        else:
            self.graphs.record = False
            
            self.graphs.closeFile()

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

        self.setParams()

    def setParams(self):
        self.f_range_low = 1
        self.f_range_high = 15

        self.frequency = 4
        
        self.k = 10
        self.T = 0.01

        self.filterLowCut = 1
        self.filterHighCut = 32

        self.record = False
        
        self.startSignal()

    def startSignal(self):
        self.b, self.a = get_filter(self.filterLowCut, self.filterHighCut, fs=int(1.0 / self.T), order=6)

        start_time = np.arange(0, 10, 0.01)
        start_signal = self.signal(start_time)
        filtered_start_signal = lfilter(self.b, self.a, start_signal)

        self.noisyPlot = TimePlot(self.axes[0, 0])
        self.noisyPlot.add_array(start_time, start_signal)

        self.filteredPlot = TimePlot(self.axes[1, 0])
        self.filteredPlot.add_array(start_time, filtered_start_signal)

        self.fourierPlot = FourierPlot(self.axes[1, 1], self.T, self.f_range_low, self.f_range_high)
        self.fourierPlot.plot(self.filteredPlot.x_axis, self.filteredPlot.y_axis)

        self.specgramPlot = SpectrogramPlot(self.fig, self.axes[0, 1], self.T, self.f_range_low, self.f_range_high)
        self.specgramPlot.plot(self.filteredPlot.x_axis, self.filteredPlot.y_axis)

    def plot(self):
        self.k = round(self.k + self.T, 2)
        s = self.signal(self.k)

        if(self.record):
            self.file.write('({}, {})\n'.format(self.k, s))

        self.noisyPlot.add_point(self.k, s)

        self.filteredPlot.add_array(self.noisyPlot.x_axis, lfilter(self.b, self.a, self.noisyPlot.y_axis))

        self.specgramPlot.plot(self.filteredPlot.x_axis, self.filteredPlot.y_axis)

        self.fourierPlot.plot(self.filteredPlot.x_axis, self.filteredPlot.y_axis)

        self.draw()
        
    def updatePlots(self):
        self.axes[0, 1].cla()
        self.specgramPlot.delete_cbar()
        self.specgramPlot = SpectrogramPlot(self.fig, self.axes[0, 1], self.T, self.f_range_low, self.f_range_high)
        
        self.axes[1, 1].cla()
        self.fourierPlot = FourierPlot(self.axes[1, 1], self.T, self.f_range_low, self.f_range_high)
        self.axes[1, 1].set(title='FFT', xlabel='Frequency', xlim=(self.f_range_low, self.f_range_high))
    
    def updateFilter(self):
        self.b, self.a = get_filter(self.filterLowCut, self.filterHighCut, fs=int(1.0 / self.T), order=6)

    def openFile(self, fileName):
        self.file = open(fileName, 'a')

    def closeFile(self):
        self.file.close()

    def signal(self, time):
        noise = 2 * sin(2 * pi * 60 * time) 
        noise += 3.3 * cos(2 * pi * 45 * time)
        noise += 3.3 * cos(2 * pi * 50 * time)
        return 5 * sin(2 * pi * self.frequency * time) + noise

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MainWindow()

    def update():
        ui.graphs.plot()

    global timer
    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(0.01)


    sys.exit(app.exec_())
