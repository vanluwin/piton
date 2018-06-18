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
        
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        
        # Play Button
        self.playBtn = QtWidgets.QPushButton(self.centralwidget)
        self.playBtn.setObjectName("playBtn")
        self.verticalLayout.addWidget(self.playBtn)
        
        # Record checkbox
        self.record = QtWidgets.QCheckBox(self.centralwidget)
        self.record.setObjectName("record")
        self.record.stateChanged.connect(self.recordSignal)
        self.verticalLayout.addWidget(self.record)
        
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)

        # Signal Settings
        self.signalSettings = QtWidgets.QGridLayout()
        self.signalSettings.setObjectName('signalSettings')
        

        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        
        # Filter Settings
        self.filterSettings = QtWidgets.QGridLayout()
        self.filterSettings.setObjectName("filterSettings")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.lowCut = QtWidgets.QSpinBox(self.centralwidget)
        self.lowCut.setObjectName("lowCut")
        self.horizontalLayout.addWidget(self.lowCut)
        self.highCut = QtWidgets.QSpinBox(self.centralwidget)
        self.highCut.setObjectName("highCut")
        self.horizontalLayout.addWidget(self.highCut)
        self.filterSettings.addLayout(self.horizontalLayout, 2, 0, 1, 1)
        # Apply Button
        self.filterBtn = QtWidgets.QPushButton(self.centralwidget)
        self.filterBtn.setObjectName("filterBtn")
        self.filterBtn.clicked.connect(self.applyFilter)
        self.filterSettings.addWidget(self.filterBtn, 3, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.filterSettings.addWidget(self.label, 1, 0, 1, 1)
        self.verticalLayout.addLayout(self.filterSettings)
        
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        
        self.specgramSettings = QtWidgets.QGridLayout()
        self.specgramSettings.setObjectName("specgramSettings")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.specgramSettings.addWidget(self.label_2, 1, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.fLowLim = QtWidgets.QSpinBox(self.centralwidget)
        self.fLowLim.setObjectName("fLowLim")
        self.horizontalLayout_2.addWidget(self.fLowLim)
        self.fHighLim = QtWidgets.QSpinBox(self.centralwidget)
        self.fHighLim.setObjectName("fHighLim")
        self.horizontalLayout_2.addWidget(self.fHighLim)
        self.specgramSettings.addLayout(self.horizontalLayout_2, 2, 0, 1, 1)
        # Apply Button
        self.fRangeBtn = QtWidgets.QPushButton(self.centralwidget)
        self.fRangeBtn.setObjectName("fRangeBtn")
        self.fRangeBtn.clicked.connect(self.applyFRange)
        self.specgramSettings.addWidget(self.fRangeBtn, 3, 0, 1, 1)
        self.verticalLayout.addLayout(self.specgramSettings)
        
        spacerItem3 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        
        # Exit Button
        self.exitBtn = QtWidgets.QPushButton(self.centralwidget)
        self.exitBtn.setObjectName("exitBtn")
        self.exitBtn.clicked.connect(self.exit)
        self.verticalLayout.addWidget(self.exitBtn)
        
        self.gridLayout.addLayout(self.verticalLayout, 1, 1, 1, 1)
        
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout.addWidget(self.graphicsView, 1, 0, 1, 1)
        
        MainWindow.setCentralWidget(self.centralwidget)
        
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
        self.filterBtn.setText(_translate("MainWindow", "Apply"))
        self.label.setText(_translate("MainWindow", "Filter"))
        self.label_2.setText(_translate("MainWindow", "Frequency Range"))
        self.fRangeBtn.setText(_translate("MainWindow", "Apply"))
        self.exitBtn.setText(_translate("MainWindow", "Exit"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionSave.setText(_translate("MainWindow", "Save "))
        self.actionSave.setToolTip(_translate("MainWindow", "Save signals"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionExit.setToolTip(_translate("MainWindow", "Close application"))

    def applyFRange(self):
        lowLim = self.fLowLim.value()
        highLim = self.fHighLim.value()

        if(lowLim > highLim or lowLim == highLim):
            print('Wrong arguments! Keeping the default ones')
            return
        
        print('Change the frequency range')

    def applyFilter(self):
        lowCut = self.lowCut.value()
        highCut = self.highCut.value()

        if(lowCut > highCut or lowCut == highCut):
            print('Wrong arguments! Keeping the default ones')
            return

        print('Change filter')

    def recordSignal(self):
        print('Record input signal')

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = MainWindow()
    sys.exit(app.exec_())
