import sys
from PyQt4 import QtGui, QtCore

<<<<<<< HEAD
class Window(object):
=======
class Window(QtGui.QMainWindow):
>>>>>>> c935633522c9b5a0dac2bcf453178c81f819fdda
    
    def __init__(self):
        super(Window, self).__init__()

        self.setGeometry(50, 50, 500, 300)
        self.setWindowTitle('PyQT Magic')
        self.setWindowIcon(QtGui.QIcon('icon.png'))

        extractAction = QtGui.QAction('Exit', self)
        extractAction.setShortcut('Ctrl+Q')
        extractAction.setStatusTip('Leave the App')
        extractAction.triggered.connect(self.close_app)

        openEditor = QtGui.QAction('Editor', self)
        openEditor.setShortcut('Ctrl+E')
        openEditor.setStatusTip('Open Editor')
        openEditor.triggered.connect(self.editor)

        openFile = QtGui.QAction('Open File', self)
        openFile.setShortcut('Ctrl+o')
        openFile.setStatusTip('Open File')
        openFile.triggered.connect(self.openFile)

        saveFile = QtGui.QAction('Save File', self)
        saveFile.setShortcut('Ctrl+s')
        saveFile.setStatusTip('Save File')
        saveFile.triggered.connect(self.saveFile)

        self.statusBar()

        mainMenu = self.menuBar()

        fileMenu = mainMenu.addMenu('File')
        fileMenu.addAction(openFile)
        fileMenu.addAction(saveFile)
        fileMenu.addAction(extractAction)

        editorMenu = mainMenu.addMenu('Editor')
        editorMenu.addAction(openEditor)
        
        self.home()

    def home(self):
        btn = QtGui.QPushButton("Quit", self)

        btn.clicked.connect(self.close_app)

        btn.resize(btn.minimumSizeHint())
        btn.move(0, 100)

        extractAction = QtGui.QAction(QtGui.QIcon('icon.png'), 'Exit', self)
        extractAction.triggered.connect(self.close_app)
        self.toolBar = self.addToolBar('Extration')
        self.toolBar.addAction(extractAction)

        fontChoice = QtGui.QAction('Font', self)
        fontChoice.triggered.connect(self.fontChoice)
        #self.toolBar = self.addToolBar('Font')
        self.toolBar.addAction(fontChoice)

        color = QtGui.QColor(0, 0, 0)

        fontColor = QtGui.QAction('Font bg Color', self)
        fontColor.triggered.connect(self.colorPicker)
        self.toolBar.addAction(fontColor)

        checkBox = QtGui.QCheckBox('Enlarge', self)
        checkBox.move(0, 70)
        checkBox.stateChanged.connect(self.enlargeWindow)

        self.progress = QtGui.QProgressBar(self)
        self.progress.setGeometry(0, 200, 250, 20)

        self.btn = QtGui.QPushButton('Download', self)
        self.btn.move(0, 220)
        self.btn.clicked.connect(self.download)

        self.styleChoice = QtGui.QLabel(self.style().objectName(), self)

        comboBox = QtGui.QComboBox(self)
        comboBox.addItem('motif')
        comboBox.addItem('Windows')
        comboBox.addItem('cde')
        comboBox.addItem('Plastique')
        comboBox.addItem('Cleanlooks')

        comboBox.move(100, 250)
        self.styleChoice.move(100, 150)
        comboBox.activated[str].connect(self.changeStyle)
        
        calendar = QtGui.QCalendarWidget(self)
        calendar.move(500, 200)
        calendar.resize(200, 200)

        
        self.show()
    
    def openFile(self):
        name = QtGui.QFileDialog.getOpenFileName(self, 'Open File')
        file = open(name, 'r')

        self.editor()

        with file:
            text = file.read()
            self.textEdit.setText(text)

    def saveFile(self):
        name = QtGui.QFileDialog.getSaveFileName(self, 'Save File')
        file = open(name, 'w')
        text = self.textEdit.toPlainText()
        file.write(text)
        file.close()
<<<<<<< HEAD
   
=======
>>>>>>> c935633522c9b5a0dac2bcf453178c81f819fdda
    def editor(self):
        self.textEdit = QtGui.QTextEdit()
        self.setCentralWidget(self.textEdit)

    def colorPicker(self):
        color = QtGui.QColorDialog.getColor()

        self.styleChoice.setStyleSheet('QWidget { background-color: %s}' % color.name())

    def fontChoice(self):
        font, valid = QtGui.QFontDialog.getFont()

        if valid:
            self.styleChoice.setFont(font)

    def changeStyle(self, text):
        self.styleChoice.setText(text)
        QtGui.QApplication.setStyle(QtGui.QStyleFactory.create(text))

    def download(self):
        self.completed = 0

        while self.completed < 100:
            self.completed += 0.001
            self.progress.setValue(self.completed)

    def enlargeWindow(self, state):
        if state == QtCore.Qt.Checked:
            self.setGeometry(50, 50, 1000, 600)
        else: 
            self.setGeometry(50, 50, 500, 300)

    def close_app(self):
        choice = QtGui.QMessageBox.question(self, 'Get Out', 'Are you sure ?', QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)

        if choice == QtGui.QMessageBox.Yes:
            sys.exit()
 


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())