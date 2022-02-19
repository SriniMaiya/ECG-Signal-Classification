"""
Note:
This Code has been Writen by XXX and XXX students at Th-Rosenheim University
The purpose of the Code is education and it is open source license
Course: Digital Signal Processing and Machine learning
Date : 14.Feb.2022
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow
from GuiControlCommand import *
import pyqtgraph as pg
import sys


class Ui_SignalProcessing(QMainWindow):

    def __init__(self):
        """ Global Variable in GUI Loop """
        super(Ui_SignalProcessing, self).__init__()
        self.setWindowTitle("Signal Processing and Machine Learning")

        ''' Global GUi Variables '''
        self.FilePath = "File Path"
        self.sig_ARR = [0] * 1000
        self.sig_CHF = [0] * 1000
        self.sig_NSR = [0] * 1000

        self.imgOne = pg.ImageItem()
        self.imgTwo = pg.ImageItem()

        self.time = [0] * 1000

        self.cutoff = 1
        self.fs = 128

        # Plot global variable
        self.redPen = pg.mkPen((255, 0, 0))
        self.history = None
        self.setupUi()


    def setupUi(self):  # SignalProcessing

        
        ''' define Windows Size: It can be move to other section as well base on u r need'''
        SignalProcessing.setObjectName("SignalProcessing")
        SignalProcessing.resize(1122, 669)
        
        self.centralwidget = QtWidgets.QWidget()  # SignalProcessing
        self.centralwidget.setObjectName("centralwidget")

        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 10, 881, 601))
        self.tabWidget.setObjectName("tabWidget")

        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")

        self.widget_4 = pg.PlotWidget(self.tab)                         #cwt
        self.widget_4.setGeometry(QtCore.QRect(10, 10, 311, 271)) 
        self.widget_4.setStyleSheet("background-color: blue")
        self.widget_4.setObjectName("widget_4")
        self.widget_4.addItem(self.imgOne)

        self.widget_2 = pg.PlotWidget(self.tab)                     
        self.widget_2.setGeometry(QtCore.QRect(10, 320, 311, 261))
        self.widget_2.setStyleSheet("background-color: red")
        self.widget_2.setObjectName("widget_2")
        self.firstSignalTwo = self.widget_2.plot(self.time, self.sig_CHF, pen= self.redPen)


        self.widget = pg.PlotWidget(self.tab)
        self.widget.setGeometry(QtCore.QRect(350, 320, 311, 251))
        self.widget.setStyleSheet("background-color: green")
        self.widget.setObjectName("widget")
        self.firstSignal = self.widget.plot(self.time, self.sig_NSR, pen= self.redPen)


        self.widget_3 = pg.PlotWidget(self.tab)
        self.widget_3.setGeometry(QtCore.QRect(350, 10, 311, 271))
        self.widget_3.setStyleSheet("background-color: white")
        self.widget_3.setObjectName("widget_3")
        self.widget_3.addItem(self.imgTwo)

        
        self.btnLoadData = QtWidgets.QPushButton(self.tab)
        self.btnLoadData.setGeometry(QtCore.QRect(680, 10, 151, 51))
        self.btnLoadData.setObjectName("btnLoadData")

        self.btnPlotRnd = QtWidgets.QPushButton(self.tab)
        self.btnPlotRnd.setGeometry(QtCore.QRect(680, 210, 151, 51))
        self.btnPlotRnd.setObjectName("btnPlotRnd")

        listSignal = ['ARR', 'CHF', 'NSR']
        self.comboBox = QtWidgets.QComboBox(self.tab)
        self.comboBox.setGeometry(QtCore.QRect(680, 270, 151, 51))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItems(listSignal)

        self.label = QtWidgets.QLabel(self.tab)
        self.label.setGeometry(QtCore.QRect(680, 90, 131, 31))
        self.label.setObjectName("label")

        self.txtLenSignal = QtWidgets.QTextEdit(self.tab)
        self.txtLenSignal.setGeometry(QtCore.QRect(680, 130, 60, 30))
        self.txtLenSignal.setObjectName("txtLenSignal")
        self.txtLenSignal.append("0")

        self.txtLenSignalEnd = QtWidgets.QTextEdit(self.tab)
        self.txtLenSignalEnd.setGeometry(QtCore.QRect(800, 130, 60, 30))
        self.txtLenSignalEnd.setObjectName("txtLenSignal")
        self.txtLenSignalEnd.append("1000")

        self.tabWidget.addTab(self.tab, "")

        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")

        self.btnTrain = QtWidgets.QPushButton(self.tab_2)
        self.btnTrain.setGeometry(QtCore.QRect(30, 30, 113, 32))
        self.btnTrain.setObjectName("btnTrain")

        listLr = ["0.0001", "0.0003", "0.001", "0.003", "0.01", "0.03"]
        self.QComboBoxRate = QtWidgets.QComboBox(self.tab_2)
        self.QComboBoxRate.setGeometry(QtCore.QRect(440, 30, 113, 32))
        self.QComboBoxRate.setObjectName("QComboBoxRate")
        self.QComboBoxRate.addItems(listLr)

        self.labelTrain = QtWidgets.QLabel(self.tab_2)
        self.labelTrain.setGeometry(QtCore.QRect(40, 10, 121, 16))
        self.labelTrain.setObjectName("labelTrain")

        networkType = ["AlexNet", "GoogLeNet", "SqueezeNet"]
        self.comboBox_2 = QtWidgets.QComboBox(self.tab_2)
        self.comboBox_2.setGeometry(QtCore.QRect(650, 60, 211, 41))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItems(networkType)

        self.labelPredict_2 = QtWidgets.QLabel(self.tab_2)
        self.labelPredict_2.setGeometry(QtCore.QRect(660, 40, 121, 16))
        self.labelPredict_2.setObjectName("labelPredict_2")

        self.widget_5 = QtWidgets.QWidget(self.tab_2)
        self.widget_5.setGeometry(QtCore.QRect(30, 120, 611, 301))
        self.widget_5.setStyleSheet("background-color: green")
        self.widget_5.setObjectName("widget_5")

        self.tabWidget.addTab(self.tab_2, "")

        ''' Tab prediction '''
        self.tabPrediction = QtWidgets.QWidget()
        self.tabPrediction.setObjectName("tabPrediction")

        self.labelPredict = QtWidgets.QLabel(self.tabPrediction)
        self.labelPredict.setGeometry(QtCore.QRect(10, 30, 121, 16))
        self.labelPredict.setObjectName("labelPredict")

        self.btnPredict = QtWidgets.QPushButton(self.tabPrediction)
        self.btnPredict.setGeometry(QtCore.QRect(10, 40, 113, 32))
        self.btnPredict.setObjectName("btnPredict")


        self.tabWidget.addTab(self.tabPrediction, "")

        SignalProcessing.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(SignalProcessing)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1122, 22))
        self.menubar.setObjectName("menubar")

        SignalProcessing.setMenuBar(self.menubar)

        self.statusbar = QtWidgets.QStatusBar(SignalProcessing)
        self.statusbar.setObjectName("statusbar")

        SignalProcessing.setStatusBar(self.statusbar)

        self.retranslateUi(SignalProcessing)
        self.tabWidget.setCurrentIndex(0)

        QtCore.QMetaObject.connectSlotsByName(SignalProcessing)

        self.connect()

    def retranslateUi(self, SignalProcessing):
        _translate = QtCore.QCoreApplication.translate
        SignalProcessing.setWindowTitle(_translate("SignalProcessing", "SignalProcessing"))
        self.btnLoadData.setText(_translate("SignalProcessing", "Load Data"))
        self.btnPlotRnd.setText(_translate("SignalProcessing", "Plot Random Signal"))
        self.label.setText(_translate("SignalProcessing", "Signal Range:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab),
                                  _translate("SignalProcessing", "Signal Pre-Processing Observation"))
        self.btnTrain.setText(_translate("SignalProcessing", "Train "))
        self.btnPredict.setText(_translate("SignalProcessing", "Test"))
        #self.QComboBoxRate.setText(_translate("SignalProcessing", "Unknown"))
        self.labelTrain.setText(_translate("SignalProcessing", "None"))
        self.labelPredict.setText(_translate("SignalProcessing", "None"))
        self.labelPredict_2.setText(_translate("SignalProcessing", "Neural Network:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("SignalProcessing", "Training Signal"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabPrediction), _translate("SignalProcessing", "Prediction"))

    def connect(self):
        ''' Define Signal and SLot fo GUI Connection '''
        self.btnLoadData.clicked.connect(self.loadData)
        self.btnTrain.clicked.connect(self.slotTrainNetwork)
        self.btnPlotRnd.clicked.connect(self.plotSignal)

    # Slots are defined here
    def loadData(self):
        LoadECGData(self)

    def plotSignal(self):
        plot_signal_rnd(self)

    def slotTrainNetwork(self):
        trainNetwork(self)

    # Test Function
    def test_func(self):
        print(self.FilePath)
        print(self.ECGData)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    SignalProcessing = QtWidgets.QMainWindow()
    ui = Ui_SignalProcessing()
    #ui.setupUi() # It is allready defined !
    SignalProcessing.show()
    sys.exit(app.exec_())