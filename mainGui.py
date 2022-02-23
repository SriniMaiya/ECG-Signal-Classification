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

        self.trainAcc = [0]*20
        self.valAcc = [0]*20
        self.trainLoss = [0]*20
        self.valLoss = [0]*20

        self.cutoff = 1
        self.fs = 128

        # Plot global variable
        self.redPen = pg.mkPen((255, 0, 0))
        self.greenPen = pg.mkPen((0,255,0))

        self.model = None
        self.weights = None
        self.best_weights = None
        self.history = None

        self.setupUi()


    def setupUi(self):  # SignalProcessing

        
        ''' define Windows Size: It can be move to other section as well base on u r need'''
        pg.setConfigOptions(antialias=True)
        SignalProcessing.setObjectName("SignalProcessing")
        SignalProcessing.resize(1400, 920)
        
        self.centralwidget = QtWidgets.QWidget()  # SignalProcessing
        self.centralwidget.setObjectName("centralwidget")

        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 10, 1380, 900))
        self.tabWidget.setObjectName("tabWidget")

        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")

        self.cwt = pg.PlotWidget(self.tab)                         #cwt
        self.cwt.setGeometry(QtCore.QRect(30, 10, 380, 380)) 
        self.cwt.setStyleSheet("background-color: blue")
        self.cwt.setObjectName("cwt")
        self.cwt.addItem(self.imgOne)
        
        self.cwtf = pg.PlotWidget(self.tab)                         #cwtf
        self.cwtf.setGeometry(QtCore.QRect(500, 10, 380, 380))
        self.cwtf.setStyleSheet("background-color: white")
        self.cwtf.setObjectName("cwtf")
        self.cwtf.addItem(self.imgTwo)

        self.sig = pg.PlotWidget(self.tab)                           #sig
        self.sig.setGeometry(QtCore.QRect(30, 430, 380, 380))
        self.sig.setStyleSheet("background-color: green")
        self.sig.setObjectName("sig")
        self.firstSignal = self.sig.plot(self.time, self.sig_NSR, pen= self.redPen)

        self.sigf = pg.PlotWidget(self.tab)                         #sigf
        self.sigf.setGeometry(QtCore.QRect(500, 430, 380, 380))
        self.sigf.setStyleSheet("background-color: red")
        self.sigf.setObjectName("sigf")
        self.firstSignalTwo = self.sigf.plot(self.time, self.sig_CHF, pen= self.greenPen)
        
        self.btnLoadData = QtWidgets.QPushButton(self.tab)
        self.btnLoadData.setGeometry(QtCore.QRect(950, 20, 151, 51))
        self.btnLoadData.setObjectName("btnLoadData")

        self.labelSigLength = QtWidgets.QLabel(self.tab)
        self.labelSigLength.setGeometry(QtCore.QRect(950, 90, 131, 31))
        self.labelSigLength.setObjectName("labelSigLength")

        self.txtSigStart = QtWidgets.QTextEdit(self.tab)
        self.txtSigStart.setGeometry(QtCore.QRect(950, 130, 60, 30))
        self.txtSigStart.setObjectName("txtSigStart")
        self.txtSigStart.append("0")

        self.txtSigEnd = QtWidgets.QTextEdit(self.tab)
        self.txtSigEnd.setGeometry(QtCore.QRect(1050, 130, 60, 30))
        self.txtSigEnd.setObjectName("txtLenSignalEnd")
        self.txtSigEnd.append("1000")

        listSignal = ['ARR', 'CHF', 'NSR']
        self.selectSig = QtWidgets.QComboBox(self.tab)
        self.selectSig.setGeometry(QtCore.QRect(950, 180, 151, 51))
        self.selectSig.setObjectName("selectSig")
        self.selectSig.addItems(listSignal)
        
        self.btnPlotRnd = QtWidgets.QPushButton(self.tab)
        self.btnPlotRnd.setGeometry(QtCore.QRect(950, 270, 151, 51))
        self.btnPlotRnd.setObjectName("btnPlotRnd")

              


        #TAB 2
        self.tabWidget.addTab(self.tab, "")

        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")

        #->
        networkType = ["AlexNet", "ResNet", "SqueezeNet"]
        self.NetworkType = QtWidgets.QComboBox(self.tab_2)
        self.NetworkType.setGeometry(QtCore.QRect(40, 125, 113, 32))
        self.NetworkType.setObjectName("NetworkType")
        self.NetworkType.addItems(networkType)

        self.labelNetworkType = QtWidgets.QLabel(self.tab_2)
        self.labelNetworkType.setGeometry(QtCore.QRect(30, 105, 121, 16))
        self.labelNetworkType.setObjectName("labelNetworkType")
        #->
        self.widgetWayOne = QtWidgets.QWidget(self.tab_2)
        self.widgetWayOne.setGeometry(QtCore.QRect(220, 40, 900, 80))
        self.widgetWayOne.setObjectName("widgetWayOne")
        self.widgetWayOne.setStyleSheet("background-color:#DCDCDC")

        self.widgetWayTwo = QtWidgets.QWidget(self.tab_2)
        self.widgetWayTwo.setGeometry(QtCore.QRect(460, 145, 450, 80))
        self.widgetWayTwo.setObjectName("widgetWayTwo")
        self.widgetWayTwo.setStyleSheet("background-color:#C0C0C0")

        self.lblbatch_size = QtWidgets.QLabel(self.tab_2)
        self.lblbatch_size.setGeometry(QtCore.QRect(240, 50, 121, 16))
        self.lblbatch_size.setObjectName("lblbatch_size")
        self.lblbatch_size.setText("Batch size:")

        listbatch_size = ["16", "32", "64"]
        self.QCombobatch_size = QtWidgets.QComboBox(self.tab_2)
        self.QCombobatch_size.setGeometry(QtCore.QRect(240, 70, 113, 32))
        self.QCombobatch_size.setObjectName("QCombobatch_size")
        self.QCombobatch_size.addItems(listbatch_size)
        #->
        self.lblRate = QtWidgets.QLabel(self.tab_2)
        self.lblRate.setGeometry(QtCore.QRect(410, 50, 121, 16))
        self.lblRate.setObjectName("lblRate")
        self.lblRate.setText("Learning rate:")

        listLr = ["0.0001", "0.0003", "0.001", "0.003", "0.01", "0.03"]
        self.QComboBoxRate = QtWidgets.QComboBox(self.tab_2)
        self.QComboBoxRate.setGeometry(QtCore.QRect(410, 70, 113, 32))
        self.QComboBoxRate.setObjectName("QComboBoxRate")
        self.QComboBoxRate.addItems(listLr)
        #->
        self.lblnum_epochs = QtWidgets.QLabel(self.tab_2)
        self.lblnum_epochs.setGeometry(QtCore.QRect(590, 50, 121, 16))
        self.lblnum_epochs.setObjectName("lblRate")
        self.lblnum_epochs.setText("Epochs:")

        self.txtNum_epochs = QtWidgets.QTextEdit(self.tab_2)
        self.txtNum_epochs.setGeometry(QtCore.QRect(590, 70, 70, 30))
        self.txtNum_epochs.setObjectName("txtNum_epochs")
        self.txtNum_epochs.setText("10")
        #->
        self.btnTrain = QtWidgets.QPushButton(self.tab_2)
        self.btnTrain.setGeometry(QtCore.QRect(720, 70, 113, 32))
        self.btnTrain.setObjectName("btnTrain")

        self.labelTrain = QtWidgets.QLabel(self.tab_2)
        self.labelTrain.setGeometry(QtCore.QRect(720, 40, 120, 32))
        self.labelTrain.setObjectName("labelTrain")
        #->
        self.btnSaveWts = QtWidgets.QPushButton(self.tab_2)
        self.btnSaveWts.setGeometry(QtCore.QRect(870, 70, 113, 32))
        self.btnSaveWts.setObjectName("btnSaveWts")
        self.btnSaveWts.setText("Save")

        self.lblSaveWts = QtWidgets.QLabel(self.tab_2)
        self.lblSaveWts.setGeometry(QtCore.QRect(870, 40, 120, 30))
        self.lblSaveWts.setObjectName("lblSaveWts")
        #->
        self.btnLoadWeights = QtWidgets.QPushButton(self.tab_2)
        self.btnLoadWeights.setGeometry(QtCore.QRect(470, 180, 121,31))
        self.btnLoadWeights.setObjectName("Load weights")

        self.labelLoadWeights = QtWidgets.QLabel(self.tab_2)
        self.labelLoadWeights.setGeometry(QtCore.QRect(470, 152, 220, 25))
        self.labelLoadWeights.setObjectName("labelLoadWeights")
        #->
        self.btnLoadBestWeights = QtWidgets.QPushButton(self.tab_2)
        self.btnLoadBestWeights.setGeometry(QtCore.QRect(740, 180, 121,31))
        self.btnLoadBestWeights.setObjectName("LoadBestweights")

        self.labelLoadBestWeights = QtWidgets.QLabel(self.tab_2)
        self.labelLoadBestWeights.setGeometry(QtCore.QRect(740, 152, 220, 25))
        self.labelLoadBestWeights.setObjectName("labelLoadBestWeights")

        style = {"color":"w", "font-size":"12px"}
        self.widgetPlotAcc = pg.PlotWidget(self.tab_2)
        self.widgetPlotAcc.setBackground("k")
        self.widgetPlotAcc.setGeometry(QtCore.QRect(10, 260, 415, 415))
        self.widgetPlotAcc.setTitle("Accuracy Plot", color="w")
        self.widgetPlotAcc.addLegend(offset = (260,0))
        
        #self.widgetPlotAcc.getPlotItem().setLabel(axis = "left", title="Accuracy ->",**style)
        #self.widgetPlotAcc.getPlotItem().setLabel(axis = "bottom", title="Epochs ->", **style)    #not working
        self.widgetPlotAcc.getPlotItem().showGrid(x=True, y=True, alpha=0.7)
        self.trainAccPlot = self.widgetPlotAcc.plot(np.linspace(1, len(self.trainAcc), len(self.trainAcc)),
                                                         self.trainAcc, pen = self.redPen, name="Train acc"  )
        self.valAccPlot = self.widgetPlotAcc.plot(np.linspace(1, len(self.valAcc), len(self.valAcc)),
                                                         self.valAcc, pen = self.greenPen, name = "Val acc"  )
        self.widgetPlotAcc.getPlotItem().setLabel('bottom', "Number of Epochs")
        self.widgetPlotAcc.getPlotItem().setLabel('left', "Accuracy")
        # self.widgetPlotAcc.getPlotItem().setLabel(axis = "left", title="Accuracy ->",**style)
        # self.widgetPlotAcc.getPlotItem().setLabel(axis = "bottom", title="Epochs ->", **style)
        #self.widgetPlotAcc.getPlotItem().showGrid(x=True, y=True, alpha=0.7)
        # self.widgetPlotAcc.getPlotItem().showLabel('left', show=True)     #not working
        #font = QtGui.QFont()
        #font.setPixelSize(12)
        
        #self.widgetPlotAcc.getPlotItem().getAxis("bottom").tickFont = font
        #self.widgetPlotAcc.getPlotItem().getAxis("bottom").setStyle(tickTextOffset=20)

        #self.widgetPlotAcc.getPlotItem().showLabel('left', show=True)        #not working
        #self.widgetPlotAcc.getPlotItem().showLabel('bottom', show=True)
        # self.widgetPlotAcc.getPlotItem().showLabel('bottom', show=True)

        #->
        self.widgetPlotLoss = pg.PlotWidget(self.tab_2)
        self.widgetPlotLoss.setTitle("Loss plot", color="w")
        self.widgetPlotAcc.setBackground("k")
        # self.widgetPlotLoss.setLabel("left", "Loss->", **style)
        self.widgetPlotLoss.setGeometry(QtCore.QRect(450, 260, 415, 415))
        self.widgetPlotLoss.setObjectName("widgetPlotLoss")
        self.widgetPlotLoss.addLegend(offset = (260,-300))
        # self.widgetPlotLoss.getPlotItem()
        #self.widgetPlotLoss.getPlotItem().setLabel(axis = "left", title="Loss ->",**style)        #not working
        #self.widgetPlotLoss.getPlotItem().setLabel(axis = "bottom", title="Epochs ->", **style)

        self.widgetPlotLoss.getPlotItem().showGrid(x=True, y=True, alpha=0.7)
        self.trainLossPlot = self.widgetPlotLoss.plot(np.linspace(0, len(self.trainLoss), len(self.trainLoss)),
                                                         self.trainLoss, pen = self.redPen , name = "Train loss" )
        self.valLossPlot = self.widgetPlotLoss.plot(np.linspace(0, len(self.valLoss), len(self.valLoss)),
                                                         self.valLoss, pen = self.greenPen  , name="Val loss")

        self.widgetPlotLoss.getPlotItem().setLabel('bottom', "Number of Epochs")
        self.widgetPlotLoss.getPlotItem().setLabel('left', "Loss")
        #->
        self.tabWidget.addTab(self.tab_2, "")

        ''' Tab prediction '''
        self.tabPrediction = QtWidgets.QWidget()
        self.tabPrediction.setObjectName("tabPrediction")

        self.labelResu = QtWidgets.QLabel(self.tabPrediction)
        self.labelResu.setGeometry(QtCore.QRect(650, 10, 151, 16))
        self.labelResu.setObjectName("labelResu")
        self.labelResu.setText("Classwise Accuracy:")
        
        self.labelResARR = QtWidgets.QLabel(self.tabPrediction)
        self.labelResARR.setGeometry(QtCore.QRect(650, 40, 121, 16))
        self.labelResARR.setObjectName("labelResARR")
        self.labelResARR.setText("ARR")
         
        self.labelResCHF = QtWidgets.QLabel(self.tabPrediction)
        self.labelResCHF.setGeometry(QtCore.QRect(650, 70, 121, 16))
        self.labelResCHF.setObjectName("labelResCHF")
        self.labelResCHF.setText("CHF")
        
        self.labelResNSR = QtWidgets.QLabel(self.tabPrediction)
        self.labelResNSR.setGeometry(QtCore.QRect(650, 100, 121, 16))
        self.labelResNSR.setObjectName("labelResNSR")
        self.labelResNSR.setText("NSR")


        self.labelPredictSCL = QtWidgets.QLabel(self.tabPrediction)
        self.labelPredictSCL.setGeometry(QtCore.QRect(10, 10, 121, 16))
        self.labelPredictSCL.setObjectName("labelPredictSCL")

        self.btnPredictSCL = QtWidgets.QPushButton(self.tabPrediction)
        self.btnPredictSCL.setGeometry(QtCore.QRect(10, 40, 113, 32))
        self.btnPredictSCL.setObjectName("btnPredictSCL")


        self.labelPredictSGN = QtWidgets.QLabel(self.tabPrediction)
        self.labelPredictSGN.setGeometry(QtCore.QRect(300, 10, 121, 16))
        self.labelPredictSGN.setObjectName("labelPredictSGN")
        

        self.btnPredictSGN = QtWidgets.QPushButton(self.tabPrediction)
        self.btnPredictSGN.setGeometry(QtCore.QRect(300, 40, 113, 32))
        self.btnPredictSGN.setObjectName("btnPredictSGN")

        self.widgetPredicSCL = pg.PlotWidget(self.tabPrediction)
        self.widgetPredicSCL.setGeometry(QtCore.QRect(30, 180, 400, 300))
        self.widgetPredicSCL.setObjectName("widgetPredicSCL")
        #self.firstSignalTwo = self.widgetPlotAcc.plot(self.time, self.sig_CHF, pen= self.redPen)

        
        self.widgetPredicSGN = pg.PlotWidget(self.tabPrediction)
        self.widgetPredicSGN.setGeometry(QtCore.QRect(450, 180, 400, 300))
        self.widgetPredicSGN.setObjectName("widgetPredicSGN")
        #self.firstSignalTwo = self.widgetPlotLoss.plot(self.time, self.sig_CHF, pen= self.redPen)
        

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
        #TAB!
        self.btnLoadData.setText(_translate("SignalProcessing", "Load Data"))
        self.btnPlotRnd.setText(_translate("SignalProcessing", "Plot Random Signal"))
        self.labelSigLength.setText(_translate("SignalProcessing", "Signal Range:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab),
                                  _translate("SignalProcessing", "Signal Pre-Processing Observation"))
        #TAB2
        self.btnTrain.setText(_translate("SignalProcessing", "Start training "))
        self.btnLoadWeights.setText(_translate("SignalProcessing", "Load"))
        self.btnLoadBestWeights.setText(_translate("SignalProcessing", "Load"))
        self.labelLoadWeights.setText(_translate("SignalProcessing", "Load pretrained weights:"))
        self.btnPredictSCL.setText(_translate("SignalProcessing", "Test Sclogram"))
        self.lblSaveWts.setText(_translate("SignalProcessing","Save weights:"))
        self.labelLoadBestWeights.setText(_translate("SignalProcessing", "Load best-pretrained weights:"))
        #self.QComboBoxRate.setText(_translate("SignalProcessing", "Unknown"))
        self.labelTrain.setText(_translate("SignalProcessing", "Train")) 
        self.labelPredictSCL.setText(_translate("SignalProcessing", "Prediction From Scalogram"))
        self.labelNetworkType.setText(_translate("SignalProcessing", "Neural Network:"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("SignalProcessing", "Training Signal"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabPrediction), _translate("SignalProcessing", "Prediction"))
        self.btnPredictSGN.setText("Test Signal")
        self.labelPredictSGN.setText("Prediction From Signal")

    def connect(self):
        ''' Define Signal and SLot fo GUI Connection '''
        self.btnLoadData.clicked.connect(self.loadData)
        self.btnTrain.clicked.connect(self.slotTrainNetwork)
        self.btnPlotRnd.clicked.connect(self.plotSignal)
        self.btnLoadWeights.clicked.connect(self.slotLoadWWeights)
        self.btnSaveWts.clicked.connect(self.save_Weights)
        self.btnLoadBestWeights.clicked.connect(self.slotLoadBWeights)

    # Slots are defined here
    def loadData(self):
        LoadECGData(self)

    def plotSignal(self):
        plot_signal_rnd(self)

    def slotTrainNetwork(self):
        trainNetwork(self)

    def slotLoadWWeights(self):
        load_weights(self, kind="weights")

    def slotLoadBWeights(self):
        load_weights(self, kind="best")

    def save_Weights(self):
        save_weights(self)

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