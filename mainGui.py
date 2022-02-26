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
        SignalProcessing.resize(1200, 760)
        
        self.centralwidget = QtWidgets.QWidget()  # SignalProcessing
        self.centralwidget.setObjectName("centralwidget")

        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 10, 1380, 900))
        self.tabWidget.setObjectName("tabWidget")

        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")

        self.cwt = pg.PlotWidget(self.tab)                         #cwt
        self.cwt.setGeometry(QtCore.QRect(30, 10, 340, 340)) 
        self.cwt.setStyleSheet("background-color: blue")
        self.cwt.setObjectName("cwt")
        self.cwt.addItem(self.imgOne)
        
        self.cwtf = pg.PlotWidget(self.tab)                         #cwtf
        self.cwtf.setGeometry(QtCore.QRect(380, 10, 340, 340))
        self.cwtf.setStyleSheet("background-color: white")
        self.cwtf.setObjectName("cwtf")
        self.cwtf.addItem(self.imgTwo)

        self.sig = pg.PlotWidget(self.tab)                           #sig
        self.sig.setGeometry(QtCore.QRect(30, 360, 340, 340))
        self.sig.setStyleSheet("background-color: green")
        self.sig.setObjectName("sig")
        self.firstSignal = self.sig.plot(self.time, self.sig_NSR, pen= self.redPen)

        self.sigf = pg.PlotWidget(self.tab)                         #sigf
        self.sigf.setGeometry(QtCore.QRect(380, 360, 340, 340))
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
        networkType = ["AlexNet", "GoogLeNet", "SqueezeNet", "ResNet"]
        self.NetworkType = QtWidgets.QComboBox(self.tab_2)
        self.NetworkType.setGeometry(QtCore.QRect(40, 125, 113, 32))
        self.NetworkType.setObjectName("NetworkType")
        self.NetworkType.addItems(networkType)

        self.labelNetworkType = QtWidgets.QLabel(self.tab_2)
        self.labelNetworkType.setGeometry(QtCore.QRect(30, 105, 121, 16))
        self.labelNetworkType.setObjectName("labelNetworkType")
        #->
        self.widgetWayOne = QtWidgets.QWidget(self.tab_2)
        self.widgetWayOne.setGeometry(QtCore.QRect(220, 40, 780, 80))
        self.widgetWayOne.setObjectName("widgetWayOne")
        self.widgetWayOne.setStyleSheet("background-color:#DCDCDC")

        self.lblbatch_size = QtWidgets.QLabel(self.tab_2)
        self.lblbatch_size.setGeometry(QtCore.QRect(240, 50, 121, 16))
        self.lblbatch_size.setObjectName("lblbatch_size")
        self.lblbatch_size.setText("Batch size:")

        listbatch_size = ["8","16", "32", "64"]
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
        self.btnTrain.setGeometry(QtCore.QRect(710, 70, 113, 32))
        self.btnTrain.setObjectName("btnTrain")

        self.labelTrain = QtWidgets.QLabel(self.tab_2)
        self.labelTrain.setGeometry(QtCore.QRect(710, 40, 120, 32))
        self.labelTrain.setObjectName("labelTrain")
        #->
        self.btnSaveWts = QtWidgets.QPushButton(self.tab_2)
        self.btnSaveWts.setGeometry(QtCore.QRect(860, 70, 113, 32))
        self.btnSaveWts.setObjectName("btnSaveWts")
        self.btnSaveWts.setText("Save")

        self.lblSaveWts = QtWidgets.QLabel(self.tab_2)
        self.lblSaveWts.setGeometry(QtCore.QRect(860, 40, 120, 30))
        self.lblSaveWts.setObjectName("lblSaveWts")


        self.widgetWayTwo = QtWidgets.QWidget(self.tab_2)
        self.widgetWayTwo.setGeometry(QtCore.QRect(390, 145, 450, 80))
        self.widgetWayTwo.setObjectName("widgetWayTwo")
        self.widgetWayTwo.setStyleSheet("background-color:#C0C0C0")
        #->
        self.btnLoadWeights = QtWidgets.QPushButton(self.tab_2)
        self.btnLoadWeights.setGeometry(QtCore.QRect(415, 180, 121,31))
        self.btnLoadWeights.setObjectName("Load weights")

        self.labelLoadWeights = QtWidgets.QLabel(self.tab_2)
        self.labelLoadWeights.setGeometry(QtCore.QRect(400, 152, 220, 25))
        self.labelLoadWeights.setObjectName("labelLoadWeights")
        #->
        self.btnLoadBestWeights = QtWidgets.QPushButton(self.tab_2)
        self.btnLoadBestWeights.setGeometry(QtCore.QRect(680, 180, 121,31))
        self.btnLoadBestWeights.setObjectName("LoadBestweights")

        self.labelLoadBestWeights = QtWidgets.QLabel(self.tab_2)
        self.labelLoadBestWeights.setGeometry(QtCore.QRect(645, 152, 220, 25))
        self.labelLoadBestWeights.setObjectName("labelLoadBestWeights")

        #-> PLOTS
        style = {"color":"w", "font-size":"12px"}
        self.widgetPlotAcc = pg.PlotWidget(self.tab_2)
        self.widgetPlotAcc.setBackground("k")
        self.widgetPlotAcc.setGeometry(QtCore.QRect(10, 260, 415, 415))
        self.widgetPlotAcc.setTitle("Accuracy Plot", color="w")
        self.widgetPlotAcc.addLegend(offset = (260,0))       
        self.widgetPlotAcc.getPlotItem().showGrid(x=True, y=True, alpha=0.7)
        self.trainAccPlot = self.widgetPlotAcc.plot(np.linspace(1, len(self.trainAcc), len(self.trainAcc)),
                                                         self.trainAcc, pen = self.redPen, name="Train acc"  )
        self.valAccPlot = self.widgetPlotAcc.plot(np.linspace(1, len(self.valAcc), len(self.valAcc)),
                                                         self.valAcc, pen = self.greenPen, name = "Val acc"  )
        self.widgetPlotAcc.getPlotItem().setLabel('bottom', "Number of Epochs")
        self.widgetPlotAcc.getPlotItem().setLabel('left', "Accuracy")


        #->
        self.widgetPlotLoss = pg.PlotWidget(self.tab_2)
        self.widgetPlotLoss.setTitle("Loss plot", color="w")
        self.widgetPlotAcc.setBackground("k")
        # self.widgetPlotLoss.setLabel("left", "Loss->", **style)
        self.widgetPlotLoss.setGeometry(QtCore.QRect(450, 260, 415, 415))
        self.widgetPlotLoss.setObjectName("widgetPlotLoss")
        self.widgetPlotLoss.addLegend(offset = (260,-300))
        self.widgetPlotLoss.getPlotItem().showGrid(x=True, y=True, alpha=0.7)
        self.trainLossPlot = self.widgetPlotLoss.plot(np.linspace(0, len(self.trainLoss), len(self.trainLoss)),
                                                         self.trainLoss, pen = self.redPen , name = "Train loss" )
        self.valLossPlot = self.widgetPlotLoss.plot(np.linspace(0, len(self.valLoss), len(self.valLoss)),
                                                         self.valLoss, pen = self.greenPen  , name="Val loss")
        self.widgetPlotLoss.getPlotItem().setLabel('bottom', "Number of Epochs")
        self.widgetPlotLoss.getPlotItem().setLabel('left', "Loss")


        #-> TEST BUTTON
        self.btnTest = QtWidgets.QPushButton(self.tab_2)
        self.btnTest.setGeometry(QtCore.QRect(1020, 125, 130, 30))
        self.btnTest.setText("Test")

        self.lblTest = QtWidgets.QLabel(self.tab_2)
        self.lblTest.setGeometry(QtCore.QRect(1010, 95, 170, 30))
        self.lblTest.setText("Run model on Test set:")

        #-> MODEL STATS AND PARAMETERS

        self.widgetCBox = QtWidgets.QWidget(self.tab_2)
        self.widgetCBox.setGeometry(QtCore.QRect(880, 260, 300, 440))
        self.widgetCBox.setObjectName("widgetWayOne")
        self.widgetCBox.setStyleSheet("background-color:#E5E4E2")

        self.lblModelStats = QtWidgets.QLabel(self.tab_2)
        self.lblModelStats.setGeometry(QtCore.QRect(995, 265, 240, 20))
        self.lblModelStats.setText("Model Stats:")
        self.lblModelStats.setStyleSheet("font: 14pt" )

        self.lblMName = QtWidgets.QLabel(self.tab_2)
        self.lblMName.setGeometry(QtCore.QRect(890, 280, 240, 20))
        self.lblMName.setText("Current model")
        

        self.txtModel = QtWidgets.QLabel(self.tab_2)
        self.txtModel.setGeometry(QtCore.QRect(1000, 280, 120, 20))
        
        self.lblLR = QtWidgets.QLabel(self.tab_2)
        self.lblLR.setGeometry(QtCore.QRect(890, 300, 240, 20))
        self.lblLR.setText("Learning rate")

        self.txtLR = QtWidgets.QLabel(self.tab_2)
        self.txtLR.setGeometry(QtCore.QRect(1000, 300, 80, 20))

        self.lblEpochs = QtWidgets.QLabel(self.tab_2)
        self.lblEpochs.setGeometry(QtCore.QRect(890, 320, 240, 20))
        self.lblEpochs.setText("Num of Epochs")

        self.txtEpochs = QtWidgets.QLabel(self.tab_2)
        self.txtEpochs.setGeometry(QtCore.QRect(1000, 320, 80, 20))

        self.lblBS = QtWidgets.QLabel(self.tab_2)
        self.lblBS.setGeometry(QtCore.QRect(890, 340, 240, 20))
        self.lblBS.setText("Batch size")

        self.txtBS = QtWidgets.QLabel(self.tab_2)
        self.txtBS.setGeometry(QtCore.QRect(1000, 340, 80, 20))

        self.valStats = QtWidgets.QLabel(self.tab_2)
        self.valStats.setGeometry(QtCore.QRect(900, 370, 240, 20))
        self.valStats.setText("Accuracy on:")
        
        self.lblValAcc = QtWidgets.QLabel(self.tab_2)
        self.lblValAcc.setGeometry(QtCore.QRect(890, 400, 240, 20))
        self.lblValAcc.setText("Validation set")

        self.txtValAcc = QtWidgets.QLabel(self.tab_2)
        self.txtValAcc.setGeometry(QtCore.QRect(1000, 400, 80, 20))

        self.lblTrainAcc = QtWidgets.QLabel(self.tab_2)
        self.lblTrainAcc.setGeometry(QtCore.QRect(890, 420, 240, 20))
        self.lblTrainAcc.setText("Training set")

        self.txtTrainAcc = QtWidgets.QLabel(self.tab_2)
        self.txtTrainAcc.setGeometry(QtCore.QRect(1000, 420, 80, 20))

        self.TestStats = QtWidgets.QLabel(self.tab_2)
        self.TestStats.setGeometry(QtCore.QRect(900, 450, 300, 20))
        self.TestStats.setText("Classwise-Accuracy on Test Set:")

        self.lblARR = QtWidgets.QLabel(self.tab_2)
        self.lblARR.setGeometry(QtCore.QRect(890, 480, 240, 20))
        self.lblARR.setText("Accuracy on ARR-Signals ")

        self.txtAccARR = QtWidgets.QLabel(self.tab_2)
        self.txtAccARR.setGeometry(QtCore.QRect(1060, 480, 80, 20))

        self.lblCHF = QtWidgets.QLabel(self.tab_2)
        self.lblCHF.setGeometry(QtCore.QRect(890, 500, 240, 20))
        self.lblCHF.setText("Accuracy on CHF-Signals ")

        self.txtAccCHF = QtWidgets.QLabel(self.tab_2)
        self.txtAccCHF.setGeometry(QtCore.QRect(1060, 500, 80, 20))

        self.lblNSR = QtWidgets.QLabel(self.tab_2)
        self.lblNSR.setGeometry(QtCore.QRect(890, 520, 240, 20))
        self.lblNSR.setText("Accuracy on NSR-Signals ")

        self.txtAccNSR = QtWidgets.QLabel(self.tab_2)
        self.txtAccNSR.setGeometry(QtCore.QRect(1060, 520, 80, 20))

        #->   TAB 3
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
        self.labelLoadWeights.setText(_translate("SignalProcessing", "Pretrained weights:"))
        self.btnPredictSCL.setText(_translate("SignalProcessing", "Test Sclogram"))
        self.lblSaveWts.setText(_translate("SignalProcessing","Save weights:"))
        self.labelLoadBestWeights.setText(_translate("SignalProcessing", "Best-pretrained weights:"))
        #self.QComboBoxRate.setText(_translate("SignalProcessing", "Unknown"))
        self.labelTrain.setText(_translate("SignalProcessing", "Train:")) 
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
        self.btnSaveWts.clicked.connect(self.slotSaveWeights)
        self.btnLoadBestWeights.clicked.connect(self.slotLoadBWeights)
        self.btnTest.clicked.connect(self.slotTest)


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

    def slotSaveWeights(self):
        save_weights(self)
    
    def slotTest(self):
        validate_test_set(self)

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