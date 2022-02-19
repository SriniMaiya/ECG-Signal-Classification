from importlib import import_module
from PyQt5.QtWidgets import QFileDialog
from scipy import io, signal
import numpy as np
from scipy.io import loadmat
from matplotlib.pyplot import get_cmap
import os
from PIL import Image
from Models.models import NeuralNetworkClassifier

"""
Function: Open QFileDialog to address and open the ECG signals 
base on the User decisions 

Connection: It is called from slot @XXX From Main Gui 
"""


def LoadECGData(self):
    (self.FilePath, ECGData) = QFileDialog.getOpenFileNames(self, "Choose File as .MAT", "", "ECG data set ("
                                                                                             "*.mat *.MAT)")
    ECGData = loadmat(self.FilePath[0])  # Need To be modified
    # ECGData.keys(), type(ECGData)
    ECGData = ECGData["ECGData"]  # Need To be modified
    ecgSignal = ECGData["Data"][0][0]

    lbls = ECGData["Labels"][0][0]

    lbls = [lbls[i][0][0] for i in range(lbls.size)]

    self.sig_ARR, lab_ARR = ecgSignal[0:95], lbls[0:95]
    self.sig_CHF, lab_CHF = ecgSignal[96:125], lbls[96:125]
    self.sig_NSR, lab_NSR = ecgSignal[125:161], lbls[126:161]

    print(self.sig_NSR)
    print(self.sig_ARR)
    print(self.sig_CHF)


"""
Function : Plot all signals in randomly in given time steps 
Connection: It is called from slot @Plot From Main Gui
and it is a in connect with @pyqtgraph 

returned : Nothing
"""


def plot_signal_rnd(self):

    if int(self.txtLenSignal.toPlainText()) >= 0 and int(self.txtLenSignal.toPlainText()) <= 60000:
        if int(self.txtLenSignalEnd.toPlainText()) > int(self.txtLenSignal.toPlainText()) and int(self.txtLenSignalEnd.toPlainText()) <= 60000 :

            lengthStart = int(self.txtLenSignal.toPlainText())
            lengthEnd = int(self.txtLenSignalEnd.toPlainText())
            Signals = creatRndPlotSignal(self.comboBox.currentIndex(), self.sig_ARR, self.sig_CHF, self.sig_NSR, lengthStart, lengthEnd)

            sig_plot = Signals[0]   # sig_plot
            #sig_plot = sig_plot[0:int(self.txtLenSignal.toPlainText())]

            sig_filter_plot = Signals[1]  # sig_filter_plot
            #sig_filter_plot = sig_filter_plot[0:int(self.txtLenSignal.toPlainText())]

            wavelet_plot = Signals[2]  # wavelet_plot
            #wavelet_plot = wavelet_plot[0:int(self.txtLenSignal.toPlainText())]

            wavelet_filter_plot = Signals[3]  # wavelet_filter_plot
            #wavelet_filter_plot = wavelet_filter_plot[0:int(self.txtLenSignal.toPlainText())]

            self.time = range(0,  len(sig_plot), 1)
            self.time = list(self.time)

            self.time = range(0,  len(sig_filter_plot), 1)
            self.time = list(self.time)

            self.time = range(0,  len(wavelet_plot), 1)
            self.time = list(self.time)

            self.time = range(0,  len(wavelet_filter_plot), 1)
            self.time = list(self.time)

            self.firstSignal.setData(sig_plot)
            self.firstSignalTwo.setData(sig_filter_plot)
            self.imgOne.setImage(wavelet_plot)
            self.imgTwo.setImage(wavelet_filter_plot)
        else:
            print("Please Enter Correct Value")
    else:
        print("Please Enter Correct Value")



"""Function : High Pass FAlter signal 
Description : Design an Nth-order digital or analog Butterworth filter and return the filter 
coefficients (MATLAB IIR Filter format)

returned : filter forward and backward signal
"""


def butter_highpass_filter(data, cutoff=1, fs=128, order=5):
    normal_cutoff = cutoff / (fs / 2)
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)  # b = ndarray, a = ndarray
    filteredSignal = signal.filtfilt(b, a, data)
    return filteredSignal


def notch_filter(data, cutoff=60, fs=128, q=30):
    normal_cutoff = cutoff / (fs / 2)
    b, a = signal.iirnotch(normal_cutoff, Q=q, fs=fs)
    filteredSignal = signal.filtfilt(b, a, data)
    return filteredSignal


# Creat Random Variable for Plotting
def creatRndPlotSignal(num, ARR, CHF, NSR,lengthStart, lengthEnd):

    ind_ARR = np.random.randint(low=0, high=ARR.shape[0])
    ind_CHF = np.random.randint(low=0, high=CHF.shape[0])
    ind_NSR = np.random.randint(low=0, high=NSR.shape[0])

    if num == 0:
        sig = ARR[ind_ARR][lengthStart:lengthEnd]
        sigf = butter_highpass_filter(sig)
        sigf = notch_filter(sigf)
        # lab = self.lab_ARR[ind_ARR]
    elif num == 1:
        sig = CHF[ind_CHF][lengthStart:lengthEnd]
        sigf = butter_highpass_filter(sig)
        sigf = notch_filter(sigf)
        # lab = self.lab_CHF[ind_CHF]
    elif num == 2:
        sig = NSR[ind_NSR][lengthStart:lengthEnd]
        sigf = butter_highpass_filter(sig)
        sigf = notch_filter(sigf)
        # lab = self.lab_NSR[ind_NSR]

        #sig = sig[0:length]  # Can be
        #sigf = sigf[0:length]

    if np.max(sig) < np.abs(np.min(sig)):
        sig = -1 * sig
    if np.max(sigf) < np.abs(np.min(sigf)):
        sigf = -1 * sigf

    cwt = signal.cwt(sig, signal.morlet2, widths=np.arange(1,81,80/1000),
                     w=8)  # cwtf is complexe number and it should be plotted as abs value
    
    cwtf = signal.cwt(sigf, signal.morlet2, widths=np.arange(1,81,80/1000),
                      w=8)  # cwtf is complexe number and it should be plotted as abs value
    cwt = np.abs(cwt)
    cwtf = np.abs(cwtf)

    cm = get_cmap('viridis')
    cwt = np.rot90((cm(cwt)[:,:,:3]*255).astype(np.uint8))
    cwtf = np.rot90((cm(cwtf)[:,:,:3]*255).astype(np.uint8))
    
    return [sig, sigf, cwt, cwtf]
    

def trainNetwork(self):
    model = NeuralNetworkClassifier(self.comboBox_2.currentText())
    dataloader = model.dataloader(dataset_path="images", batch_size=1) # add Qcombox for batch size
    model.initialize_model(num_classes=3, feature_extract=True, learning_rate=float(self.QComboBoxRate.currentText()))
    model, self.history = model.start_training(dataloaders=dataloader, num_epochs=1, save_weights= True, weights_path="Weights")
    #torch.cuda.empty_cache()

    '''
    classifier = NeuralNetworkClassifier(model_name="squeezenet")
    dataloader = classifier.dataloader(dataset_path="images", batch_size=16)
    classifier.initialize_model(num_classes=3, feature_extract=True, learning_rate=0.001)
    model, history = classifier.start_training(dataloaders=dataloader, num_epochs=1, save_weights= True, weights_path="Weights")
    torch.cuda.empty_cache()
    '''