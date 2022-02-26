from PyQt5.QtWidgets import QFileDialog
from scipy import io, signal
import numpy as np
from scipy.io import loadmat
from matplotlib.pyplot import get_cmap
import os
import torch
from torchvision import transforms
from PIL import Image
from Models.models import NeuralNetworkClassifier
from Models.utils import train_dataloader
import pickle

"""
Function: Open QFileDialog to address and open the ECG signals 
base on the User decisions 

Connection: It is called from slot @XXX From Main Gui 
"""

# HOw to open(Brows) data frmo PyQt
def LoadECGData(self):
    (self.FilePath, ECGData) = QFileDialog.getOpenFileNames(self, "Choose File as .MAT", "", "ECG data set ("
                                                                                             "*.mat *.MAT)")
    ECGData = loadmat(self.FilePath[0])  # Need To be modified
    # ECGData.keys(), type(ECGData)
    ECGData = ECGData["ECGData"]  # Need To be modified
    ecgSignal = ECGData["Data"][0][0]

    lbls = ECGData["Labels"][0][0]

    lbls = [lbls[i][0][0] for i in range(lbls.size)]

    self.sig_ARR, _ = ecgSignal[0:95], lbls[0:95]
    self.sig_CHF, _ = ecgSignal[96:125], lbls[96:125]
    self.sig_NSR, _ = ecgSignal[125:161], lbls[126:161]

    print("\n--> Data successfully loaded!")
    print("Number of ARR samples: ", self.sig_ARR.shape[0])
    print("Number of NSR samples: ", self.sig_NSR.shape[0])
    print("Number of CHF samples: ", self.sig_CHF.shape[0])


"""
Function : Plot all signals in randomly in given time steps 
Connection: It is called from slot @Plot From Main Gui
and it is a in connect with @pyqtgraph 

returned : Nothing
"""


def plot_signal_rnd(self):

    if int(self.txtSigStart.toPlainText()) >= 0 and int(self.txtSigStart.toPlainText()) <= 60000:
        if(    (int(self.txtSigEnd.toPlainText()) > int(self.txtSigStart.toPlainText())) 
                                                      and int(self.txtSigEnd.toPlainText()) <= 60000    ):

            lengthStart = int(self.txtSigStart.toPlainText())
            lengthEnd = int(self.txtSigEnd.toPlainText())
            Signals = creatRndPlotSignal(self.selectSig.currentIndex(), self.sig_ARR, self.sig_CHF, self.sig_NSR, lengthStart, lengthEnd)

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
                     w=3.5)  # cwtf is complex number and it should be plotted as abs value
    
    cwtf = signal.cwt(sigf, signal.morlet2, widths=np.arange(1,81,80/1000),
                      w=3.5)  # cwtf is complex number and it should be plotted as abs value
    cwt = np.abs(cwt)
    cwtf = np.abs(cwtf)

    cm = get_cmap('viridis')
    cwt = np.rot90((cm(cwt)[:,:,:3]*255).astype(np.uint8))
    cwtf = np.rot90((cm(cwtf)[:,:,:3]*255).astype(np.uint8))
    
    return [sig, sigf, cwt, cwtf]
    

def trainNetwork(self):
    torch.cuda.empty_cache()
    self.batch_size=int(self.QCombobatch_size.currentText())
    self.num_epochs=int(self.txtNum_epochs.toPlainText())
    self.learning_rate=float(self.QComboBoxRate.currentText())
    model_name = self.NetworkType.currentText()

    print(self.batch_size, self.num_epochs, self.learning_rate     )
    classifier = NeuralNetworkClassifier(model_name=model_name)
    dataloader = train_dataloader(input_size = 224, dataset_path="images", batch_size=self.batch_size) # add Qcombox for batch size
    classifier.initialize_model(num_classes=3)
    self.model, self.history, self.weights, self.best_weights = classifier.start_training(dataloaders=dataloader, 
                                    num_epochs=self.num_epochs,learning_rate=self.learning_rate)
    self.model.load_state_dict(self.weights)
            # This Value will goes to the plot 
    plotAccTrain = self.history["train"]["acc"]
    plotLossTrain = self.history["train"]["loss"]
    plotAccVal = self.history["val"]["acc"]
    plotLossVal = self.history["val"]["loss"]
    self.trainAccPlot.setData(plotAccTrain)
    self.valAccPlot.setData(plotAccVal)
    self.trainLossPlot.setData(plotLossTrain)
    self.valLossPlot.setData(plotLossVal)
    self.ValAcc = plotAccVal[-1]
    self.TrainAcc = plotAccTrain[-1]

    print_model_stats(self)
    
def save_weights(self):
    weights_path = os.path.join("Weights", self.model._get_name())
    os.makedirs(weights_path, exist_ok=True)
    torch.save(self.weights, os.path.join(weights_path,"weights.pth"))
    torch.save(self.best_weights, os.path.join(weights_path,"best_weights.pth"))
    with open(os.path.join("Weights", self.model._get_name(),"model_hist.pkl"), 'wb') as f:
        pickle.dump(self.history, f)
    path = os.path.join("Weights", self.model._get_name(),"model_params.npy")
    params = [self.batch_size, self.num_epochs, self.learning_rate, self.ValAcc, self.TrainAcc, self.batch_size]
    np.save(path, params)
    print("\n--> Saved weights and model history and train-parameters successfully.\n")

def load_weights(self, kind:str):
    self.model = None
    self.weights = None
    torch.cuda.empty_cache()
    model_name = self.NetworkType.currentText()
    self.classifier = NeuralNetworkClassifier(model_name=model_name)
    self.classifier.initialize_model(num_classes=3)
    self.model = self.classifier.model
    #Load Weights
    if kind == "weights":
        self.weights = torch.load(os.path.join("Weights", self.model._get_name(),"weights.pth"))

    elif kind == "best":
        self.weights = torch.load(os.path.join("Weights", self.model._get_name(),"best_weights.pth"))
    self.model.load_state_dict(self.weights)
    self.model.eval()
    #Load the training and accuracy curves
    with open(os.path.join("Weights", self.model._get_name(),"model_hist.pkl"), 'rb') as f:
        self.history = pickle.load(f)
    #Load Model Parameters
    path = os.path.join("Weights", self.model._get_name(),"model_params.npy")
    self.batch_size, self.num_epochs, self.learning_rate, self.ValAcc, self.TrainAcc, self.batch_size = [str(x) for x in np.load(path)]
    
    
    plotAccTrain = self.history["train"]["acc"]
    plotLossTrain = self.history["train"]["loss"]
    plotAccVal = self.history["val"]["acc"]
    plotLossVal = self.history["val"]["loss"]

    self.trainAccPlot.setData(plotAccTrain)
    self.valAccPlot.setData(plotAccVal)
    self.trainLossPlot.setData(plotLossTrain)
    self.valLossPlot.setData(plotLossVal)
    #print model stats
    print_model_stats(self)
    print("--> Weights and training curves loaded successfully.")

def validate_test_set(self):
    signals = ["ARR", "CHF", "NSR"]
    acc = [0,0,0]
    for i, sig in enumerate(signals):
        img_loc = os.listdir(os.path.join("images","test", sig))
        dir_len = len(img_loc)

        for img in img_loc:
            img = os.path.join("images","test",sig, img)
            img = Image.open(img)
            totensor = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(   mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]) ])
            img = totensor(img)
            img = torch.unsqueeze(img, 0).cuda()
            try:
                # if self.model._get_name() == "GoogLeNet":
                #     output = self.model(img)
                #     output = output.logits
                # else:
                output = self.model(img)
                pred = torch.argmax(output)
                if pred.data == i:
                    acc[i] = acc[i] + 1 
                excp = None
            except Exception as e:
                excp = e
                pass
            # print(pred.data == 0)
        acc[i] /= dir_len
    self.txtAccARR.setText(":  "+"{:.4f}".format(round(acc[0], 4)))    
    self.txtAccCHF.setText(":  "+"{:.4f}".format(round(acc[1], 4)))
    self.txtAccNSR.setText(":  "+"{:.4f}".format(round(acc[2], 4)))

    if excp:
        print(excp)
        print("\n\n--> Weights are not loaded yet.\n\tOptions:\n1. Train the model first -> Test\n2. Load Weights -> Test ")
    print(acc)

def print_model_stats(self):
    self.txtModel.setText(":  "+self.model._get_name())
    self.txtLR.setText(":  "+str(self.learning_rate))    
    self.txtEpochs.setText(":  "+str(int(float(self.num_epochs))) )
    self.txtBS.setText(":  "+str(int(float(self.batch_size)))   )
    self.txtValAcc.setText(":  "+"{:.4f}".format(round(float(self.ValAcc), 4)) )
    self.txtTrainAcc.setText(":  "+"{:.4f}".format(round(float(self.TrainAcc), 4)) )


            
            


