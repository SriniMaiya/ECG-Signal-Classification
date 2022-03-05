from PyQt5.QtWidgets import QFileDialog
from scipy import io, signal
import numpy as np
from scipy.io import loadmat
from matplotlib.pyplot import get_cmap
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
from torch import nn
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from Models.models import NeuralNetworkClassifier
from Models.utils import train_dataloader
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu" ).type

# HOw to open(Brows) data frmo PyQt
def LoadECGData(self):     
    '''
    Function: Open QFileDialog to address and open the ECG signals 
    base on the User decisions 

    Connection: It is called from slot @XXX From Main Gui 
    '''
    (self.FilePath, _ )= QFileDialog.getOpenFileNames(self, "Choose File as .MAT", "", "ECG data set (*.mat *.MAT)")
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



def plot_signal_rnd(self):
    '''
    Function : Plot all signals in randomly in given time steps 
    Connection: It is called from slot @Plot From Main Gui
    and it is a in connect with @pyqtgraph 
    For display puposes
    '''
    if int(self.txtSigStart.toPlainText()) >= 0 and int(self.txtSigStart.toPlainText()) <= 60000:
        if ((int(self.txtSigEnd.toPlainText()) > int(self.txtSigStart.toPlainText()))
                and int(self.txtSigEnd.toPlainText()) <= 60000):

            lengthStart = int(self.txtSigStart.toPlainText())
            lengthEnd = int(self.txtSigEnd.toPlainText())
            Signals = creatRndPlotSignal(self.selectSig.currentIndex(), self.sig_ARR, self.sig_CHF, self.sig_NSR,
                                         lengthStart, lengthEnd)

            sig_plot = Signals[0]  # sig_plot

            sig_filter_plot = Signals[1]  # sig_filter_plot

            wavelet_plot = Signals[2]  # wavelet_plot

            wavelet_filter_plot = Signals[3]  # wavelet_filter_plot

            self.time = range(0, len(sig_plot), 1)
            self.time = list(self.time)

            self.time = range(0, len(sig_filter_plot), 1)
            self.time = list(self.time)

            self.time = range(0, len(wavelet_plot), 1)
            self.time = list(self.time)

            self.time = range(0, len(wavelet_filter_plot), 1)
            self.time = list(self.time)

            self.firstSignal.setData(sig_plot)
            self.firstSignalTwo.setData(sig_filter_plot)
            self.imgOne.setImage(wavelet_plot)
            self.imgTwo.setImage(wavelet_filter_plot)
        else:
            print("Please Enter Correct Value")
    else:
        print("Please Enter Correct Value")

def butter_highpass_filter(data, cutoff=1, fs=128, order=5):
    '''Function : High Pass FAlter signal 
    Description : Design an Nth-order digital or analog Butterworth filter and return the filter 
    coefficients (MATLAB IIR Filter format)

    returned : filter forward and backward signal
    '''
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
def creatRndPlotSignal(num, ARR, CHF, NSR, lengthStart, lengthEnd):
    ind_ARR = np.random.randint(low=0, high=ARR.shape[0])
    ind_CHF = np.random.randint(low=0, high=CHF.shape[0])
    ind_NSR = np.random.randint(low=0, high=NSR.shape[0])

    if num == 0:
        sig = ARR[ind_ARR][lengthStart:lengthEnd]
        sigf = butter_highpass_filter(sig)
        sigf = notch_filter(sigf)
    elif num == 1:
        sig = CHF[ind_CHF][lengthStart:lengthEnd]
        sigf = butter_highpass_filter(sig)
        sigf = notch_filter(sigf)
    elif num == 2:
        sig = NSR[ind_NSR][lengthStart:lengthEnd]
        sigf = butter_highpass_filter(sig)
        sigf = notch_filter(sigf)


    if np.max(sig) < np.abs(np.min(sig)):
        sig = -1 * sig
    if np.max(sigf) < np.abs(np.min(sigf)):
        sigf = -1 * sigf

    cwt = signal.cwt(sig, signal.morlet2, widths=np.arange(1, 81, 80 / 1000),
                     w=3.5)  # cwtf is complex number and it should be plotted as abs value

    cwtf = signal.cwt(sigf, signal.morlet2, widths=np.arange(1, 81, 80 / 1000),
                      w=3.5)  # cwtf is complex number and it should be plotted as abs value
    cwt = np.abs(cwt)
    cwtf = np.abs(cwtf)

    cm = get_cmap('viridis')
    cwt = np.rot90((cm(cwt)[:, :, :3] * 255).astype(np.uint8))
    cwtf = np.rot90((cm(cwtf)[:, :, :3] * 255).astype(np.uint8))

    return [sig, sigf, cwt, cwtf]


def trainNetwork(self):
    '''Get the values from GUI and call the training function from models.py '''
    if device == "cuda":
        torch.cuda.empty_cache()
    #get batch_size, number_of_epochs, learning_rate and model_name from user input
    self.batch_size = int(self.QCombobatch_size.currentText())
    self.num_epochs = int(self.txtNum_epochs.toPlainText())
    self.learning_rate = float(self.QComboBoxRate.currentText())
    model_name = self.NetworkType.currentText()

    print(self.batch_size, self.num_epochs, self.learning_rate)
    #Select the user specified neural network
    classifier = NeuralNetworkClassifier(model_name=model_name)
    #Get dataloader and load the images of size batch number at a time
    dataloader = train_dataloader(input_size=224, dataset_path="images",
                                  batch_size=self.batch_size)  # add Qcombox for batch size
    classifier.initialize_model(num_classes=3)
    #Start training and return model parameters
    self.model, self.history, self.weights, self.best_weights = classifier.start_training(dataloaders=dataloader,
                                                                                          num_epochs=self.num_epochs,
                                                                                          learning_rate=self.learning_rate)
    #Load the model with trained weights                                                                                      
    self.model.load_state_dict(self.weights)
    
    # Add the train, validation loss and accuracy plots 
    plotAccTrain = self.history["train"]["acc"]
    plotLossTrain = self.history["train"]["loss"]
    plotAccVal = self.history["val"]["acc"]
    plotLossVal = self.history["val"]["loss"]
    #Send the plots to GUI
    self.trainAccPlot.setData(plotAccTrain)
    self.valAccPlot.setData(plotAccVal)
    self.trainLossPlot.setData(plotLossTrain)
    self.valLossPlot.setData(plotLossVal)
    
    #Store training and validation accuracies for Text-Display
    self.ValAcc = plotAccVal[-1]
    self.TrainAcc = plotAccTrain[-1]

    print_model_stats(self)


def save_weights(self):
    '''Saves weights, loss-accuracy data and model parameters(batchsize, learning rate,
     train and val accuracy) for further calls when the weights are loaded back'''
    weights_path = os.path.join("Weights", self.model._get_name())
    os.makedirs(weights_path, exist_ok=True)
    torch.save(self.weights, os.path.join(weights_path, "weights_"+device+".pth"))
    with open(os.path.join("Weights", self.model._get_name(), "model_hist_"+device+".pkl"), 'wb') as f:
        pickle.dump(self.history, f)
    path = os.path.join("Weights", self.model._get_name(), "model_params_"+device+".npy")
    params = [self.batch_size, self.num_epochs, self.learning_rate, self.ValAcc, self.TrainAcc, self.batch_size]
    np.save(path, params)
    print("\n--> Saved weights and model history and train-parameters successfully.\n")


def load_weights(self):
    '''Load the weights and all model information back to the GUI 
        when the user clicks the button to load the weights'''
    self.model = None
    self.weights = None
    torch.cuda.empty_cache()
    model_name = self.NetworkType.currentText()
    self.classifier = NeuralNetworkClassifier(model_name=model_name)
    self.classifier.initialize_model(num_classes=3)
    self.model = self.classifier.model
    # Load Weights
    self.weights = torch.load(os.path.join("Weights", self.model._get_name(), "weights_"+device+".pth"))

    # Load the training and accuracy curves
    with open(os.path.join("Weights", self.model._get_name(), "model_hist_"+device+".pkl"), 'rb') as f:
        self.history = pickle.load(f)
    # Load Model Parameters
    path = os.path.join("Weights", self.model._get_name(), "model_params_"+device+".npy")
    self.batch_size, self.num_epochs, self.learning_rate, self.ValAcc, self.TrainAcc, self.batch_size = [str(x) for x in
                                                                                                         np.load(path)]
    #Plot the train and accuracy curves
    plotAccTrain = self.history["train"]["acc"]
    plotLossTrain = self.history["train"]["loss"]
    plotAccVal = self.history["val"]["acc"]
    plotLossVal = self.history["val"]["loss"]

    self.model.load_state_dict(self.weights)
    self.model.eval()

    self.trainAccPlot.setData(plotAccTrain)
    self.valAccPlot.setData(plotAccVal)
    self.trainLossPlot.setData(plotLossTrain)
    self.valLossPlot.setData(plotLossVal)
    # print model stats
    print_model_stats(self)
    print("--> Weights and training curves loaded successfully.")


def validate_test_set(self):
    '''Validation of the model on the test set.
        Predict on the whole test set.
        Calculate test accuracy and plot the Confusion matrix in the GUI '''
    signals = ["ARR", "CHF", "NSR"]
    acc = [0, 0, 0] #Classwise accuracy
    preds = []      #Predictions
    labels = []     #True label

    for i, sig in enumerate(signals):                                           # For each class of the test set, store the image names
        img_loc = os.listdir(os.path.join("images", "test", sig))
        dir_len = len(img_loc)

        for img in img_loc:                                                     # For image in the stored images
            img = os.path.join("images", "test", sig, img)
            img = Image.open(img)                                               # Open  the image

            totensor = transforms.Compose([                                     # Transformations similar to dataset loader    
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            
            img = totensor(img)                                                 # Apply transformations on the image and convert to tensor    
            
            if device == "cuda":                                                # Add the missing dimension as the single image has 3 Dimensions, but 4 are required
                img = torch.unsqueeze(img, 0).cuda()                            #   to mimic the dataloader and to match the weights format
            else:
                img = torch.unsqueeze(img, 0)

            output = self.model(img)                                            # Predict    
            pred = torch.argmax(output)                                         # Get the index of the prediction array with maximum value
            labels.append(i)                                                    # Append the original labels
            preds.append(pred.cpu().numpy())                                    # Append the prediction to an array

            if pred.data == i:                                                  # IF correct prediction has been made,    
                acc[i] = acc[i] + 1                                             #   increase the number of correct counts by 1    
 
        acc[i] /= dir_len                                                       # Divide the correct predictions by the number of images in each class 

    self.txtAccARR.setText(":  " + "{:.4f}".format(round(acc[0], 4)))           # Display accuracy of each in GUI
    self.txtAccCHF.setText(":  " + "{:.4f}".format(round(acc[1], 4)))
    self.txtAccNSR.setText(":  " + "{:.4f}".format(round(acc[2], 4)))

    print(acc)
    preds = np.array(preds)
    labels = np.array(labels)
    # print(preds)
    # print(labels)
    cnf_mat = confusion_matrix(labels, preds)                                                           # Create confusion matrix from the array       
    ax = plt.subplot()
    sns.heatmap(cnf_mat, cbar=False, ax=ax, cmap="Blues", fmt="g", xticklabels=["ARR", "CHF", "NSR"],   # Create seaborn confusion matrix image
                yticklabels=["ARR", "CHF", "NSR"], annot=True, annot_kws={'size': '15'})
    sns.set(font_scale=3.0)
    ax.set_xlabel("Predicted labels", {'size': '15'})
    ax.set_ylabel("True labels", {'size': '15'})
    ax.set_title("Confusion Matrix", {'size': '15'})
    ax.tick_params(axis='both', which='major', labelsize=15)

    plt.plot()
    pth = os.path.join("Weights", self.model._get_name(), "ConfMat" + self.model._get_name() + ".jpg")
    plt.tight_layout()
    plt.savefig(pth, pad_inches=0, dpi=512 / 4)                                                         # Save the confusion matrix
    img = Image.open(pth)
    img = np.array(img.resize((1024, 1024), Image.LANCZOS))
    img = np.rot90(img, 3)
    img = img[80:-80]

    plt.close()
    self.conf_Plt.setImage(img)                                                                         # Display the confusion matrix
    self.conf_Plt.render()

def pred_SCL(self):         
    '''Prediction of the scalogram from user input and display'''                               
    
    #Get the file path from the user                                     
    (self.filepath, _) = QFileDialog.getOpenFileNames(self, "Open a scalogram to predict","", "Scalogram(*.jpeg, *.png)" )
    img = Image.open(self.filepath[0])
    #Convert the image to tensor and preprocess the tensor
    totensor = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225]),
                                    transforms.Resize(224)
                                ])
    scalogram = totensor(img)
    #Send the image to appropriate device
    if device == "cuda":
        scalogram = torch.unsqueeze(scalogram, 0).cuda()
    else:
        scalogram = torch.unsqueeze(scalogram, 0)

    # Get the probabilities of prediction for each class using nn.Softmax and convert to a numpy array
    preds = nn.Softmax(dim=1)(self.model(scalogram))
    preds = preds.cpu().detach().numpy()[0] * 100

    # Send the each index of array to appropriate class
    self.predARR.setText(": {:.2f}%".format(preds[0]))
    self.predCHF.setText(": {:.2f}%".format(preds[1]))
    self.predNSR.setText(": {:.2f}%".format(preds[2]))
   
    img = np.array(img)
    img = np.rot90(img)
    self.predImg.setImage(img)


def print_model_stats(self):
    ''' A unified function to print the gathered model statistics'''
    self.txtModel.setText(":  " + self.model._get_name())
    self.txtLR.setText(":  " + str(self.learning_rate))
    self.txtEpochs.setText(":  " + str(int(float(self.num_epochs))))
    self.txtBS.setText(":  " + str(int(float(self.batch_size))))
    self.txtValAcc.setText(":  " + "{:.4f}".format(round(float(self.ValAcc), 4)))
    self.txtTrainAcc.setText(":  " + "{:.4f}".format(round(float(self.TrainAcc), 4)))


def getPredictFile(self):
    ''' Load Signal or Scologram Directory '''
    self.FilePathPredic = QFileDialog.getExistingDirectory(self, "Select Directory")


