from math import ceil
import random
from scipy.io import loadmat
from scipy import signal
import numpy as np
import os
from PIL import Image
from matplotlib.pyplot import get_cmap
import shutil


def butter_highpass_filter(data, cutoff=1, fs=128, order = 5):
    ''' -> Used to remove the low frequency signals causing base wandering [http://www.jscholaronline.org/articles/JBER/Signal-Processing.pdf] 
    Parameters: data   : ECG Signal [np.array]
                cutoff : cutoff frequency
                fs     : Sampling frequency
                order  : Order of the filter
    Output: Signal with no base wandering
    '''
    normal_cutoff = cutoff / (fs/ 2)
    b, a = signal.butter(order, normal_cutoff, btype="high", analog = False)
    y = signal.filtfilt(b, a, data)
    return y

def notch_filter(data, cutoff=60, fs=128, q = 30):
    ''' -> Used to remove the electromagnetic noise caused by the 60Hz power line [http://www.jscholaronline.org/articles/JBER/Signal-Processing.pdf]
    Parameters: data   : ECG Signal [np.array]
                cutoff : cutoff frequency
                fs     : Sampling frequency
                order  : Order of the filter
    Output: Signal with no Electromagnetic noise
    '''
    normal_cutoff = cutoff / (fs/2)
    b, a = signal.iirnotch(normal_cutoff, Q=q, fs=fs)
    y = signal.filtfilt(b, a, data)
    return y

def process_signal(data):
    ''' -> Function, removing both base wandering and electromagnetic noise
    Input  : Unprocessed signal array
    Output : Processes signal
    '''
    data = butter_highpass_filter(data) 
    data = notch_filter(data)
    return data


cutoff = 1
fs = 128

random.seed(0)
data = loadmat("ECGData.mat")
data = data["ECGData"]
ecg = data["Data"][0][0]
labels = data["Labels"][0][0]
labels = [labels[i][0][0] for i in range(labels.size)]
sig_ARR, lab_ARR = ecg[0:95] , labels[0:95]
sig_CHF, lab_CHF = ecg[96:125] , labels[96:125]
sig_NSR, lab_NSR = ecg[125:161] , labels[126:161]

def create_save_wavelets(data:np.ndarray, name:str):
    ''' -> Function to create wavelets from a category of signal.
    Parameters: data  : Array of signals of a particular type   [np.array]
                name  : Types of signals ("ARR", "CHF", "NSR")  [string]
    '''
    
    #Create the necessary directories in the project folder
    [os.makedirs(x, exist_ok=True) for x in ["Dataset/ARR", "Dataset/CHF", "Dataset/NSR"]]
    path = "Dataset/" + name.upper()
    if not os.path.isdir(path):
        assert NotADirectoryError(f"{path} is not a valid path")

    # For all the signals of a particular type:
    for i, sig in enumerate(data):
        
        #If the signal is inverted, correct it
        if(np.max(sig) < np.abs(np.min(sig)) ):
            sig = -1*sig
        sig = process_signal(sig)
        
        #Subsample the signal to create a robust dataset [Signal length = 1280 i.e. Sampling frequency(128) * 10]
        start = [5000, 7000, 10000,13000, 15000, 17000, 20000,23000, 25000,27000, 30000,33000, 35000, 40000,45000,47000, 50000,53000, 55000, 57000, 60000]
        stop = [x+1280 for x in start]  

        #SAMPLE CORRECTION : To get almost uniform dataset
        #Reducing the number of samples for signal of type ARR to 1/3 of the rest as the number of ARR signal is thrice of that of CHF and NSR
        if name == "ARR":
            start = start[0::3]
            stop = [x+1280 for x in start]
        if name == "CHF":
            start = start[:-2]

        
        cnt = 0
        #For a signal, create images
        for k,l in zip(start, stop):
            #For each sample create a unique directory
            imdir = os.path.join(path , name+"_"+str(i)+"_"+str(cnt)+".png")
            #Scipy cwt: [https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cwt.html]
            cwt = signal.cwt(sig[k:l], signal.morlet2, widths=np.arange(1,101,100/1000), w=3.5)
            #Convert cwt to absolute values
            cwt = np.abs(cwt)
            #Get a colormap to generate color images
            cm = get_cmap('viridis')
            #Apply color map while converting the pixel values from [0~1] to [0~255]
            cwt = (cm(cwt)[:,:,:3]*255).astype(np.uint8)
            #Create Image from array
            result = Image.fromarray(cwt)
            # Resize the image to (224,224) i.e. resolution needed for Neural networks
            result = result.resize((224,224), resample = Image.BICUBIC)
            #Save the image
            result.save(imdir, format="png")
            cnt += 1

   

class Create_Database:
    ''' -> Utility to create dataset in the format suitable for neural netowrk
        Input: Folder format :   Dataset--ARR
                                        |_CHF
                                        |_NSR
                split        : [train_ratio, val_ratio, test_ratio] such that: train_ratio + val_ratio + test_ratio == 1

        Output: Folder format:  images- _train ---ARR
                                       |       |__CHF
                                       |       |__NSR
                                       |
                                       |_val------ARR
                                       |       |__CHF 
                                       |       |__NSR
                                       |
                                       |_test-----ARR
                                               |__CHF
                                               |__NSR  
    '''                                 
    def __init__(self, src = "Dataset", dst = "images", split = [0.7, 0.2, 0.1]) -> None:
        self.signals = os.listdir(src)
        print(self.signals)
        self.src = src
        self.dst = dst
        self.seed = random.seed(random.random())
        self.splits = [split[0], split[1]/ (split[1]+ split[2]), None]

    #Create train folder
    def ds_train(self):
        print(self.signals)
        dst_path = os.path.join(self.dst, "train")
        [os.makedirs(os.path.join(dst_path, x), exist_ok=True) for x in self.signals]

        for sig in self.signals:
            path = os.path.join(self.src, sig)
            files = os.listdir(path)
            random.shuffle(files)
            print(len(files))
            train = files[:ceil(self.splits[0]*len(files))]
            print(len(train))

            for file in train:
                file = os.path.join(self.src, sig, file)
                shutil.move(file, os.path.join(dst_path, sig))
    #Create Validation folder
    def ds_valid(self):
        dst_path = os.path.join(self.dst, "val")
        [os.makedirs(os.path.join(dst_path, x), exist_ok=True) for x in self.signals]

        for sig in self.signals:
            path = os.path.join(self.src, sig)
            files = os.listdir(path)
            random.shuffle(files)
            print(len(files))
            val = files[:ceil(self.splits[1]*len(files))]
            print(len(val))

            for file in val:
                file = os.path.join(self.src, sig, file)
                shutil.move(file, os.path.join(dst_path, sig))
    #Create test folder
    def ds_test(self):
        dst_path = os.path.join(self.dst, "test")
        [os.makedirs(os.path.join(dst_path, x), exist_ok=True) for x in self.signals]

        for sig in self.signals:
            path = os.path.join(self.src, sig)
            files = os.listdir(path)

            for file in files:
                file = os.path.join(self.src, sig, file)
                shutil.move(file, os.path.join(dst_path, sig))
    #Masking the individual functions into one function for the class
    def create(self):
        self.ds_train()
        self.ds_valid()
        self.ds_test()

if __name__ == "__main__":
    create_save_wavelets(sig_NSR, "NSR")        
    create_save_wavelets(sig_ARR, "ARR")
    create_save_wavelets(sig_CHF, "CHF") 
    Create_Database(src="Dataset", dst = "images", split = [0.7, 0.2, 0.1]).create()


