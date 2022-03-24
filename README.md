# ECG-Signal-Classification

This project focuses on training neural networks to classify the given ECG signal into Arrhythmia(ARR) Congestive Heart Failure (CHF)  Normal Sinus Rhythm (NSR) categories. The project is combination of both Signal Processing and Computer Vision domains. 

- Continuos Wavelet Transformation was used to generate the spectrograms which were used as input data, to perform a Image-Classification using 4 different Convolutional Neural Networks. 
- A GUI is built using PyQt, where tasks like Visualization, training of models, Loading and saving of weights, prediction tasks can be performed.

A short demo of the operation of the GUI can be seen below.

- ```pip install requirements_CPU.txt``` for running CPU version of PyTorch library
- ```pip install requirements_GPU.txt``` for running CPU version of PyTorch library
- Run [mainGui.py](mainGui.py) to load the GUI.
- The weights are not included as the size exceeds Github limit. Model needs to be trained first to save the weights and pridict.



https://user-images.githubusercontent.com/75990547/159910104-0e60b8f7-a583-4154-bee7-8135df199aa7.mp4



       
## Signal Preprocessing

The file [createDataset.py](createDataset.py) is used to preprocess the signal and to create wavelets. The input signal is prone to baseline wandering and powerline noise. A filter-bank of a high pass filter and a notch filter are created to remove the baseline-wander and powerline noise.

```python
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
```
## Features
----
#### Signal Processing:
- Signal Base wander removal
- Powerline noise removal
- Scalogram (Dataset) creation
#### Neural Networsk:
- Found in Models folder
- Training calss-weights to reduce skewness in the dataloader batch.
- Training on both GPU and CPU libraries
#### GUI
- Visualize original and corrected signals of all classes, along with corresponding Scalograms
- Train a model with choosable Learning rate, Batch Size, Number of epochs.
- Train the model. If happy with the results save weights.
- Confusion matrix, Train and Validation loss as well as accuracy plots
- Model Statistics, training parameters display, classwise accuracy display
- Prediction using scalogram
