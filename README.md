# ECG-Signal-Classification

This project focuses on training neural networks to classify the given ECG signal into Arrhythmia(ARR) Congestive Heart Failure (CHF)  Normal Sinus Rhythm (NSR) categories. The project is combination of both Signal Processing and Computer Vision domains. 

- Continuos Wavelet Transformation was used to generate the spectrograms which were used as input data, to perform a Image-Classification using 4 different Convolutional Neural Networks. 
- A GUI is built using PyQt, where tasks like Visualization, training of models, Loading and saving of weights, prediction tasks can be performed.

A short demo of the operation of the GUI can be seen below.


https://user-images.githubusercontent.com/75990547/159734552-6b23bf35-13de-416b-831c-2b3641b07d6a.mp4


       
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
As there are varying number of ECG signal for each class of 
