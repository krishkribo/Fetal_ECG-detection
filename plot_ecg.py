
import matplotlib as m
import os
from matplotlib import pyplot as plt
import numpy as np
import pywt
import padasip
from time import sleep
from scipy import signal
from ssnf import ssnf

def hp_filter(inp_signal):
    sos = signal.butter(2, 0.6, btype='highpass', fs=1000, output='sos')
    filtered = signal.sosfilt(sos, inp_signal)
    return filtered

def get_data(files):
    d_point=[]
    with open(files+".txt") as file:
        fread=file.readlines()
        for data in fread:
            d_point.append(float(data))
    return d_point

# Normalize input data
def normalize_data(data):
    mean = sum(data) / len(data)
    data = [d - mean for d in data]
    m = max(data)
    return [d / m for d in data]


def plot_data(data,title,t_lst):
    fig,ax=plt.subplots(len(data))
    plt.subplots_adjust(hspace=1)
    fig.suptitle(title)
    for d in range(len(data)):
        ax[d].plot([x for x in range(0,len(data[d]))],data[d])
        ax[d].set_title(t_lst[d])
    plt.draw()

def subplot_data(data,title):
    fig,ax=plt.subplots(np.shape(data)[0])
    plt.subplots_adjust(hspace=1)
    fig.suptitle(title)
    for i in range(0,np.shape(data)[0]):
        ax[i].plot([x for x in range(0,len(data[i]))],data[i])
        ax[i].set_title("Filtered with thorax wavelet coefficients :"+str(i))
    plt.draw()

    
def subplot_data1(data,title):
    fig,ax=plt.subplots(np.shape(data)[0])
    plt.subplots_adjust(hspace=1)
    fig.suptitle(title)
    d=1
    for i in range(0,np.shape(data)[0]):
        ax[i].plot([x for x in range(0,len(data[i]))],data[i])
        if i%2==0 or i==0: j=1
        else : j=2
        ax[i].set_title("Abdome signal:"+str(d)+" Thorax signal:"+str(j))
        if j==2: d+=1
    plt.draw()

def swt(data,wstyle):
    #data=get_data(data)
    return  pywt.swt(normalize_data(data),wstyle,level=5,trim_approx=True)

def inv_swt(coeff,wstyle):
    return pywt.iswt(coeff,wstyle)

def lms_algo(d,x):
    filt=padasip.filters.FilterLMS(6,mu=0.01,w='random')
    y,e,w=filt.run(d,x)
    return y

def calculate_lms(input_signals,reference_signals):
    # input signal -- abdomen signal
    # reference signal -- thorax signal
    res=[]
    inp_data=np.transpose(input_signals)
    #print(np.shape(inp_data))
    for ref_data in reference_signals:
        ref_data=np.transpose(ref_data)
        y=[lms_algo(ref_data[:,i],inp_data) for i in range(0,np.shape(ref_data)[1])]
        res.append(y)

    return res



if __name__=="__main__":
    inp_data=[]
    data=os.listdir()
    ## implementing the adaptive filtering
    for i in range(0,len(data)):
        if len(data[i].split('.')) ==1 :
            continue
        if (data[i].split('.')[1])=='txt':
            inp_data.append((data[i].split('.'))[0])
    print(inp_data)
    #plot_data([get_data(data) for data in inp_data],"ECG signals",inp_data)
    print("-----> ECG data readed <-----")

    # pre processing
    # apply high pass filter
    hp_data_filtered=[hp_filter(get_data(data)) for data in inp_data]
    #plot_data(hp_data_filtered,"High pass filtered data",inp_data)
    print("-----> High pass filter applied <-----")

    # step 1 - process the signal by the stationary wavelet transfrom method
    wavelet_data=[swt(data,pywt.Wavelet("bior1.5")) for data in hp_data_filtered]
    print("-----> wavelet tranformed <-----")

    # step 2 - filter the wavelet coefficients obtained from the previous step

    # send abdomen1 data as the input signal and thorax signals as the reference
    filter_abd1_data=calculate_lms(wavelet_data[0],wavelet_data[3:])
    subplot_data(filter_abd1_data[0],"Input data: Abdomen signal 1 ; Reference data: thorax_signal 1")
    subplot_data(filter_abd1_data[1],"Input data: Abdomen signal 1 ; Reference data: thorax_signal 2")
    print("-----> Abdomen 1 data filtered <-----")
    
    # send abdomen2 data as the input signal and thorax signals as the reference
    filter_abd2_data=calculate_lms(wavelet_data[1],wavelet_data[3:])
    subplot_data(filter_abd2_data[0],"Input data: Abdomen signal 2 ; Reference data: thorax_signal 1")
    subplot_data(filter_abd2_data[1],"Input data: Abdomen signal 2 ; Reference data: thorax_signal 2")
    print("-----> Abdomen 2 data filtered <-----")
    
    # send abdomen3 data as the input signal and thorax signals as the reference
    filter_abd3_data=calculate_lms(wavelet_data[2],wavelet_data[3:])
    subplot_data(filter_abd3_data[0],"Input data: Abdomen signal 3 ; Reference data: thorax_signal 1")
    subplot_data(filter_abd3_data[1],"Input data: Abdomen signal 3 ; Reference data: thorax_signal 2")
    print("-----> Abdomen 3 data filtered <-----")
    
    print("----------------------------- SSNF filteration ---------------------------------")
    ssnf_abd1_1=ssnf(filter_abd1_data[0],5,5*[10])
    ssnf_abd1_2=ssnf(filter_abd1_data[1],5,5*[10])
    subplot_data(ssnf_abd1_1,"SSNF applied Input data: Abdomen signal 1 ; Reference data: thorax_signal 1")
    subplot_data(ssnf_abd1_2,"SSNF applied Input data: Abdomen signal 1 ; Reference data: thorax_signal 2")
    ssnf_abd1=[ssnf_abd1_1,ssnf_abd1_2]
    ssnf_abd2_1=ssnf(filter_abd2_data[0],5,5*[10])
    ssnf_abd2_2=ssnf(filter_abd2_data[1],5,5*[10])
    subplot_data(ssnf_abd2_1,"SSNF applied Input data: Abdomen signal 2 ; Reference data: thorax_signal 1")
    subplot_data(ssnf_abd2_2,"SSNF applied Input data: Abdomen signal 2 ; Reference data: thorax_signal 2")
    ssnf_abd2=[ssnf_abd2_1,ssnf_abd2_2]
    ssnf_abd3_1=ssnf(filter_abd3_data[0],5,5*[10])
    ssnf_abd3_2=ssnf(filter_abd3_data[1],5,5*[10])
    subplot_data(ssnf_abd3_1,"SSNF applied Input data: Abdomen signal 3 ; Reference data: thorax_signal 1")
    subplot_data(ssnf_abd3_2,"SSNF applied Input data: Abdomen signal 3 ; Reference data: thorax_signal 2")
    ssnf_abd3=[ssnf_abd3_1,ssnf_abd3_2]

    filterd_data=[filter_abd1_data,filter_abd2_data,filter_abd3_data]
    
    # Inverse wavelet of abdomen 1 signal
    n_data=np.zeros((np.shape(filterd_data)[2],np.shape(filterd_data)[3]))
    count=0;
    for i in range(len(filterd_data)):
        for j in range(len(filterd_data[1])):
            inv_wav=inv_swt(filterd_data[i][j],pywt.Wavelet("bior1.5"))
            n_data[count]=inv_wav
            count=count+1

    subplot_data1(n_data,"inverse wavelet transformed data")
    print("-----> Inverse wavelet transform applied <-----")

    plt.show()
