#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install librosa')
import librosa as lbr
import librosa.display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import csv
import sklearn
import scipy.stats as stats


# In[137]:


HOP_SIZE=512
FRAME_SIZE=2048
SAMPLE_RATE = 44100
from scipy.stats import kurtosis
from scipy.stats import skew


# In[2]:


np.set_printoptions(suppress=True)


# In[229]:


def ftrStats(ftr):
    st1=np.mean(ftr,axis=1)
    st2=np.std(ftr,axis=1)
    st3=skew(ftr,axis=1)
    st4=kurtosis(ftr,axis=1)
    st5=np.median(ftr,axis=1)
    st6=np.min(ftr,axis=1)
    st7=np.max(ftr,axis=1)
    ret = np.stack((st1,st2,st3,st4,st5,st6,st7),axis=0)
    return ret.flatten()


# In[169]:


stats = ['Mean','Std','Skw','Krt','Med','Min','Max']
featNames = ['ZCR','RMS','Spcent','Spband','Sprof','Spflat','Spcont','Mfcc','Chroma']
hepsi = []
for feat in featNames:
    for stat in stats:
        if(feat !='Mfcc' and feat !='Chroma' and feat!='Spcont'):
            hepsi.append(feat+stat)
        elif(feat=='Mfcc'):
            for i in range(0,13):
                ekl = "%s" %i
                hepsi.append(feat+ekl+stat)
        elif(feat=='Chroma'):
            for i in range(0,12):
                ekl = "%s" %i
                hepsi.append(feat+ekl+stat)
        elif(feat=='Spcont'):
            for i in range(0,7):
                ekl = "%s" %i
                hepsi.append(feat+ekl+stat)
        
#(266,0)


# In[215]:


hepsi


# In[168]:


turevHepsi = []
turev = ['Del1','Del2']
for n in turev:
    for feat in featNames:
        for stat in stats:
            if(feat !='Mfcc' and feat !='Chroma' and feat!='Spcont'):
                turevHepsi.append(n+feat+stat)
            elif(feat=='Mfcc'):
                for i in range(0,13):
                    ekl = "%s" %i
                    turevHepsi.append(n+feat+ekl+stat)
            elif(feat=='Chroma'):
                for i in range(0,12):
                    ekl = "%s" %i
                    turevHepsi.append(n+feat+ekl+stat)
            elif(feat=='Spcont'):
                for i in range(0,7):
                    ekl = "%s" %i
                    turevHepsi.append(n+feat+ekl+stat)
#(532,0)


# In[232]:


def retFtrArr(file):
    y, sr = lbr.load(file,mono=True,duration=30) 
    Y = lbr.stft(y)
    S, phase = lbr.magphase(Y)  
    zcr = lbr.feature.zero_crossing_rate(y)
    remese = lbr.feature.rms(y=y)  
    spcent = lbr.feature.spectral_centroid(y=y, sr=sr)
    spbw = lbr.feature.spectral_bandwidth(y=y, sr=sr)
    spcont = lbr.feature.spectral_contrast(S=S, sr=sr)
    spflat = lbr.feature.spectral_flatness(S=S)
    sprollof = lbr.feature.spectral_rolloff(S=S, sr=sr)
    mfcc = lbr.feature.mfcc(y=y, sr=sr,n_mfcc=13)
    chroma = lbr.feature.chroma_stft(y=y, sr=sr)
    ZCRStats = ftrStats(zcr)
    RMSStats = ftrStats(remese)
    SpcentStats = ftrStats(spcent)
    SpbandStats = ftrStats(spbw)
    SpcontStats = ftrStats(spcont)
    SpflatStats = ftrStats(spflat)
    SprofStats = ftrStats(sprollof)
    MfccStats = ftrStats(mfcc)
    ChromaStats = ftrStats(chroma)
    finalndarr = np.concatenate((ZCRStats,
                                RMSStats,
                                SpcentStats,
                                SpbandStats,
                                SpcontStats,
                                SpflatStats,
                                SprofStats,
                                MfccStats,
                                ChromaStats),axis=0)
    return finalndarr


# In[3]:


os.chdir(r'C:\Users\User\Desktop\ARAPROJE\gtzan\Data\genres')


# In[285]:


def gtzanToCSV(method1, cols, nums):
    data = np.zeros(nums)
    labels = []
    files = []
    anadrct = os.getcwd()
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    for g in genres:
        for filename in os.scandir(r'.\{}'.format(g)):   #first chdir to directory where genres files exists
            directo = os.path.join(anadrct,g,filename.name)
            ekle = method1(directo)
            data = np.vstack((data, ekle))
            files.append(filename.name)
            labels.append((filename.name).split(".")[0])
    data = np.delete(data,(0),axis=0)
    filenames = pd.DataFrame(files, columns=['FileNames'])
    finaldf = pd.DataFrame(data,columns=cols,index = files)
    finaldf['Labels'] = labels
    finaldf = pd.concat([filenames, finaldf], axis=1)
    return finaldf


# In[265]:


finaldefe = gtzanToCSV(retFtrArr, hepsi, 267)


# In[269]:


finaldefe.to_csv(r'C:\Users\User\Desktop\features.csv', index=False)


# In[276]:


def derivFtrs(file):
    y, sr = lbr.load(file,mono=True,duration=30) 
    Y = lbr.stft(y)
    S, phase = lbr.magphase(Y)  
    zcr = lbr.feature.zero_crossing_rate(y)
    zcrDel1=librosa.feature.delta(zcr)
    zcrDel2=librosa.feature.delta(zcr,order=2)
    remese = lbr.feature.rms(y=y)  
    rmsDel1=librosa.feature.delta(remese)
    rmsDel2=librosa.feature.delta(remese)
    
    spcent = lbr.feature.spectral_centroid(y=y, sr=sr)
    spcentDel1=librosa.feature.delta(spcent)
    spcentDel2=librosa.feature.delta(spcent,order=2)
    
    spbw = lbr.feature.spectral_bandwidth(y=y, sr=sr)
    spbwDel1=librosa.feature.delta(spbw)
    spbwDel2=librosa.feature.delta(spbw,order=2)
    
    spcont = lbr.feature.spectral_contrast(S=S, sr=sr)
    spcontDel1=librosa.feature.delta(spcont)
    spcontDel2=librosa.feature.delta(spcont,order=2)
    
    spflat = lbr.feature.spectral_flatness(S=S)
    spflatDel1 = librosa.feature.delta(spflat)
    spflatDel2 = librosa.feature.delta(spflat,order=2)
    
    sprollof = lbr.feature.spectral_rolloff(S=S, sr=sr)
    sprollofDel1=librosa.feature.delta(sprollof)
    sprollofDel2=librosa.feature.delta(sprollof,order=2)
    
    mfcc = lbr.feature.mfcc(y=y, sr=sr,n_mfcc=13)
    mfccDel1 = librosa.feature.delta(mfcc)
    mfccDel2 = librosa.feature.delta(mfcc,order=2)
    
    chroma = lbr.feature.chroma_stft(y=y, sr=sr)
    chromaDel1=librosa.feature.delta(chroma)
    chromaDel2=librosa.feature.delta(chroma,order=2)
    
    ZCR1Stats = ftrStats(zcrDel1)
    ZCR2Stats = ftrStats(zcrDel2)
    
    RMS1Stats = ftrStats(rmsDel1)
    RMS2Stats = ftrStats(rmsDel2)
    
    Spcent1Stats = ftrStats(spcentDel1)
    Spcent2Stats = ftrStats(spcentDel2)
    
    Spband1Stats = ftrStats(spbwDel1)
    Spband2Stats = ftrStats(spbwDel2)
    
    Spcont1Stats = ftrStats(spcontDel1)
    Spcont2Stats = ftrStats(spcontDel2)
    
    Spflat1Stats = ftrStats(spflatDel1)
    Spflat2Stats = ftrStats(spflatDel2)
    
    Sprof1Stats = ftrStats(sprollofDel1)
    Sprof2Stats = ftrStats(sprollofDel2)
    
    Mfcc1Stats = ftrStats(mfccDel1)
    Mfcc2Stats = ftrStats(mfccDel2)
    
    Chroma1Stats = ftrStats(chromaDel1)
    Chroma2Stats = ftrStats(chromaDel2)
    
    finalndarr = np.concatenate((ZCR1Stats,
                                RMS1Stats,
                                Spcent1Stats,
                                Spband1Stats,
                                Spcont1Stats,
                                Spflat1Stats,
                                Sprof1Stats,
                                Mfcc1Stats,
                                Chroma1Stats,
                                ZCR2Stats,
                                RMS2Stats,
                                Spcent2Stats,
                                Spband2Stats,
                                Spcont2Stats,
                                Spflat2Stats,
                                Sprof2Stats,
                                Mfcc2Stats,
                                Chroma2Stats),axis=0)
    return finalndarr


# In[286]:


derivsarr = gtzanToCSV(derivFtrs,turevHepsi,532)


# In[289]:


derivsarr.to_csv(r'C:\Users\User\Desktop\featuresDeriv.csv', index=False)


# In[11]:





# In[ ]:




