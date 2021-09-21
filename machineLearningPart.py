#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import csv
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


#PCA,Kernel PCA ile feature secip ML algoritmalarının performanslarını
#ölç


# In[69]:


df1 = pd.read_csv(r'C:\Users\User\Desktop\featuresSon.csv')  
df2 = pd.read_csv(r'C:\Users\User\Desktop\featuresDeriv.csv')
df2 = df2.drop(['Labels'], axis=1)
df = pd.concat([df1,df2], axis=1)
df.head()


# In[70]:


le = preprocessing.LabelEncoder()
le.fit(df['Labels'])


# In[71]:


labelEncoded = le.transform(df['Labels'])


# In[72]:


df['LabelEncoded'] = labelEncoded
df.head()


# In[76]:


scaler = StandardScaler()
y = df['LabelEncoded']
X = df.drop(['LabelEncoded','Labels'], axis=1)


# In[74]:


namesDf = X.loc[:, ['FileNames']]
X_scaled = pd.DataFrame(scaler.fit_transform(X.iloc[:,1:799]),columns = (X.drop('FileNames', axis=1)).columns)
X_scaled = pd.concat([namesDf,X_scaled], axis=1)
X_scaled.set_index('FileNames',inplace=True)
X_scaled.head()


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)


# In[9]:


def plotVariance():
    pca = PCA().fit(X_train)
    plt.rcParams["figure.figsize"] = (20,6)

    fig, ax = plt.subplots()
    xi = np.arange(0, 798, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)

    plt.ylim(0.0,1.1)
    plt.plot(xi, y, linestyle='--', color='b')

    plt.xlabel('Number of Components')
    plt.xticks(np.arange(0, 798, step=20)) #change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance')

    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.text(0.5, 0.85, '95% cut-off threshold', color = 'red', fontsize=16)

    ax.grid(axis='x')
    plt.show()
plotVariance()


# In[10]:


pca = PCA(n_components=260, random_state=0)
lda = LinearDiscriminantAnalysis(n_components=9)
kpca = KernelPCA(n_components=260, kernel='linear')


# In[11]:


X_train_pca = pca.fit(X_train) #(900,260) PCA train data
X_train_pcatrf = X_train_pca.transform(X_train)
X_test_pcatrf = X_train_pca.transform(X_test)  #(100,260) PCA test data

X_train_lda = lda.fit(X_train,y_train) #(900,9) LDA train data
X_train_ldatrf = X_train_lda.transform(X_train)
X_test_ldatrf = lda.fit(X_test,y_test).transform(X_test)  #(100,9) LDA test data

X_train_kpca = kpca.fit(X_train).transform(X_train)  #(900,260) KPCA train data
X_train_kpcatrf = X_train_kpca.transform(X_train)
X_test_kpca = X_train_kpca.transform(X_test)    #(100,260) KPCA test data


# In[63]:


def showMLPerformances(X, y):
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    models = [MLPClassifier(),
           LogisticRegression(),
           RandomForestClassifier(),
           LinearDiscriminantAnalysis(),
           KNeighborsClassifier(),
           SVC(),
           GaussianNB(),
           GradientBoostingClassifier(),
           AdaBoostClassifier(),
           LinearSVC(),
           SVC()]
    
    
    scoring = ['accuracy','recall_macro','precision_macro']
    
    classifierNames = ['MLP', 'LogisticRegression','RandomForest','LDA','KNN','SVC','GaussianNB','GradBoost'
                       ,'AdaBoost','LinearSVC']
    
    
    param_grid = {'MLP' : [{'max_iter': [500,1000,2000], 'activation': ['tanh','logistic','identity']}],
                  'LogisticRegression' : [{'solver': ['newton-cg', 'lbfgs','saga'], 'max_iter': [200,500,1000]}],
                  'RandomForest' : [{'n_estimators': [50,100,150], 'criterion': ['gini', 'entropy']}],
                  'LDA' : [{'solver': ['eigen', 'svd', 'lsqr']}],
                  'KNN' : [{'n_neighbors': [5,10,20], 'weights': ['uniform', 'distance']}],
                  'SVC' : [{'kernel': ['poly', 'rbf', 'sigmoid'], 'degree': [2,3], 'C': [1, 50, 100]}],
                  'GaussianNB' : [{}],
                  'GradBoost' : [{}],
                  'AdaBoost' : [{}],
                  'LinearSVC' : [{'max_iter': [500000,1000000], 'multi_class': ['ovr', 'crammer_singer']}]}
    
    
    for cname, clf in zip(classifierNames, models):
        print('SCORES FOR '+cname)
        GridSearch = GridSearchCV(estimator=clf, param_grid=param_grid[cname], scoring=scoring, refit= 'accuracy', n_jobs=-1, cv=10)
        GridSearch.fit(X, y)
        i = GridSearch.best_index_
        best_precision = GridSearch.cv_results_['mean_test_precision_macro'][i]
        best_recall = GridSearch.cv_results_['mean_test_recall_macro'][i]
        print('Best parameters:')
        print(GridSearch.best_params_)
        print('gaves accuracy: %.4f' %(GridSearch.best_score_))
        print('gives recall: %.4f' %(best_recall))
        print('gives precision: %.4f' %(best_precision))
        print('++++++++++++++++++++++++++++')


# In[54]:


MFCCDf = X_scaled.filter(like='Mfcc', axis=1)
Specs = X_scaled.filter(like='Sp', axis=1)
SpecDf = pd.concat([MFCCDf,Specs], axis = 1)
ZCRCols = X_scaled.filter(like='ZCR', axis=1)
RMSCols = X_scaled.filter(like='RMS', axis=1)
TempDf = pd.concat([ZCRCols,RMSCols], axis=1)
ChromaDf = X_scaled.filter(like='Chroma', axis=1)


# In[228]:


from sklearn.feature_selection import SelectKBest, f_classif 
selecter = SelectKBest(f_classif, k=100)
selecter.fit(X_scaled, y)
cols = selecter.get_support(indices=True)
df_new = X_scaled.iloc[:,cols]
df_new.head()


# In[229]:


from sklearn.utils import shuffle
df_new, y = shuffle(df_new, y, random_state=0)


# In[ ]:





# In[119]:





# In[118]:


showMLPerformances(df_new, y)


# In[64]:


showMLPerformances(SpecDf, y) 


# In[66]:


showMLPerformances(X_scaled, y) 


# In[ ]:


from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=3, n_jobs=-1)
model_knn.


# In[81]:


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


# In[82]:


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


# In[124]:


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


# In[231]:


X_scaled


# In[236]:


from sklearn.model_selection import cross_validate
models=[MLPClassifier(max_iter=2000, activation='logistic'),
           LogisticRegression(dual=False,multi_class='auto',solver='saga',random_state=7, max_iter=200),
           RandomForestClassifier(n_estimators=50,criterion='entropy'),
           LinearDiscriminantAnalysis(solver='svd'),
           KNeighborsClassifier(n_neighbors=5, weights='distance'),
           SVC(kernel='rbf',degree=2,C=50),
           GaussianNB(),
           GradientBoostingClassifier(),
           AdaBoostClassifier(),
           LinearSVC(multi_class='crammer_singer',max_iter=500000)]

scoring = ["accuracy","precision_macro","recall_macro"]

for i in range(len(models)):
    kf = KFold(n_splits=10, shuffle = True)
    score = cross_validate(models[i], X_scaled, labelEncoded, cv=kf, scoring=scoring)
    print(models[i])
    print('accuracy:')
    print(np.mean(score['test_accuracy']))
    print('recall:')
    print(np.mean(score['test_recall_macro']))
    print('precision:')
    print(np.mean(score['test_precision_macro']))


# In[243]:


for i in range(len(models)):
    kf = KFold(n_splits=10, shuffle = True)
    score = cross_validate(models[i], TempDf, labelEncoded, cv=kf, scoring=scoring)
    print(models[i])
    print('accuracy:')
    print(np.mean(score['test_accuracy']))
    print('recall:')
    print(np.mean(score['test_recall_macro']))
    print('precision:')
    print(np.mean(score['test_precision_macro']))


# In[244]:


pca = PCA(n_components=180, random_state=0)
X_pca = pca.fit_transform(X_scaled)
for i in range(len(models)):
    kf = KFold(n_splits=10, shuffle = True)
    score = cross_validate(models[i], X_pca, labelEncoded, cv=kf, scoring=scoring)
    print(models[i])
    print('accuracy:')
    print(np.mean(score['test_accuracy']))
    print('recall:')
    print(np.mean(score['test_recall_macro']))
    print('precision:')
    print(np.mean(score['test_precision_macro']))


# In[245]:


MfccChromaDf = pd.concat([MFCCDf,ChromaDf], axis = 1)


# In[247]:


for i in range(len(models)):
    kf = KFold(n_splits=10, shuffle = True)
    score = cross_validate(models[i], MfccChromaDf, labelEncoded, cv=kf, scoring=scoring)
    print(models[i])
    print('accuracy:')
    print(np.mean(score['test_accuracy']))
    print('recall:')
    print(np.mean(score['test_recall_macro']))
    print('precision:')
    print(np.mean(score['test_precision_macro']))


# In[ ]:




