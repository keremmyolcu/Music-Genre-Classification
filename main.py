# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


import librosa as lbr
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pathlib
import csv
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import sqlite3
from scipy.stats import kurtosis
from scipy.stats import skew


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot
import sys
from PyQt5.QtGui import QIcon
import pygame


import joblib
from joblib import dump, load
from sklearn.preprocessing import StandardScaler

MLPClassifier = load('mlp.joblib')
RFClassifier = load('rfc.joblib')
KNNClassifier = load('knn.joblib')
LRClassifier = load('lrc.joblib')
LDAClassifier = load('lda.joblib')
SVClassifier = load('svc.joblib')
GNBClassifier = load('gnb.joblib')
AdaBoostClf = load('adaboost.joblib')
LinearSVCClf = load('linearsvc.joblib')
GradBoostClf = load('gradboost.joblib')
wavPath = ''

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        #Form.resize(1123, 823)
        Form.setFixedSize(1123, 823)
        self.showWavPath = QtWidgets.QTextEdit(Form)
        self.showWavPath.setGeometry(QtCore.QRect(70, 100, 251, 31))
        self.showWavPath.setObjectName("showWavPath")
        self.uploadButton = QtWidgets.QPushButton(Form)
        self.uploadButton.setGeometry(QtCore.QRect(330, 100, 41, 31))
        self.uploadButton.setObjectName("uploadButton")
        self.songList = QtWidgets.QListWidget(Form)
        self.songList.setGeometry(QtCore.QRect(70, 250, 401, 411))
        self.songList.setObjectName("songList")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(640, 50, 351, 41))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.playButton = QtWidgets.QPushButton(Form)
        self.playButton.setGeometry(QtCore.QRect(370, 100, 51, 31))
        self.playButton.setObjectName("playButton")
        self.classifyButton = QtWidgets.QPushButton(Form)
        self.classifyButton.setGeometry(QtCore.QRect(630, 670, 401, 61))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.classifyButton.setFont(font)
        self.classifyButton.setObjectName("classifyButton")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(70, 40, 231, 61))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(620, 290, 361, 51))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(70, 180, 341, 51))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.mlBox = QtWidgets.QComboBox(Form)
        self.mlBox.setGeometry(QtCore.QRect(640, 90, 341, 31))
        self.mlBox.setObjectName("mlBox")
        self.pauseButton = QtWidgets.QPushButton(Form)
        self.pauseButton.setGeometry(QtCore.QRect(420, 100, 31, 31))
        self.pauseButton.setObjectName("pauseButton")
        self.showGenreText = QtWidgets.QTextEdit(Form)
        self.showGenreText.setGeometry(QtCore.QRect(620, 350, 431, 301))
        font = QtGui.QFont()
        font.setPointSize(36)
        self.showGenreText.setFont(font)
        self.showGenreText.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.showGenreText.setObjectName("showGenreText")
        self.onerButton = QtWidgets.QPushButton(Form)
        self.onerButton.setGeometry(QtCore.QRect(140, 680, 221, 71))
        font = QtGui.QFont()
        font.setPointSize(19)
        self.onerButton.setFont(font)
        self.onerButton.setObjectName("onerButton")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("Form", "Music Genre Classifier and Recommender"))
        self.uploadButton.setText(_translate("Form", "->"))
        self.label_3.setText(_translate("Form", "Makine Öğrenmesi Yöntemi"))
        self.playButton.setText(_translate("Form", "Oynat"))
        self.classifyButton.setText(_translate("Form", "Sınıflandır"))
        self.label.setText(_translate("Form", "Müzik Dosyası Seç"))
        self.label_4.setText(_translate("Form", "Müzik örneğinin türü:"))
        self.label_2.setText(_translate("Form", "Şarkı Öner"))
        self.pauseButton.setText(_translate("Form", "Dur"))
        self.showGenreText.setHtml(_translate("Form",
                                              "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                              "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                              "p, li { white-space: pre-wrap; }\n"
                                              "</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:26pt; font-weight:400; font-style:normal;\">\n"
                                              "<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
                                              "<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.onerButton.setText(_translate("Form", "Öner"))


        self.mlBox.addItem("KNN")
        self.mlBox.addItem("MLP")
        self.mlBox.addItem("Linear SVC")
        self.mlBox.addItem("Gaussian NB")
        self.mlBox.addItem("LDA Sınıflandırıcı")
        self.mlBox.addItem("Support Vector Sınıflandırıcı")
        self.mlBox.addItem("Random Forest")
        self.mlBox.addItem("AdaBoost")
        self.mlBox.addItem("Gradient Boosting")
        self.mlBox.addItem("Lojistik Regresyon")

        self.playButton.clicked.connect(self.play)
        self.pauseButton.clicked.connect(self.stop)
        self.uploadButton.clicked.connect(self.browseFile)
        self.classifyButton.clicked.connect(self.classify)
        self.onerButton.clicked.connect(self.recommend)

    def browseFile(self):
        filter = "wav(*.wav)"
        global wavPath
        fpath = QtWidgets.QFileDialog.getOpenFileName(None, caption='Select .wav File', directory='C:\\', filter=filter)
        self.showWavPath.setText(fpath[0])
        wavPath = fpath[0]

    def ftrStats(self, ftr):
        st1 = np.mean(ftr, axis=1)
        st2 = np.std(ftr, axis=1)
        st3 = skew(ftr, axis=1)
        st4 = kurtosis(ftr, axis=1)
        st5 = np.median(ftr, axis=1)
        st6 = np.min(ftr, axis=1)
        st7 = np.max(ftr, axis=1)
        ret = np.stack((st1, st2, st3, st4, st5, st6, st7), axis=0)
        return ret.flatten()

    def retFtrArr(self, path):
        y, sr = lbr.load(path, mono=True, duration=30)
        Y = lbr.stft(y)
        S, phase = lbr.magphase(Y)
        zcr = lbr.feature.zero_crossing_rate(y)
        remese = lbr.feature.rms(y=y)
        spcent = lbr.feature.spectral_centroid(y=y, sr=sr)
        spbw = lbr.feature.spectral_bandwidth(y=y, sr=sr)
        spcont = lbr.feature.spectral_contrast(S=S, sr=sr)
        spflat = lbr.feature.spectral_flatness(S=S)
        sprollof = lbr.feature.spectral_rolloff(S=S, sr=sr)
        mfcc = lbr.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = lbr.feature.chroma_stft(y=y, sr=sr)
        ZCRStats = self.ftrStats(zcr)
        RMSStats = self.ftrStats(remese)
        SpcentStats = self.ftrStats(spcent)
        SpbandStats = self.ftrStats(spbw)
        SpcontStats = self.ftrStats(spcont)
        SpflatStats = self.ftrStats(spflat)
        SprofStats = self.ftrStats(sprollof)
        MfccStats = self.ftrStats(mfcc)
        ChromaStats = self.ftrStats(chroma)
        finalndarr = np.concatenate((ZCRStats,
                                     RMSStats,
                                     SpcentStats,
                                     SpbandStats,
                                     SpcontStats,
                                     SpflatStats,
                                     SprofStats,
                                     MfccStats,
                                     ChromaStats), axis=0)
        return finalndarr

#Siniflandirma tusu butonu metodu
    def classify(self):
        LabelsCodes = {0: 'Blues', 1: 'Classical', 2: 'Country', 3: 'Disco', 4: 'Hiphop', 5: 'Jazz', 6: 'Metal', 7: 'Pop',
                       8: 'Raggae', 9: 'Rock'}

        if (wavPath != ''):
            ftrarr = self.retFtrArr(wavPath)
            ftrarr = ftrarr.reshape(1, -1)
            if (self.mlBox.currentText() == 'Gaussian NB'):
                i = GNBClassifier.predict(ftrarr)[0]
                self.showGenreText.setText(LabelsCodes[i])
                pass

            if (self.mlBox.currentText() == 'KNN'):
                i = KNNClassifier.predict(ftrarr)[0]
                self.showGenreText.setText(LabelsCodes[i])
                pass

            if (self.mlBox.currentText() == 'MLP'):
                arr = MLPClassifier.predict(ftrarr)[0]
                self.showGenreText.setText(LabelsCodes[arr])
                pass

            if (self.mlBox.currentText() == 'LDA Sınıflandırıcı'):
                i = LDAClassifier.predict(ftrarr)[0]
                self.showGenreText.setText(LabelsCodes[i])
                pass

            if (self.mlBox.currentText() == 'Linear SVC'):
                i = LinearSVCClf.predict(ftrarr)[0]
                self.showGenreText.setText(LabelsCodes[i])
                pass

            if (self.mlBox.currentText() == 'Support Vector Sınıflandırıcı'):
                i = SVClassifier.predict(ftrarr)[0]
                self.showGenreText.setText((LabelsCodes[i]))
                pass

            if (self.mlBox.currentText() == 'Random Forest'):
                i = RFClassifier.predict(ftrarr)[0]
                self.showGenreText.setText((LabelsCodes[i]))
                pass

            if(self.mlBox.currentText() == 'AdaBoost'):
                i = AdaBoostClf.predict(ftrarr)[0]
                self.showGenreText.setText((LabelsCodes[i]))
                pass

            if(self.mlBox.currentText() == 'Gradient Boosting'):
                i = GradBoostClf.predict(ftrarr)[0]
                self.showGenreText.setText((LabelsCodes[i]))
                pass

            if(self.mlBox.currentText() == 'Lojistik Regresyon'):
                i = LRClassifier.predict(ftrarr)[0]
                self.showGenreText.setText((LabelsCodes[i]))
                pass

#Sarki onerme tusu butonu
    def recommend(self):
        self.songList.clear()
        from sklearn.neighbors import NearestNeighbors
        con = sqlite3.connect("dataMain.db")
        df = pd.read_sql_query("SELECT * from {}".format('OZELLIK'), con)
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(df['Labels'])
        labelEncoded = le.transform(df['Labels'])
        df['LabelEncoded'] = labelEncoded
        y = df['LabelEncoded']
        X = df.drop(['LabelEncoded', 'Labels'], axis=1)
        X.set_index('FileNames', inplace=True)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        ftrarr = self.retFtrArr(wavPath)
        ftrarr = ftrarr.reshape(1, -1)
        ftrarr_scaled = scaler.transform(ftrarr)
        bestSelect = SelectKBest(k=50)
        X_new = bestSelect.fit_transform(X_scaled, y)
        ftr_new = bestSelect.transform(ftrarr_scaled)
        neighbours = NearestNeighbors(n_neighbors=5)
        neighbours.fit(X_new)
        indices = neighbours.kneighbors(ftr_new, return_distance=False)
        indices = indices.reshape(-1)
        X['FileNames'] = X.index
        for i in range(0,5):
            self.songList.addItem(X['FileNames'][indices].values[i])
        con.close()


#Muzik oynatma tusu metodu
    def play(self):
        if(wavPath != ''):
            pygame.init()
            pygame.mixer.init()
            channel1 = pygame.mixer.Channel(0)
            click = pygame.mixer.Sound(wavPath)
            channel1.play(click)


#Muzik durdurma tusu metodu
    def stop(self):
        pygame.mixer.pause()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

