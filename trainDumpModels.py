""""

import pandas as pd
import sqlite3
import joblib
from joblib import dump, load
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

def getDfs():
    con = sqlite3.connect("dataMain.db")
    df = pd.read_sql_query("SELECT * from OZELLIK", con)
    le = preprocessing.LabelEncoder()
    le.fit(df['Labels'])
    labelEncoded = le.transform(df['Labels'])
    df['LabelEncoded'] = labelEncoded
    #scaler = StandardScaler()
    #dump(scaler, 'scaler.save')
    y = df['LabelEncoded']
    X = df.drop(['LabelEncoded','Labels'], axis=1)
    #namesDf = X.loc[:, ['FileNames']]
    #X_scaled = pd.DataFrame(scaler.fit_transform(X.iloc[:,1:799]),columns = (X.drop('FileNames', axis=1)).columns)
    #X_scaled = pd.concat([namesDf,X_scaled], axis=1)
    X.set_index('FileNames',inplace=True)
    #from sklearn.feature_selection import SelectKBest, f_classif
    #selecter = SelectKBest(f_classif, k=200)
    #selecter.fit_transform(X_scaled, y)
    #cols = selecter.get_support(indices=True)
    #df_new = X_scaled.iloc[:,cols]
    #from sklearn.utils import shuffle
    #df_new, y = shuffle(df_new, y, random_state=1)
    con.close()
    #return df_new, y, cols
    return X, y

MLPClassifier = MLPClassifier(activation='logistic', max_iter=2000)
RFClassifier = RandomForestClassifier(criterion='entropy', n_estimators=50)
KNNClassifier = KNeighborsClassifier(n_neighbors=5, weights='distance')
LRClassifier = LogisticRegression(max_iter=200, solver='saga')
LDAClassifier = LinearDiscriminantAnalysis(solver='svd')
SVClassifier = SVC(C=50, degree=2, kernel='rbf')
GNBClassifier = GaussianNB()
GradBoostClf = GradientBoostingClassifier()
AdaBoostClf = AdaBoostClassifier()
LinearSVCClf = LinearSVC(max_iter=500000, multi_class='crammer_singer')


df_new, y = getDfs()
MLPClassifier.fit(df_new, y)
RFClassifier.fit(df_new, y)
KNNClassifier.fit(df_new, y)
LRClassifier.fit(df_new, y)
LDAClassifier.fit(df_new, y)
SVClassifier.fit(df_new, y)
GNBClassifier.fit(df_new, y)
GradBoostClf.fit(df_new, y)
AdaBoostClf.fit(df_new, y)
LinearSVCClf.fit(df_new, y)

dump(MLPClassifier, 'mlp.joblib')
dump(RFClassifier, 'rfc.joblib')
dump(KNNClassifier, 'knn.joblib')
dump(LRClassifier, 'lrc.joblib')
dump(LDAClassifier, 'lda.joblib')
dump(SVClassifier, 'svc.joblib')
dump(GNBClassifier, 'gnb.joblib')
dump(AdaBoostClf, 'adaboost.joblib')
dump(LinearSVCClf, 'linearsvc.joblib')
dump(GradBoostClf, 'gradboost.joblib')

"""


