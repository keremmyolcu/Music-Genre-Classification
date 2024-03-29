# Music Genre Classification
My goal is to extract the acoustic features(librosa library is used) of each .wav file of a music dataset(GTZAN is used) before using them for training my machine learning models(KNN, Random Forest, Decision Tree, Naive Bayes. scikit-learn is used); examine performance ratios after dimensionality reduction techniques(PCA,LDA,KPCA) and classifying the genre of a different music file that user has chosen. There are 10 music genres(Rock, Jazz, Classical etc.).\
\
\
1)Feature extraction(to .csv file) ✓ \
2)Training the models ✓ \
3)Classifying a music file - ✓\
4)Creating a user interface - ✓\
\
For dataset : https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification

Screenshot of the desktop application : ![image](https://user-images.githubusercontent.com/63878088/134176478-84b16371-616e-412a-9ff7-da3fbedf693e.png)
Video demonstration : https://www.youtube.com/watch?v=YgARwYtmpK8

Firstly, you must use the gtzanToCSV() method in the featureExtract.ipynb file(change your directory to the "genres" file in GTZAN dataset). After obtaining the .csv file, you can use its directory address in the csvToDb.py file which creates a database named "dataMain.db" in your Python program. To train and save the machine learning models, you must run the "trainDumpModels.py". After these processes, you can run the main.py and use the desktop application with no problem. To see the experimental results and hyperparameter tuning process, you can check the IPython files.
