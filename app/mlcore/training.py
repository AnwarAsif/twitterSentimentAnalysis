# from app.mlcore.models import custom_model
from pandas.core import base
from data import *
from models import *
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np 
from joblib import dump
from collections import Counter
from imblearn.combine import SMOTEENN
from sklearn.feature_extraction.text import TfidfVectorizer



def evaluate_clf(vec, clf, df):
    start = timer()
    testing_data_text_sparse = vec.transform(df["sentence"])
    pred = clf.predict(testing_data_text_sparse)
    end = timer()
    print ("Time to apply classifier on {} data points: {} seconds".format(df.shape[0], end - start))

    print ("\nReport:")
    print (classification_report(df["label"], pred, digits = 4))

def vectorize_data(df):
    
    vec = TfidfVectorizer(stop_words = 'english', max_features = 5000)
    training_data_text_sparse = vec.fit_transform(df["sentence"])
    
    return vec, training_data_text_sparse


def oversample(X_train, y_train):
    smote_enn = SMOTEENN(random_state=0)
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
    print(sorted(Counter(y_train).items()))
    print(sorted(Counter(y_resampled).items()))

    return X_resampled, y_resampled

if __name__ == "__main__":
    # Load dataset 
    train, test = read_data()

    # Pre processed data 
    pross_train = preprocess_data(train)
    vec, X_train = vectorize_data(pross_train)
    y_train = train['label']
    X_test = vec.transform(test['sentence'])
    y_test = test['label']   

    # Training the base model 
    base_clf = base_model(X_train, y_train)
    evaluate_clf(vec, base_clf, test) 
    
    # Over sampling the Negative class
    X_resampled, y_resampled = oversample(X_train, y_train)

    # Best model manually found 
    best_clf = custom_model(X_resampled, y_resampled)
    evaluate_clf(vec, best_clf, test) 

    # Grid Search 
    grid_clf = grid_search(X_resampled, y_resampled)
    evaluate_clf(vec, grid_clf, test) 


    # Test with a test sample tweet 
    sample_tweet = "In December alone , the members of the Lithuanian Brewers ' Association sold a total of 20.3 million liters of beer , an increase of 1.9 percent"
    sample_tweet = pd.Series(sample_tweet)
    sample_tweet = vec.transform(sample_tweet)
    result = best_clf.predict(sample_tweet)
    print(result)

    # same models 
    dump(vec, 'app/mlcore/vec.joblib')
    dump(base_clf,'app/mlcore/base.joblib')
    dump(best_clf, 'app/mlcore/best.joblib')
    dump(grid_clf, 'app/mlcore/grid.joblib')
    