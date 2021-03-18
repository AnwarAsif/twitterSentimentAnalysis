import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

def Logistic_clf(df):
    # clf = LogisticRegression(C = 1, random_state = 42, class_weight = "balanced")
    # return clf

    vec = TfidfVectorizer(stop_words = 'english', max_features = 5000)
    training_data_text_sparse = vec.fit_transform(df["sentence"])

    clf = LogisticRegression(C = 1, random_state = 42, class_weight = "balanced")
    clf.fit(training_data_text_sparse, df["label"])
    return vec, clf
