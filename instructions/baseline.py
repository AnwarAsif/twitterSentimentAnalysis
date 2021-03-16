import pandas
import re
import string
from timeit import default_timer as timer

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def read_data():
    """
    Read train/test data

    Returns:
    train -- Training dataframe, with 'sentence' and 'label'
    test -- Testing dataframe, with 'sentence' and 'label'
    """
    start = timer()
    train = pandas.read_csv("training.tsv", sep = "\t", encoding = "utf-8")
    test = pandas.read_csv("testing.tsv", sep = "\t", encoding = "utf-8")
    end = timer()
    print ("Number of data points - train: {}, test: {}".format(train.shape[0], test.shape[0]))
    print ("Time to read data: {} seconds".format(end - start))
    return train, test


def preprocess_data(df):
    """
    Preprocess data:
    -- strip spaces and lowercase
    -- get rid of numbers
    -- get rid of punctuations
    -- get rid of single letters

    Keyword arguments:
    df -- Input two column dataframe with 'sentence' and 'label'

    Returns:
    df -- With preprocessed 'sentence' column and 'label'
    """

    def preprocessor(sentence):
        sentence = sentence.strip().lower()
        sentence = re.sub(r"\d+", "", sentence)
        sentence = sentence.translate(sentence.maketrans(string.punctuation, ' '*len(string.punctuation)))
        sentence = " ".join([w for w in nltk.word_tokenize(sentence) if len(w) > 1])
        
        return sentence

    start = timer()
    df["sentence"] = df["sentence"].apply(preprocessor)
    end = timer()
    print ("Time to preprocess {} data points: {} seconds".format(df.shape[0], end - start))
    return df


def train_clf(df):
    """
    Train classifier
    -- logistic regression
    -- with TFIDF features

    Keyword arguments:
    df -- Input two column dataframe with 'sentence' and 'label'

    Returns:
    vec -- Fitted vectorizer
    clf -- Fitted classifier
    """
    start = timer()
    vec = TfidfVectorizer(stop_words = 'english', max_features = 5000)
    training_data_text_sparse = vec.fit_transform(df["sentence"])

    clf = LogisticRegression(C = 1, random_state = 42, class_weight = "balanced")
    clf.fit(training_data_text_sparse, df["label"])
    end = timer()
    print ("Time to train on {} data points: {} seconds".format(df.shape[0], end - start))

    return vec, clf


def evaluate_clf(vec, clf, df):
    """
    Calculate classification performance on test set
    
    Keyword arguments:
    vec -- Fitted vectorizer
    clf -- Fitted classifier
    df -- Two column test dataframe with 'sentence' and 'label'
    """
    start = timer()
    testing_data_text_sparse = vec.transform(df["sentence"])
    pred = clf.predict(testing_data_text_sparse)
    end = timer()
    print ("Time to apply classifier on {} data points: {} seconds".format(df.shape[0], end - start))

    print ("\nReport:")
    print (classification_report(df["label"], pred, digits = 4))


if __name__ == "__main__":
    train, test = read_data()
    train = preprocess_data(train)
    test = preprocess_data(test)
    vec, clf = train_clf(train)
    evaluate_clf(vec, clf, test)


    """
    Report:
              precision    recall  f1-score   support

    negative     0.5789    0.5000    0.5366       154
     neutral     0.7844    0.9277    0.8501       761
    positive     0.7725    0.5128    0.6164       351

    accuracy                         0.7607      1266
    macro avg     0.7120    0.6468    0.6677      1266
    weighted avg     0.7561    0.7607    0.7472      1266
    """