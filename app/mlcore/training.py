from data import *
from models import * 
from sklearn.metrics import classification_report
import pandas as pd
from joblib import dump



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
    vec, clf = Logistic_clf(train)
    evaluate_clf(vec, clf, test)

    sample_tweet = "In December alone , the members of the Lithuanian Brewers ' Association sold a total of 20.3 million liters of beer , an increase of 1.9 percent"
    # sample_tweet = pd.DataFrame(sample_tweet, columns='sentence')
    sample_tweet = pd.Series(sample_tweet)
    sample_tweet = vec.transform(sample_tweet)
    result = clf.predict(sample_tweet)
    print(result)
    dump(clf, 'app/mlcore/logistic_cls.joblib')
    dump(vec, 'app/mlcore/logistic_vec.joblib')